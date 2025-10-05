"""Evaluation utilities for WGAN super-resolution models."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import SRGenerator, SRGeneratorConfig


def to_low_res(images: torch.Tensor, scale_factor: int) -> torch.Tensor:
    height = images.shape[-2] // scale_factor
    width = images.shape[-1] // scale_factor
    return torch.nn.functional.interpolate(
        images,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.nn.functional.mse_loss(pred, target)
    if mse.item() == 0:
        return torch.tensor(float("inf"), device=pred.device)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian_kernel(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = gauss / gauss.sum()
    return kernel[:, None] * kernel[None, :]


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    device = pred.device
    channel = pred.size(1)
    kernel = gaussian_kernel(window_size, sigma, device=device).expand(channel, 1, window_size, window_size)

    mu_pred = torch.nn.functional.conv2d(pred, kernel, padding=window_size // 2, groups=channel)
    mu_target = torch.nn.functional.conv2d(target, kernel, padding=window_size // 2, groups=channel)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = torch.nn.functional.conv2d(pred * pred, kernel, padding=window_size // 2, groups=channel) - mu_pred_sq
    sigma_target_sq = torch.nn.functional.conv2d(target * target, kernel, padding=window_size // 2, groups=channel) - mu_target_sq
    sigma_pred_target = torch.nn.functional.conv2d(pred * target, kernel, padding=window_size // 2, groups=channel) - mu_pred_target

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    ssim_map = numerator / denominator
    return ssim_map.mean()


def evaluate(generator: SRGenerator, dataloader: DataLoader, scale_factor: int, device: torch.device) -> tuple[float, float]:
    generator.eval()
    psnr_scores = []
    ssim_scores = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            low_res = to_low_res(images, scale_factor)
            recon = generator(low_res)
            recon_clamped = recon.clamp(-1, 1)
            target = images
            pred_norm = (recon_clamped + 1) / 2
            target_norm = (target + 1) / 2
            psnr_scores.append(psnr(pred_norm, target_norm).item())
            ssim_scores.append(ssim(pred_norm, target_norm).item())
    return float(sum(psnr_scores) / len(psnr_scores)), float(sum(ssim_scores) / len(ssim_scores))


def build_dataloader(data_root: Path, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(data_root, train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate super-resolution generator using PSNR and SSIM.")
    parser.add_argument("data_root", type=Path, help="Directory containing CIFAR-10 dataset.")
    parser.add_argument("checkpoint", type=Path, help="Path to generator checkpoint.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--scale-factor", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    generator = SRGenerator(SRGeneratorConfig(scale_factor=args.scale_factor)).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(state_dict)

    dataloader = build_dataloader(args.data_root, args.batch_size, args.num_workers)
    psnr_score, ssim_score = evaluate(generator, dataloader, args.scale_factor, device)
    print(f"PSNR: {psnr_score:.4f} dB")
    print(f"SSIM: {ssim_score:.4f}")


if __name__ == "__main__":
    main()
