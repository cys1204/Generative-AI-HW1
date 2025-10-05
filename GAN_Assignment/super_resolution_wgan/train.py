"""Training loop for WGAN-based super-resolution on CIFAR-10."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import autograd, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import Critic, CriticConfig, SRGenerator, SRGeneratorConfig


@dataclass
class SRTrainingConfig:
    data_root: Path
    output_dir: Path
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-4
    critic_iters: int = 5
    lambda_gp: float = 10.0
    scale_factor: int = 4
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_dataloader(config: SRTrainingConfig) -> DataLoader:
    hr_size = 32 * config.scale_factor
    transform = transforms.Compose(
        [
            transforms.Resize(hr_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(config.data_root, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)


def degrade(images: torch.Tensor, scale_factor: int) -> torch.Tensor:
    height = images.shape[-2] // scale_factor
    width = images.shape[-1] // scale_factor
    return torch.nn.functional.interpolate(
        images,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )


def gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=real.device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)
    critic_interpolated = critic(interpolated)
    gradients = autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def train(config: SRTrainingConfig) -> None:
    device = torch.device(config.device)
    dataloader = build_dataloader(config)

    generator = SRGenerator(SRGeneratorConfig(scale_factor=config.scale_factor)).to(device)
    critic = Critic(CriticConfig()).to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=config.lr, betas=(0.0, 0.9))
    optimizer_c = optim.Adam(critic.parameters(), lr=config.lr, betas=(0.0, 0.9))

    step = 0
    for epoch in range(1, config.epochs + 1):
        for real_hr, _ in dataloader:
            real_hr = real_hr.to(device)
            real_lr = degrade(real_hr, config.scale_factor)

            # Update critic
            for _ in range(config.critic_iters):
                fake_hr = generator(real_lr)
                optimizer_c.zero_grad(set_to_none=True)
                loss_real = critic(real_hr).mean()
                loss_fake = critic(fake_hr.detach()).mean()
                gp = gradient_penalty(critic, real_hr, fake_hr.detach())
                loss_c = loss_fake - loss_real + config.lambda_gp * gp
                loss_c.backward()
                optimizer_c.step()

            # Update generator
            fake_hr = generator(real_lr)
            optimizer_g.zero_grad(set_to_none=True)
            loss_g = -critic(fake_hr).mean()
            loss_recon = torch.nn.functional.l1_loss(fake_hr, real_hr)
            total_loss_g = loss_g + 0.01 * loss_recon
            total_loss_g.backward()
            optimizer_g.step()

            if step % 100 == 0:
                print(
                    f"Epoch [{epoch}/{config.epochs}] Step [{step}] "
                    f"Loss_C: {loss_c.item():.4f} Loss_G: {total_loss_g.item():.4f}"
                )
            step += 1

        torch.save(generator.state_dict(), config.output_dir / f"generator_epoch_{epoch:03d}.pt")
        torch.save(critic.state_dict(), config.output_dir / f"critic_epoch_{epoch:03d}.pt")

    torch.save(generator.state_dict(), config.output_dir / "generator_final.pt")
    torch.save(critic.state_dict(), config.output_dir / "critic_final.pt")


def parse_args() -> SRTrainingConfig:
    parser = argparse.ArgumentParser(description="Train WGAN-based super resolution on CIFAR-10.")
    parser.add_argument("data_root", type=Path, help="Directory to download CIFAR-10 dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/wgan_sr"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--critic-iters", type=int, default=5)
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--scale-factor", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return SRTrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        critic_iters=args.critic_iters,
        lambda_gp=args.lambda_gp,
        scale_factor=args.scale_factor,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train(cfg)
