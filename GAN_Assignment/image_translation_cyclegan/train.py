"""Training script for CycleGAN on horse2zebra or similar datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from .model.discriminator import PatchDiscriminator, PatchDiscriminatorConfig
from .model.generator import ResNetGenerator, ResNetGeneratorConfig
from ..data_utils import UnlabeledImageDataset


@dataclass
class CycleGANTrainingConfig:
    data_root_a: Path
    data_root_b: Path
    output_dir: Path
    batch_size: int = 4
    epochs: int = 200
    lr: float = 2e-4
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_loader(data_root: Path, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.12), transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = UnlabeledImageDataset(data_root, transform=transform)
    print(f"Loaded {len(dataset)} images from {data_root}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def cycle_gan_loss(
    generator_ab: ResNetGenerator,
    generator_ba: ResNetGenerator,
    discriminator_a: PatchDiscriminator,
    discriminator_b: PatchDiscriminator,
    real_a: torch.Tensor,
    real_b: torch.Tensor,
    criterion_gan: nn.Module,
    criterion_l1: nn.Module,
    lambda_cycle: float,
    lambda_identity: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Identity loss
    same_b = generator_ab(real_b)
    loss_identity_b = criterion_l1(same_b, real_b) * lambda_cycle * lambda_identity
    same_a = generator_ba(real_a)
    loss_identity_a = criterion_l1(same_a, real_a) * lambda_cycle * lambda_identity

    # GAN loss
    fake_b = generator_ab(real_a)
    pred_fake_b = discriminator_b(fake_b)
    loss_gan_ab = criterion_gan(pred_fake_b, torch.ones_like(pred_fake_b))

    fake_a = generator_ba(real_b)
    pred_fake_a = discriminator_a(fake_a)
    loss_gan_ba = criterion_gan(pred_fake_a, torch.ones_like(pred_fake_a))

    # Cycle consistency
    recov_a = generator_ba(fake_b)
    recov_b = generator_ab(fake_a)
    loss_cycle_a = criterion_l1(recov_a, real_a) * lambda_cycle
    loss_cycle_b = criterion_l1(recov_b, real_b) * lambda_cycle

    loss_g = loss_identity_a + loss_identity_b + loss_gan_ab + loss_gan_ba + loss_cycle_a + loss_cycle_b
    return loss_g, fake_a.detach(), fake_b.detach(), recov_a.detach()


def train(config: CycleGANTrainingConfig) -> None:
    device = torch.device(config.device)

    loader_a = build_loader(config.data_root_a, image_size=256, batch_size=config.batch_size, num_workers=config.num_workers)
    loader_b = build_loader(config.data_root_b, image_size=256, batch_size=config.batch_size, num_workers=config.num_workers)

    gen_config = ResNetGeneratorConfig()
    generator_ab = ResNetGenerator(gen_config).to(device)
    generator_ba = ResNetGenerator(gen_config).to(device)

    disc_config = PatchDiscriminatorConfig()
    discriminator_a = PatchDiscriminator(disc_config).to(device)
    discriminator_b = PatchDiscriminator(disc_config).to(device)

    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    optimizer_g = optim.Adam(
        list(generator_ab.parameters()) + list(generator_ba.parameters()),
        lr=config.lr,
        betas=(0.5, 0.999),
    )
    optimizer_d = optim.Adam(
        list(discriminator_a.parameters()) + list(discriminator_b.parameters()),
        lr=config.lr,
        betas=(0.5, 0.999),
    )

    loader_b_iter = cycle(loader_b)
    step = 0
    for epoch in range(1, config.epochs + 1):
        for real_a, _ in loader_a:
            real_b, _ = next(loader_b_iter)
            real_a = real_a.to(device)
            real_b = real_b.to(device)

            loss_g, fake_a, fake_b, _ = cycle_gan_loss(
                generator_ab,
                generator_ba,
                discriminator_a,
                discriminator_b,
                real_a,
                real_b,
                criterion_gan,
                criterion_l1,
                config.lambda_cycle,
                config.lambda_identity,
            )

            optimizer_g.zero_grad(set_to_none=True)
            loss_g.backward()
            optimizer_g.step()

            optimizer_d.zero_grad(set_to_none=True)

            pred_real_a = discriminator_a(real_a)
            pred_fake_a = discriminator_a(fake_a)
            loss_d_a = 0.5 * (criterion_gan(pred_real_a, torch.ones_like(pred_real_a)) + criterion_gan(pred_fake_a, torch.zeros_like(pred_fake_a)))

            pred_real_b = discriminator_b(real_b)
            pred_fake_b = discriminator_b(fake_b)
            loss_d_b = 0.5 * (criterion_gan(pred_real_b, torch.ones_like(pred_real_b)) + criterion_gan(pred_fake_b, torch.zeros_like(pred_fake_b)))

            loss_d = loss_d_a + loss_d_b
            loss_d.backward()
            optimizer_d.step()

            if step % 100 == 0:
                print(
                    f"Epoch [{epoch}/{config.epochs}] Step [{step}] "
                    f"Loss_G: {loss_g.item():.4f} Loss_D: {loss_d.item():.4f}"
                )

            step += 1

        torch.save(generator_ab.state_dict(), config.output_dir / f"generator_ab_epoch_{epoch:03d}.pt")
        torch.save(generator_ba.state_dict(), config.output_dir / f"generator_ba_epoch_{epoch:03d}.pt")

    torch.save(generator_ab.state_dict(), config.output_dir / "generator_ab_final.pt")
    torch.save(generator_ba.state_dict(), config.output_dir / "generator_ba_final.pt")


def parse_args() -> CycleGANTrainingConfig:
    parser = argparse.ArgumentParser(description="Train CycleGAN for unpaired image translation.")
    parser.add_argument(
        "data_root",
        type=Path,
        help="Root directory containing trainA/trainB folders (e.g., data/horse2zebra).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/cyclegan"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_root_a = args.data_root / "trainA"
    data_root_b = args.data_root / "trainB"
    return CycleGANTrainingConfig(
        data_root_a=data_root_a,
        data_root_b=data_root_b,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train(cfg)
