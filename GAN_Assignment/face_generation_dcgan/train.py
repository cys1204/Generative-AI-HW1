"""Training script for DCGAN-based face generation on CelebA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from .model import DCGANConfig, Discriminator, Generator


@dataclass
class TrainingConfig:
    data_root: Path
    output_dir: Path
    epochs: int = 50
    batch_size: int = 128
    lr: float = 2e-4
    beta1: float = 0.5
    latent_dim: int = 128
    image_size: int = 64
    save_every: int = 5
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_dataloader(config: TrainingConfig) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.ImageFolder(config.data_root, transform=transform)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)


def save_samples(generator: Generator, latent_dim: int, output_dir: Path, step: int, device: torch.device) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        z = torch.randn(64, latent_dim, device=device)
        samples = generator(z).detach().cpu()
    grid = utils.make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
    utils.save_image(grid, output_dir / f"samples_step_{step:06d}.png")
    generator.train()


def train(config: TrainingConfig) -> None:
    device = torch.device(config.device)
    dataloader = build_dataloader(config)

    model_config = DCGANConfig(latent_dim=config.latent_dim, image_size=config.image_size)
    generator = Generator(model_config).to(device)
    discriminator = Discriminator(model_config).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    step = 0
    for epoch in range(1, config.epochs + 1):
        for images, _ in dataloader:
            images = images.to(device)
            batch_size = images.size(0)

            # Train discriminator
            noise = torch.randn(batch_size, config.latent_dim, device=device)
            fake_images = generator(noise)

            optimizer_d.zero_grad(set_to_none=True)
            logits_real = discriminator(images)
            logits_fake = discriminator(fake_images.detach())

            loss_real = criterion(logits_real, torch.ones_like(logits_real))
            loss_fake = criterion(logits_fake, torch.zeros_like(logits_fake))
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # Train generator
            optimizer_g.zero_grad(set_to_none=True)
            logits_fake_for_g = discriminator(fake_images)
            loss_g = criterion(logits_fake_for_g, torch.ones_like(logits_fake_for_g))
            loss_g.backward()
            optimizer_g.step()

            if step % 100 == 0:
                print(
                    f"Epoch [{epoch}/{config.epochs}] Step [{step}] "
                    f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}"
                )

            if step % (config.save_every * len(dataloader)) == 0:
                save_samples(generator, config.latent_dim, config.output_dir / "samples", step, device)

            step += 1

        torch.save(generator.state_dict(), config.output_dir / f"generator_epoch_{epoch:03d}.pt")
        torch.save(discriminator.state_dict(), config.output_dir / f"discriminator_epoch_{epoch:03d}.pt")
        save_samples(generator, config.latent_dim, config.output_dir / "samples", step, device)

    save_samples(generator, config.latent_dim, config.output_dir / "samples", step, device)
    torch.save(generator.state_dict(), config.output_dir / "generator_final.pt")
    torch.save(discriminator.state_dict(), config.output_dir / "discriminator_final.pt")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train DCGAN for face generation.")
    parser.add_argument("data_root", type=Path, help="Path to CelebA root folder (ImageFolder layout).")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/dcgan"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return TrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        save_every=args.save_every,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train(cfg)
