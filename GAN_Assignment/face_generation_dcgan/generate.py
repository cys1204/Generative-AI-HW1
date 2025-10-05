"""Utility for sampling images from a trained DCGAN generator."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import utils

from .model import DCGANConfig, Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate samples using a trained DCGAN generator.")
    parser.add_argument("checkpoint", type=Path, help="Path to the generator checkpoint.")
    parser.add_argument("--output", type=Path, default=Path("generated.png"), help="Output image file.")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    config = DCGANConfig(latent_dim=args.latent_dim)
    generator = Generator(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()

    with torch.no_grad():
        z = torch.randn(args.num_samples, args.latent_dim, device=device)
        samples = generator(z).cpu()
    grid = utils.make_grid(samples, nrow=int(args.num_samples ** 0.5), normalize=True, value_range=(-1, 1))
    utils.save_image(grid, args.output)
    print(f"Saved generated samples to {args.output}")


if __name__ == "__main__":
    main()
