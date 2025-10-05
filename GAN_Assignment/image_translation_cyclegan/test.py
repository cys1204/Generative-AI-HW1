"""Inference utilities for CycleGAN image translation."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from .model.generator import ResNetGenerator, ResNetGeneratorConfig


def build_loader(data_root: Path, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.ImageFolder(data_root, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def save_translations(generator: ResNetGenerator, dataloader: DataLoader, output_dir: Path, device: torch.device) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    for idx, (images, _) in enumerate(dataloader):
        images = images.to(device)
        with torch.no_grad():
            translated = generator(images)
        translated = translated.cpu()
        for i, sample in enumerate(translated):
            utils.save_image(sample, output_dir / f"image_{idx * dataloader.batch_size + i:06d}.png", normalize=True, value_range=(-1, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CycleGAN translation on a folder of images.")
    parser.add_argument("data_root", type=Path, help="Directory with images to translate (ImageFolder layout).")
    parser.add_argument("checkpoint", type=Path, help="Path to generator checkpoint.")
    parser.add_argument("--direction", choices=["A2B", "B2A"], default="A2B")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cyclegan"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    generator = ResNetGenerator(ResNetGeneratorConfig()).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(state_dict)

    dataloader = build_loader(args.data_root, args.image_size, args.batch_size, args.num_workers)
    save_translations(generator, dataloader, args.output_dir / args.direction, device)
    print(f"Saved translated images to {args.output_dir / args.direction}")


if __name__ == "__main__":
    main()
