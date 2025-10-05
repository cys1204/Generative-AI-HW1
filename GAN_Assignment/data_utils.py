"""Shared dataset utilities for GAN assignment projects."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS: Sequence[str] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tiff",
    ".webp",
)


def _iter_image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
            yield path


class UnlabeledImageDataset(Dataset):
    """Dataset that loads every image under a directory without requiring labels."""

    def __init__(self, root: Path, transform=None) -> None:  # type: ignore[override]
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Path] = list(_iter_image_files(self.root))
        if not self.samples:
            raise ValueError(f"No image files found under {self.root}")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int):  # type: ignore[override]
        path = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0

