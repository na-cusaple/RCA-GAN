from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


class ImagePairDataset(Dataset):
    def __init__(self, noisy_dir: str | Path, clean_dir: str | Path, transform: Callable | None = None) -> None:
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.transform = transform
        self.noisy_images = sorted([p for p in self.noisy_dir.iterdir() if p.is_file()])
        self.clean_images = sorted([p for p in self.clean_dir.iterdir() if p.is_file()])
        if len(self.noisy_images) != len(self.clean_images):
            raise ValueError("noisy_images and clean_images must contain the same number of files.")

    def __len__(self) -> int:
        return len(self.noisy_images)

    def __getitem__(self, idx: int):
        noisy = Image.open(self.noisy_images[idx]).convert("RGB")
        clean = Image.open(self.clean_images[idx]).convert("RGB")
        if self.transform is not None:
            noisy = self.transform(noisy)
            clean = self.transform(clean)
        return noisy, clean
