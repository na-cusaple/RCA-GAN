"""
Dataset utilities for RCA-GAN.

Provides PairedImageDataset: a PyTorch Dataset that loads matched pairs of
noisy and clean images from two parallel directory trees.

Expected directory layout
-------------------------
root/
├── noisy_images/   (or any name passed to ``noisy_dir``)
│   ├── img_001.png
│   └── ...
└── clean_images/   (or any name passed to ``clean_dir``)
    ├── img_001.png   ← must match filenames in noisy_images/
    └── ...
"""

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Supported image extensions
_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in _IMG_EXTENSIONS


def default_transform(image_size: int = 256) -> Callable:
    """
    Return the default torchvision transform pipeline.

    Resizes both images to ``image_size × image_size``, converts to a
    float tensor, and normalises pixel values to the range [-1, 1].
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),                        # [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
        ]
    )


class PairedImageDataset(Dataset):
    """
    Dataset of paired (noisy, clean) images for supervised denoising training.

    The dataset discovers every image in ``noisy_dir`` and looks up the
    matching file (same filename) in ``clean_dir``.  Only pairs that exist in
    both directories are kept.

    Args:
        noisy_dir  : Directory containing noisy / degraded input images.
        clean_dir  : Directory containing corresponding clean / ground-truth images.
        transform  : A callable applied to *both* images (after loading as PIL
                     Images).  When ``None`` the default transform is used.
        image_size : Passed to ``default_transform`` if ``transform`` is ``None``.
    """

    def __init__(
        self,
        noisy_dir: str,
        clean_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 256,
    ):
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.transform = transform if transform is not None else default_transform(image_size)

        if not self.noisy_dir.is_dir():
            raise FileNotFoundError(f"Noisy image directory not found: {self.noisy_dir}")
        if not self.clean_dir.is_dir():
            raise FileNotFoundError(f"Clean image directory not found: {self.clean_dir}")

        self.pairs: List[Tuple[Path, Path]] = self._find_pairs()

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No matching image pairs found in:\n"
                f"  noisy: {self.noisy_dir}\n"
                f"  clean: {self.clean_dir}"
            )

    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        pairs: List[Tuple[Path, Path]] = []
        for noisy_path in sorted(self.noisy_dir.iterdir()):
            if not _is_image(noisy_path):
                continue
            clean_path = self.clean_dir / noisy_path.name
            if clean_path.exists():
                pairs.append((noisy_path, clean_path))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy_path, clean_path = self.pairs[index]
        noisy_img = Image.open(noisy_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")
        noisy_tensor = self.transform(noisy_img)
        clean_tensor = self.transform(clean_img)
        return noisy_tensor, clean_tensor


def build_dataloader(
    noisy_dir: str,
    clean_dir: str,
    batch_size: int = 4,
    image_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
) -> DataLoader:
    """
    Convenience factory that wraps :class:`PairedImageDataset` in a DataLoader.

    Args:
        noisy_dir   : Path to noisy images.
        clean_dir   : Path to clean images.
        batch_size  : Images per mini-batch.
        image_size  : Spatial size after resizing.
        shuffle     : Whether to shuffle the dataset each epoch.
        num_workers : Parallel data loading workers.
        transform   : Optional custom transform; falls back to default if None.

    Returns:
        A configured :class:`torch.utils.data.DataLoader`.
    """
    dataset = PairedImageDataset(
        noisy_dir=noisy_dir,
        clean_dir=clean_dir,
        transform=transform,
        image_size=image_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
