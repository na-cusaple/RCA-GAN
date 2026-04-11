"""
infer.py — Run RCA-GAN generator for image denoising inference.

Examples
--------
Single image:
    /usr/bin/python3 infer.py \
        --input datasets/val/noisy_images/example.jpg \
        --checkpoint checkpoints/ckpts/generator_epoch_0001.pth

Folder:
    /usr/bin/python3 infer.py \
        --input datasets/val/noisy_images \
        --checkpoint checkpoints/ckpts/generator_epoch_0001.pth \
        --output_dir outputs/denoised
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

from models import Generator


_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in _IMG_EXTENSIONS


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
        ]
    )


def _normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)


def preprocess_keep_aspect(
    img: Image.Image,
    image_size: int,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int, int]]:
    """
    Resize with aspect ratio preserved, then pad to a square canvas.

    Returns:
        Tensor in model input space and metadata for unpadding/restoring size.
    """
    orig_w, orig_h = img.size
    scale = min(image_size / max(orig_w, 1), image_size / max(orig_h, 1))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
    canvas = Image.new("RGB", (image_size, image_size))
    left = (image_size - new_w) // 2
    top = (image_size - new_h) // 2
    canvas.paste(resized, (left, top))

    x = transforms.ToTensor()(canvas)
    x = _normalize_tensor(x)
    meta = (orig_w, orig_h, new_w, new_h, left, top)
    return x.unsqueeze(0), meta


def restore_keep_aspect(
    out: Image.Image,
    meta: Tuple[int, int, int, int, int, int],
) -> Image.Image:
    orig_w, orig_h, new_w, new_h, left, top = meta
    cropped = out.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((orig_w, orig_h), resample=Image.BICUBIC)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    # x: (1, C, H, W) in [-1, 1]
    x = x.detach().cpu().squeeze(0)
    x = (x * 0.5 + 0.5).clamp(0, 1)  # [0, 1]
    x = transforms.ToPILImage()(x)
    return x


def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    candidates = sorted(ckpt_dir.glob("generator_epoch_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No generator checkpoints found in: {ckpt_dir}")
    return candidates[-1]


def collect_inputs(input_path: Path) -> List[Path]:
    if input_path.is_file() and is_image_file(input_path):
        return [input_path]
    if input_path.is_dir():
        return [p for p in sorted(input_path.iterdir()) if p.is_file() and is_image_file(p)]
    raise FileNotFoundError(f"Input not found or unsupported: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RCA-GAN denoising inference")
    parser.add_argument("--input", type=str, required=True, help="Input image path or folder path")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to generator .pth; if empty, use latest from checkpoints/ckpts")
    parser.add_argument("--output_dir", type=str, default="outputs/denoised", help="Where denoised images are saved")
    parser.add_argument("--image_size", type=int, default=256, help="Resize size used by model")
    parser.add_argument("--keep_aspect", action="store_true",
                        help="Keep aspect ratio by letterboxing to image_size and restoring original size on save")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=9)
    parser.add_argument("--device", type=str, default="", help="cuda / mps / cpu (auto if empty)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_latest_checkpoint(Path("checkpoints/ckpts"))

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = Generator(
        base_channels=args.base_channels,
        num_res_blocks=args.num_res_blocks,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    image_paths = collect_inputs(input_path)
    if len(image_paths) == 0:
        raise RuntimeError(f"No image found in: {input_path}")

    tfm = build_transform(args.image_size)

    print(f"[RCA-GAN] Device: {device}")
    print(f"[RCA-GAN] Checkpoint: {ckpt_path}")
    print(f"[RCA-GAN] Processing {len(image_paths)} image(s)...")

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")

            meta: Optional[Tuple[int, int, int, int, int, int]] = None
            if args.keep_aspect:
                x, meta = preprocess_keep_aspect(img, args.image_size)
            else:
                x = tfm(img).unsqueeze(0)

            x = x.to(device)
            y = model(x)
            out = tensor_to_pil(y)

            if meta is not None:
                out = restore_keep_aspect(out, meta)

            # Keep same filename for convenience
            out_path = output_dir / img_path.name
            out.save(out_path)
            print(f"[RCA-GAN] Saved -> {out_path}")

    print(f"[RCA-GAN] Done. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
