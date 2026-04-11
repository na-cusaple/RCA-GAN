"""
train.py — Main training loop for RCA-GAN.

Usage
-----
    python train.py [OPTIONS]

Key options (all have sensible defaults):
    --data_root   Root of the dataset directory   (default: datasets/)
    --epochs      Number of training epochs        (default: 100)
    --batch_size  Images per mini-batch            (default: 4)
    --image_size  Spatial size after resizing      (default: 256)
    --lr          Learning rate                    (default: 2e-4)
    --lambda_adv  Adversarial loss weight          (default: 1.0)
    --lambda_pixel Pixel-wise (L1) loss weight     (default: 100.0)
    --lambda_perc Perceptual (VGG) loss weight     (default: 10.0)
    --lambda_tex  Texture loss weight              (default: 5.0)
    --lambda_gp   WGAN-GP gradient penalty weight  (default: 10.0)
    --n_critic    Discriminator updates per G step (default: 5)
    --save_dir    Directory to save checkpoints    (default: checkpoints/)
    --log_interval Steps between console logs      (default: 10)
    --device      cuda / cpu                       (default: auto-detect)
"""

import argparse
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.utils import save_image

from models import Generator, Discriminator
from utils.losses import VGGLoss, TextureLoss, gradient_penalty
from utils.dataset import build_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RCA-GAN for image denoising / restoration"
    )
    # Paths
    parser.add_argument("--data_root", type=str, default="datasets",
                        help="Root directory containing train/ and val/ sub-folders")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints and sample images")
    # Training hyper-parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam β₁")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam β₂")
    # Loss weights
    parser.add_argument("--lambda_adv", type=float, default=1.0,
                        help="Weight for adversarial loss (WGAN-GP)")
    parser.add_argument("--lambda_pixel", type=float, default=100.0,
                        help="Weight for pixel-wise L1 loss")
    parser.add_argument("--lambda_perc", type=float, default=10.0,
                        help="Weight for VGG perceptual loss")
    parser.add_argument("--lambda_tex", type=float, default=5.0,
                        help="Weight for texture (Gram) loss")
    parser.add_argument("--lambda_gp", type=float, default=10.0,
                        help="Gradient-penalty weight (WGAN-GP)")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="Discriminator updates per generator update")
    # Misc
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Steps between console log prints")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Epochs between checkpoint saves")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="",
                        help="Force device (cuda/cpu). Auto-detect when empty.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest saved checkpoint in save_dir/ckpts")
    parser.add_argument("--resume_state", type=str, default="",
                        help="Path to a full training state checkpoint (.pth). Overrides --resume search.")
    parser.add_argument("--auto_generate_noisy", action="store_true",
                        help="Automatically create noisy train images from clean images when pairs are missing")
    parser.add_argument("--auto_generate_val_noisy", action="store_true",
                        help="Automatically create noisy validation images from val clean images")
    parser.add_argument("--noise_sigma", type=float, default=25.0,
                        help="Std-dev of Gaussian noise in pixel value space [0..255]")
    parser.add_argument("--val_noise_sigma", type=float, default=-1.0,
                        help="Std-dev for validation noisy generation; if < 0, reuse --noise_sigma")
    parser.add_argument("--overwrite_noisy", action="store_true",
                        help="Overwrite existing noisy images when auto generating")
    parser.add_argument("--noise_seed", type=int, default=42,
                        help="Random seed used for auto noisy-image generation")
    # Generator / discriminator hyper-params
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=9)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def weights_init(module: nn.Module) -> None:
    """Apply standard GAN weight initialisation (N(0, 0.02))."""
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif classname.find("InstanceNorm2d") != -1:
        if module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)


_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in _IMG_EXTENSIONS


def _extract_epoch_from_name(name: str) -> int:
    match = re.search(r"epoch_(\d+)", name)
    if match is None:
        return -1
    return int(match.group(1))


def _find_latest_weights(ckpt_dir: Path) -> tuple[Optional[Path], Optional[Path], int]:
    g_files = sorted(ckpt_dir.glob("generator_epoch_*.pth"), key=lambda p: _extract_epoch_from_name(p.name))
    d_files = sorted(ckpt_dir.glob("discriminator_epoch_*.pth"), key=lambda p: _extract_epoch_from_name(p.name))

    if not g_files or not d_files:
        return None, None, -1

    g_by_epoch = {_extract_epoch_from_name(p.name): p for p in g_files}
    d_by_epoch = {_extract_epoch_from_name(p.name): p for p in d_files}
    common_epochs = sorted(set(g_by_epoch.keys()) & set(d_by_epoch.keys()))
    if not common_epochs:
        return None, None, -1

    latest_epoch = common_epochs[-1]
    return g_by_epoch[latest_epoch], d_by_epoch[latest_epoch], latest_epoch


def generate_noisy_from_clean(
    clean_dir: str,
    noisy_dir: str,
    sigma: float = 25.0,
    overwrite: bool = False,
    seed: int = 42,
) -> int:
    """
    Create noisy images from clean images using additive Gaussian noise.

    The generated noisy image keeps the same filename as the clean image,
    which is required by ``PairedImageDataset``.

    Returns:
        Number of noisy images generated.
    """
    clean_root = Path(clean_dir)
    noisy_root = Path(noisy_dir)
    noisy_root.mkdir(parents=True, exist_ok=True)

    if not clean_root.is_dir():
        raise FileNotFoundError(f"Clean directory not found: {clean_root}")

    rng = np.random.default_rng(seed)
    generated = 0
    noise_scale = sigma / 255.0

    for clean_path in sorted(clean_root.iterdir()):
        if not clean_path.is_file() or not _is_image_file(clean_path):
            continue

        noisy_path = noisy_root / clean_path.name
        if noisy_path.exists() and not overwrite:
            continue

        img = Image.open(clean_path).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0

        noise = rng.normal(loc=0.0, scale=noise_scale, size=arr.shape).astype(np.float32)
        noisy_arr = np.clip(arr + noise, 0.0, 1.0)
        noisy_img = Image.fromarray((noisy_arr * 255.0).astype(np.uint8))
        noisy_img.save(noisy_path)
        generated += 1

    return generated


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # ---- Device -----------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RCA-GAN] Using device: {device}")

    # ---- Directories -------------------------------------------------------
    save_dir = Path(args.save_dir)
    sample_dir = save_dir / "samples"
    ckpt_dir = save_dir / "ckpts"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data --------------------------------------------------------------
    train_noisy = os.path.join(args.data_root, "train", "noisy_images")
    train_clean = os.path.join(args.data_root, "train", "clean_images")

    if args.auto_generate_noisy:
        generated = generate_noisy_from_clean(
            clean_dir=train_clean,
            noisy_dir=train_noisy,
            sigma=args.noise_sigma,
            overwrite=args.overwrite_noisy,
            seed=args.noise_seed,
        )
        print(
            f"[RCA-GAN] Auto noisy generation: created {generated} image(s) "
            f"in {train_noisy}"
        )

    train_loader: DataLoader = build_dataloader(
        noisy_dir=train_noisy,
        clean_dir=train_clean,
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Validation loader (optional; skip if val images are absent)
    val_noisy = os.path.join(args.data_root, "val", "noisy_images")
    val_clean = os.path.join(args.data_root, "val", "clean_images")

    if args.auto_generate_val_noisy:
        val_sigma = args.noise_sigma if args.val_noise_sigma < 0 else args.val_noise_sigma
        generated_val = generate_noisy_from_clean(
            clean_dir=val_clean,
            noisy_dir=val_noisy,
            sigma=val_sigma,
            overwrite=args.overwrite_noisy,
            seed=args.noise_seed,
        )
        print(
            f"[RCA-GAN] Auto val noisy generation: created {generated_val} image(s) "
            f"in {val_noisy}"
        )

    val_loader: Optional[DataLoader] = None
    try:
        val_loader = build_dataloader(
            noisy_dir=val_noisy,
            clean_dir=val_clean,
            batch_size=1,
            image_size=args.image_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        print(f"[RCA-GAN] Validation set: {len(val_loader.dataset)} pairs")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[RCA-GAN] No validation data: {exc}")

    print(f"[RCA-GAN] Training set : {len(train_loader.dataset)} pairs")

    # ---- Models ------------------------------------------------------------
    G = Generator(
        base_channels=args.base_channels,
        num_res_blocks=args.num_res_blocks,
    ).to(device)
    D = Discriminator(base_channels=64).to(device)

    G.apply(weights_init)
    D.apply(weights_init)

    # ---- Losses ------------------------------------------------------------
    criterion_pixel = nn.L1Loss().to(device)
    criterion_vgg = VGGLoss(device=device).to(device)
    criterion_tex = TextureLoss(device=device).to(device)

    # ---- Optimisers --------------------------------------------------------
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Learning rate schedulers: linear decay in the second half of training
    def lr_lambda(epoch: int) -> float:
        decay_start = args.epochs // 2
        if epoch < decay_start:
            return 1.0
        return 1.0 - (epoch - decay_start) / max(args.epochs - decay_start, 1)

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    # ---- Resume (optional) -------------------------------------------------
    start_epoch = 1
    global_step = 0
    state_path = ckpt_dir / "training_state_latest.pth"

    requested_resume = bool(args.resume or args.resume_state)
    if requested_resume:
        loaded = False
        if args.resume_state:
            candidate = Path(args.resume_state)
            if not candidate.is_file():
                raise FileNotFoundError(f"--resume_state not found: {candidate}")
            state_path = candidate

        if state_path.is_file():
            state = torch.load(state_path, map_location=device)
            G.load_state_dict(state["generator"])
            D.load_state_dict(state["discriminator"])
            opt_G.load_state_dict(state["optimizer_g"])
            opt_D.load_state_dict(state["optimizer_d"])
            scheduler_G.load_state_dict(state["scheduler_g"])
            scheduler_D.load_state_dict(state["scheduler_d"])
            last_epoch = int(state["epoch"])
            global_step = int(state.get("global_step", 0))
            start_epoch = last_epoch + 1
            loaded = True
            print(f"[RCA-GAN] Resumed full training state from {state_path} at epoch {last_epoch}")

        if not loaded:
            g_path, d_path, last_epoch = _find_latest_weights(ckpt_dir)
            if g_path is None or d_path is None:
                raise FileNotFoundError(
                    f"No checkpoints found to resume in {ckpt_dir}. "
                    f"Expected training_state_latest.pth or matching generator/discriminator_epoch_*.pth"
                )
            G.load_state_dict(torch.load(g_path, map_location=device))
            D.load_state_dict(torch.load(d_path, map_location=device))
            start_epoch = last_epoch + 1
            global_step = max(last_epoch, 0) * len(train_loader)
            print(
                f"[RCA-GAN] Resumed model weights from epoch {last_epoch} "
                f"({g_path.name}, {d_path.name}). "
                "Optimizers/schedulers were reset."
            )

    # ---- Training loop -----------------------------------------------------
    if start_epoch > args.epochs:
        print(
            f"[RCA-GAN] Resume epoch ({start_epoch - 1}) is already >= target epochs ({args.epochs}). "
            "Nothing to train."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        G.train()
        D.train()

        for step, (noisy, clean) in enumerate(train_loader, start=1):
            noisy = noisy.to(device)
            clean = clean.to(device)

            # ================================================================
            # (1) Train Discriminator  (n_critic times per G update)
            # ================================================================
            for _ in range(args.n_critic):
                opt_D.zero_grad()
                fake = G(noisy).detach()

                # WGAN losses
                loss_D_real = -D(clean).mean()
                loss_D_fake = D(fake).mean()

                # Gradient penalty
                gp = gradient_penalty(D, clean, fake, device, args.lambda_gp)

                loss_D = loss_D_real + loss_D_fake + gp
                loss_D.backward()
                opt_D.step()

            # ================================================================
            # (2) Train Generator
            # ================================================================
            opt_G.zero_grad()
            fake = G(noisy)

            # Adversarial loss (WGAN)
            loss_adv = -D(fake).mean() * args.lambda_adv

            # Pixel-wise L1
            loss_pixel = criterion_pixel(fake, clean) * args.lambda_pixel

            # Perceptual (VGG) loss
            loss_perc = criterion_vgg(fake, clean) * args.lambda_perc

            # Texture loss
            loss_tex = criterion_tex(fake, clean) * args.lambda_tex

            loss_G = loss_adv + loss_pixel + loss_perc + loss_tex
            loss_G.backward()
            opt_G.step()

            global_step += 1

            if global_step % args.log_interval == 0:
                print(
                    f"[Epoch {epoch:>4}/{args.epochs}] "
                    f"[Step {step:>5}] "
                    f"D: {loss_D.item():.4f}  "
                    f"G: {loss_G.item():.4f}  "
                    f"adv: {loss_adv.item():.4f}  "
                    f"pixel: {loss_pixel.item():.4f}  "
                    f"perc: {loss_perc.item():.4f}  "
                    f"tex: {loss_tex.item():.4f}"
                )

        scheduler_G.step()
        scheduler_D.step()

        # ---- Validation ----------------------------------------------------
        if val_loader is not None:
            G.eval()
            with torch.no_grad():
                for i, (val_noisy, val_clean) in enumerate(val_loader):
                    val_noisy = val_noisy.to(device)
                    val_clean = val_clean.to(device)
                    val_fake = G(val_noisy)
                    # Save a side-by-side comparison for the first batch
                    if i == 0:
                        sample_path = sample_dir / f"epoch_{epoch:04d}.png"
                        save_image(
                            torch.cat([val_noisy, val_fake, val_clean], dim=3),
                            sample_path,
                            normalize=True,
                            value_range=(-1, 1),
                        )
                        print(f"[RCA-GAN] Saved sample → {sample_path}")

        # ---- Checkpoint ----------------------------------------------------
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            torch.save(G.state_dict(), ckpt_dir / f"generator_epoch_{epoch:04d}.pth")
            torch.save(D.state_dict(), ckpt_dir / f"discriminator_epoch_{epoch:04d}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "generator": G.state_dict(),
                    "discriminator": D.state_dict(),
                    "optimizer_g": opt_G.state_dict(),
                    "optimizer_d": opt_D.state_dict(),
                    "scheduler_g": scheduler_G.state_dict(),
                    "scheduler_d": scheduler_D.state_dict(),
                },
                ckpt_dir / "training_state_latest.pth",
            )
            print(f"[RCA-GAN] Checkpoints saved at epoch {epoch}")

    print("[RCA-GAN] Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(parse_args())
