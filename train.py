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
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import Generator, Discriminator
from utils.losses import VGGLoss, TextureLoss, gradient_penalty
from utils.dataset import build_dataloader


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

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

    # ---- Training loop -----------------------------------------------------
    global_step = 0
    for epoch in range(1, args.epochs + 1):
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
            print(f"[RCA-GAN] Checkpoints saved at epoch {epoch}")

    print("[RCA-GAN] Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(parse_args())
