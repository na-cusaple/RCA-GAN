from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models.discriminator import Discriminator
from models.generator import Generator
from utils.dataset import ImagePairDataset
from utils.losses import TextureLoss, VGGLoss, gradient_penalty

GP_WEIGHT = 10.0
PERCEPTUAL_WEIGHT = 0.1
TEXTURE_WEIGHT = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RCA-GAN denoising model")
    parser.add_argument("--data-root", type=str, default="datasets/train", help="Path to train dataset directory")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to store checkpoints")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImagePairDataset(
        noisy_dir=data_root / "noisy_images",
        clean_dir=data_root / "clean_images",
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    generator = Generator().to(args.device)
    discriminator = Discriminator().to(args.device)
    adv_criterion = torch.nn.BCEWithLogitsLoss()
    rec_criterion = torch.nn.L1Loss()
    perceptual_criterion = VGGLoss().to(args.device)
    texture_criterion = TextureLoss().to(args.device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        total_d_loss = 0.0
        total_g_loss = 0.0
        num_batches = 0
        for noisy, clean in dataloader:
            noisy = noisy.to(args.device)
            clean = clean.to(args.device)

            fake = generator(noisy)

            d_optimizer.zero_grad()
            real_logits = discriminator(clean)
            fake_logits = discriminator(fake.detach())
            real_targets = torch.ones_like(real_logits)
            fake_targets = torch.zeros_like(fake_logits)
            d_loss = adv_criterion(real_logits, real_targets) + adv_criterion(fake_logits, fake_targets)
            d_loss = d_loss + GP_WEIGHT * gradient_penalty(discriminator, clean, fake.detach())
            d_loss.backward()
            d_optimizer.step()
            total_d_loss += d_loss.item()

            g_optimizer.zero_grad()
            fake_logits = discriminator(fake)
            g_adv = adv_criterion(fake_logits, torch.ones_like(fake_logits))
            g_rec = rec_criterion(fake, clean)
            g_per = perceptual_criterion(fake, clean)
            g_tex = texture_criterion(fake, clean)
            g_loss = g_adv + g_rec + PERCEPTUAL_WEIGHT * g_per + TEXTURE_WEIGHT * g_tex
            g_loss.backward()
            g_optimizer.step()
            total_g_loss += g_loss.item()
            num_batches += 1

        safe_num_batches = max(num_batches, 1)
        mean_d_loss = total_d_loss / safe_num_batches
        mean_g_loss = total_g_loss / safe_num_batches
        print(f"Epoch [{epoch + 1}/{args.epochs}] d_loss={mean_d_loss:.4f} g_loss={mean_g_loss:.4f}")
        torch.save(
            {
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "g_optimizer_state_dict": g_optimizer.state_dict(),
                "d_optimizer_state_dict": d_optimizer.state_dict(),
            },
            output_dir / f"epoch_{epoch + 1}.pt",
        )


if __name__ == "__main__":
    main()
