"""
Discriminator module for RCA-GAN.

Uses a PatchGAN discriminator that classifies overlapping image patches as
real or fake, encouraging sharp local textures in the generated output.
"""

import torch
import torch.nn as nn
from typing import List


class Discriminator(nn.Module):
    """
    PatchGAN discriminator.

    Produces a feature map of patch-level real/fake predictions instead of a
    single scalar, which promotes high-frequency detail in the generator output.

    Args:
        in_channels  : Channels of the input image (default: 3 for RGB).
        base_channels: Feature channels at the first layer (default: 64).
        num_layers   : Number of strided downsampling layers (default: 3).
                       Effective receptive field ≈ 70×70 for num_layers=3.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()

        layers: List[nn.Module] = [
            # First layer — no normalisation, LeakyReLU
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        channels = base_channels
        for i in range(1, num_layers):
            prev_channels = channels
            channels = min(channels * 2, 512)
            stride = 2 if i < num_layers - 1 else 1
            layers += [
                nn.Conv2d(prev_channels, channels, kernel_size=4, stride=stride, padding=1, bias=False),
                nn.InstanceNorm2d(channels, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Output layer: single-channel prediction map (no sigmoid for WGAN-GP)
        layers.append(
            nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1, bias=True)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
