"""
Generator module for RCA-GAN.

Contains:
- ResBlock        : Residual block with two convolutional layers.
- CooperativeAttention : Region-based Cooperative Attention (RCA) mechanism.
- Generator       : Full encoder-decoder generator with residual + attention blocks.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Residual block: two 3×3 convolutions with Instance Normalization and ReLU.
    The skip connection adds the input directly to the output.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CooperativeAttention(nn.Module):
    """
    Region-based Cooperative Attention (RCA).

    Combines channel attention (squeeze-and-excitation style) and spatial
    attention, then adds the two attention maps cooperatively before applying
    them to the feature map.

    Args:
        channels  : Number of input/output feature channels.
        reduction : Channel reduction ratio for the channel-attention branch.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        # --- Channel attention ---
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels, bias=False),
        )

        # --- Spatial attention ---
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        # --- Cooperative fusion ---
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Channel attention
        avg_out = self.channel_fc(self.channel_avg_pool(x).view(B, C))
        max_out = self.channel_fc(self.channel_max_pool(x).view(B, C))
        channel_attn = self.sigmoid((avg_out + max_out).view(B, C, 1, 1))

        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        spatial_attn = self.sigmoid(
            self.spatial_conv(torch.cat([avg_spatial, max_spatial], dim=1))
        )

        # Cooperative combination: element-wise product of both attentions
        return x * channel_attn * spatial_attn


class Generator(nn.Module):
    """
    Encoder-decoder generator for image denoising/restoration.

    Architecture:
        Encoder → num_res_blocks × ResBlock → CooperativeAttention → Decoder

    Args:
        in_channels      : Channels of the input image  (default: 3 for RGB).
        out_channels     : Channels of the output image (default: 3 for RGB).
        base_channels    : Feature channels at the first encoder level.
        num_res_blocks   : Number of residual blocks in the bottleneck.
        attention_reduction: Reduction ratio for CooperativeAttention.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_res_blocks: int = 9,
        attention_reduction: int = 16,
    ):
        super().__init__()

        # --- Encoder ---
        self.encoder = nn.Sequential(
            # Layer 1: 7×7 conv, same spatial size
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
            # Layer 2: downsample ×2
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.ReLU(inplace=True),
            # Layer 3: downsample ×2
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.ReLU(inplace=True),
        )

        bottleneck_channels = base_channels * 4

        # --- Residual blocks ---
        res_blocks: List[nn.Module] = [
            ResBlock(bottleneck_channels) for _ in range(num_res_blocks)
        ]
        self.res_blocks = nn.Sequential(*res_blocks)

        # --- Cooperative Attention ---
        self.attention = CooperativeAttention(bottleneck_channels, reduction=attention_reduction)

        # --- Decoder ---
        self.decoder = nn.Sequential(
            # Upsample ×2
            nn.ConvTranspose2d(bottleneck_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.ReLU(inplace=True),
            # Upsample ×2
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
            # Output layer: 7×7 conv + Tanh
            nn.Conv2d(base_channels, out_channels, kernel_size=7, padding=3, bias=True),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        features = self.res_blocks(features)
        features = self.attention(features)
        out = self.decoder(features)
        return out
