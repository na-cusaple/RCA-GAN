"""
Loss functions for RCA-GAN.

Contains:
- VGGLoss        : Perceptual loss using pre-trained VGG-19 features.
- TextureLoss    : Gram-matrix based texture / style loss.
- gradient_penalty: WGAN-GP gradient penalty term.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGLoss(nn.Module):
    """
    Perceptual loss computed as the L1 distance between intermediate VGG-19
    feature activations of the generated and target images.

    The network is loaded with ImageNet-pretrained weights and kept frozen.

    Args:
        feature_layers: Indices of the VGG-19 feature layers to use.
                        Defaults to [3, 8, 17, 26] (relu1_2, relu2_2,
                        relu3_4, relu4_4).
        weights       : Relative weight for each layer's contribution.
                        Defaults to equal weighting.
        device        : Device to place the VGG model on.
    """

    # Normalisation statistics expected by VGG (ImageNet)
    _MEAN = torch.tensor([0.485, 0.456, 0.406])
    _STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(
        self,
        feature_layers: List[int] = None,
        weights: List[float] = None,
        device: torch.device = None,
    ):
        super().__init__()

        if feature_layers is None:
            feature_layers = [3, 8, 17, 26]
        if weights is None:
            weights = [1.0 / len(feature_layers)] * len(feature_layers)
        if len(weights) != len(feature_layers):
            raise ValueError(
                f"len(weights)={len(weights)} must equal len(feature_layers)={len(feature_layers)}"
            )

        self.feature_layers = feature_layers
        self.weights = weights

        # Build truncated VGG-19 feature extractor
        vgg_full = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        max_layer = max(feature_layers)
        self.vgg = nn.Sequential(*list(vgg_full.children())[: max_layer + 1])
        for param in self.vgg.parameters():
            param.requires_grad_(False)

        if device is not None:
            self.vgg = self.vgg.to(device)
            self._MEAN = self._MEAN.to(device)
            self._STD = self._STD.to(device)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Re-scale from [-1, 1] to [0, 1] then apply ImageNet normalisation."""
        x = (x + 1.0) / 2.0
        mean = self._MEAN.to(x.device).view(1, 3, 1, 1)
        std = self._STD.to(x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features

    def forward(
        self, generated: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        gen_feats = self._extract_features(self._preprocess(generated))
        tgt_feats = self._extract_features(self._preprocess(target))
        loss = torch.tensor(0.0, device=generated.device)
        for feat_gen, feat_tgt, w in zip(gen_feats, tgt_feats, self.weights):
            loss = loss + w * F.l1_loss(feat_gen, feat_tgt.detach())
        return loss


class TextureLoss(nn.Module):
    """
    Texture / style loss based on Gram matrices of VGG feature maps.

    The Gram matrix G[i,j] = (1/N) Σ_k F_ik * F_jk captures feature
    correlations that represent texture.  The loss is the L2 distance
    between Gram matrices of the generated and target images.

    Args:
        feature_layers: VGG-19 layer indices for texture extraction.
        weights       : Per-layer weights.
        device        : Device for the VGG model.
    """

    def __init__(
        self,
        feature_layers: List[int] = None,
        weights: List[float] = None,
        device: torch.device = None,
    ):
        super().__init__()
        if feature_layers is None:
            feature_layers = [3, 8, 17]
        if weights is None:
            weights = [1.0 / len(feature_layers)] * len(feature_layers)

        # Reuse VGGLoss for feature extraction
        self.vgg_loss = VGGLoss(
            feature_layers=feature_layers,
            weights=weights,
            device=device,
        )

    @staticmethod
    def _gram_matrix(feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, H * W)
        gram = torch.bmm(feat_flat, feat_flat.transpose(1, 2)) / (C * H * W)
        return gram

    def forward(
        self, generated: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        gen_feats = self.vgg_loss._extract_features(
            self.vgg_loss._preprocess(generated)
        )
        tgt_feats = self.vgg_loss._extract_features(
            self.vgg_loss._preprocess(target)
        )
        loss = torch.tensor(0.0, device=generated.device)
        for feat_gen, feat_tgt, w in zip(
            gen_feats, tgt_feats, self.vgg_loss.weights
        ):
            gram_gen = self._gram_matrix(feat_gen)
            gram_tgt = self._gram_matrix(feat_tgt.detach())
            loss = loss + w * F.mse_loss(gram_gen, gram_tgt)
        return loss


def gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Compute the WGAN-GP gradient penalty.

    Interpolates between real and fake samples and penalises the discriminator
    whenever the gradient norm deviates from 1.

    Args:
        discriminator: The discriminator network.
        real         : Batch of real images, shape (B, C, H, W).
        fake         : Batch of generated images, same shape as ``real``.
        device       : Computation device.
        lambda_gp    : Penalty coefficient (default: 10).

    Returns:
        Scalar gradient-penalty loss term.
    """
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1.0 - alpha) * fake.detach()).requires_grad_(True)

    d_interp = discriminator(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(B, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1.0) ** 2).mean()
    return penalty
