from __future__ import annotations

import threading

import torch
import torch.nn as nn
from torch import autograd
from torchvision import models


class VGGLoss(nn.Module):
    _cached_features_by_device: dict[str, nn.Module] = {}
    _cache_lock = threading.Lock()

    def __init__(self, device: str | torch.device = "cpu") -> None:
        super().__init__()
        device_key = str(device)
        with VGGLoss._cache_lock:
            if device_key not in VGGLoss._cached_features_by_device:
                features = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].eval()
                for param in features.parameters():
                    param.requires_grad = False
                VGGLoss._cached_features_by_device[device_key] = features.to(device)
        self.features = VGGLoss._cached_features_by_device[device_key]
        self.criterion = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(self.features(pred), self.features(target))


class TextureLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(_gram_matrix(pred), _gram_matrix(target))


def _gram_matrix(feature_maps: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feature_maps.shape
    features = feature_maps.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


def gradient_penalty(discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    d_interpolated = discriminator(interpolated)
    grad_outputs = torch.ones_like(d_interpolated, device=real.device)
    gradients = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()
