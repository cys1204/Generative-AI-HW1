"""DCGAN model definitions for face generation experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DCGANConfig:
    """Configuration for DCGAN generator and discriminator."""

    latent_dim: int = 128
    image_size: int = 64
    channels: int = 3
    feature_maps_gen: int = 64
    feature_maps_disc: int = 64


class Generator(nn.Module):
    """Generator network for DCGAN."""

    def __init__(self, config: DCGANConfig):
        super().__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.ConvTranspose2d(config.latent_dim, config.feature_maps_gen * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config.feature_maps_gen * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(config.feature_maps_gen * 8, config.feature_maps_gen * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.feature_maps_gen * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(config.feature_maps_gen * 4, config.feature_maps_gen * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.feature_maps_gen * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(config.feature_maps_gen * 2, config.feature_maps_gen, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.feature_maps_gen),
            nn.ReLU(True),
            nn.ConvTranspose2d(config.feature_maps_gen, config.channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(z.size(0), self.config.latent_dim, 1, 1)
        return self.model(z)


class Discriminator(nn.Module):
    """Discriminator network for DCGAN."""

    def __init__(self, config: DCGANConfig):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(config.channels, config.feature_maps_disc, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(config.feature_maps_disc, config.feature_maps_disc * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.feature_maps_disc * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(config.feature_maps_disc * 2, config.feature_maps_disc * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.feature_maps_disc * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(config.feature_maps_disc * 4, config.feature_maps_disc * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.feature_maps_disc * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(config.feature_maps_disc * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits.view(-1)
