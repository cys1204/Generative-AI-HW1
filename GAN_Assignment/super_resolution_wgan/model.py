"""Model definitions for WGAN-based super resolution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SRGeneratorConfig:
    scale_factor: int = 4
    channels: int = 3
    num_residuals: int = 16
    feature_maps: int = 64


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int, scale: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SRGenerator(nn.Module):
    def __init__(self, config: SRGeneratorConfig):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(config.channels, config.feature_maps, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        self.residuals = nn.Sequential(*[ResidualBlock(config.feature_maps) for _ in range(config.num_residuals)])
        self.conv2 = nn.Sequential(
            nn.Conv2d(config.feature_maps, config.feature_maps, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.feature_maps),
        )

        upsample_layers = []
        scale = config.scale_factor
        while scale > 1:
            upsample_layers.append(UpsampleBlock(config.feature_maps, 2))
            scale //= 2
        self.upsample = nn.Sequential(*upsample_layers)

        self.conv3 = nn.Conv2d(config.feature_maps, config.channels, kernel_size=9, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        initial = self.conv1(x)
        residual = self.residuals(initial)
        residual = self.conv2(residual)
        residual = residual + initial
        upsampled = self.upsample(residual)
        out = self.conv3(upsampled)
        return torch.tanh(out)


@dataclass
class CriticConfig:
    channels: int = 3
    feature_maps: int = 64


class Critic(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        layers = [
            nn.Conv2d(config.channels, config.feature_maps, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_channels = config.feature_maps
        for out_channels in [config.feature_maps * 2, config.feature_maps * 4, config.feature_maps * 8]:
            layers += [
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, 1, 4, 1, 0))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).view(-1)
