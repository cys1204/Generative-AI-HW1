"""ResNet-based generators used in CycleGAN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ResNetGeneratorConfig:
    in_channels: int = 3
    out_channels: int = 3
    filters: int = 64
    num_blocks: int = 6


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, config: ResNetGeneratorConfig):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(config.in_channels, config.filters, kernel_size=7),
            nn.InstanceNorm2d(config.filters),
            nn.ReLU(inplace=True),
        ]

        curr_filters = config.filters
        for _ in range(2):
            model += [
                nn.Conv2d(curr_filters, curr_filters * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(curr_filters * 2),
                nn.ReLU(inplace=True),
            ]
            curr_filters *= 2

        for _ in range(config.num_blocks):
            model.append(ResidualBlock(curr_filters))

        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_filters, curr_filters // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(curr_filters // 2),
                nn.ReLU(inplace=True),
            ]
            curr_filters //= 2

        model += [nn.ReflectionPad2d(3), nn.Conv2d(curr_filters, config.out_channels, kernel_size=7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
