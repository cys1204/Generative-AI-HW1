"""PatchGAN discriminator for CycleGAN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PatchDiscriminatorConfig:
    in_channels: int = 3
    filters: int = 64
    num_layers: int = 3


class PatchDiscriminator(nn.Module):
    def __init__(self, config: PatchDiscriminatorConfig):
        super().__init__()
        layers = [
            nn.Conv2d(config.in_channels, config.filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        in_filters = config.filters
        for i in range(1, config.num_layers):
            out_filters = min(in_filters * 2, 512)
            stride = 1 if i == config.num_layers - 1 else 2
            layers += [
                nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1, bias=False),
                nn.InstanceNorm2d(out_filters),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_filters = out_filters

        layers += [nn.Conv2d(in_filters, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
