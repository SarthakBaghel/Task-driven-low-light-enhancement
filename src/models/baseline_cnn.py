#!/usr/bin/env python3
"""Lightweight CNN baseline for binary eye-state classification."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """A simple convolution block used by the baseline CNN."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class BaselineCNN(nn.Module):
    """A compact 3-block CNN for open-vs-closed eye classification."""

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return self.classifier(x)
