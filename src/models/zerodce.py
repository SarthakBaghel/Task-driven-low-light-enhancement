#!/usr/bin/env python3
"""Zero-DCE style lightweight CNN for low-light image enhancement."""

from __future__ import annotations

import torch
from torch import nn


class ZeroDCE(nn.Module):
    """A compact curve-estimation network inspired by the original Zero-DCE."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        num_iterations: int = 8,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("ZeroDCE expects RGB input with 3 channels.")
        if hidden_channels <= 0:
            raise ValueError("'hidden_channels' must be greater than 0.")
        if num_iterations <= 0:
            raise ValueError("'num_iterations' must be greater than 0.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_iterations = num_iterations
        self.output_channels = in_channels * num_iterations

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv4 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv5 = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv6 = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv7 = nn.Conv2d(
            hidden_channels * 2,
            self.output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def predict_curve_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Predict curve parameter maps for each enhancement iteration."""
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        curve_maps = torch.tanh(self.conv7(torch.cat([x1, x6], dim=1)))

        batch_size, _, height, width = curve_maps.shape
        return curve_maps.view(
            batch_size,
            self.num_iterations,
            self.in_channels,
            height,
            width,
        )

    @staticmethod
    def apply_curve(x: torch.Tensor, curve_map: torch.Tensor) -> torch.Tensor:
        """Apply one Zero-DCE enhancement curve iteration."""
        return x + curve_map * (torch.square(x) - x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the enhanced image and the predicted curve maps."""
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                "Expected input of shape [batch, 3, height, width], "
                f"but received {tuple(x.shape)}."
            )

        working = torch.clamp(x, 0.0, 1.0)
        curve_maps = self.predict_curve_maps(working)

        enhanced = working
        for curve_map in torch.unbind(curve_maps, dim=1):
            enhanced = self.apply_curve(enhanced, curve_map)

        return torch.clamp(enhanced, 0.0, 1.0), curve_maps
