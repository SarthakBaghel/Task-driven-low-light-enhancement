#!/usr/bin/env python3
"""Small tests and usage examples for Zero-DCE enhancement losses."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.enhancement_losses import (
    color_constancy_loss,
    enhancement_loss,
    exposure_control_loss,
    illumination_smoothness_loss,
    spatial_consistency_loss,
)


def build_sample_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create small dummy tensors that mimic Zero-DCE training inputs."""
    torch.manual_seed(7)
    input_image = torch.rand(2, 3, 32, 32)
    enhanced_image = torch.clamp(input_image * 1.15 + 0.05, 0.0, 1.0)
    curve_maps = torch.rand(2, 8, 3, 32, 32) * 0.2 - 0.1
    return input_image, enhanced_image, curve_maps


def example_usage() -> None:
    """Print a tiny example showing how to call each loss function."""
    input_image, enhanced_image, curve_maps = build_sample_tensors()

    exp_loss = exposure_control_loss(
        enhanced_image,
        patch_size=8,
        target_mean=0.6,
    )
    col_loss = color_constancy_loss(
        enhanced_image,
        use_sqrt=True,
    )
    smooth_loss = illumination_smoothness_loss(
        curve_maps,
        power=2.0,
    )
    spa_loss = spatial_consistency_loss(
        input_image,
        enhanced_image,
        pool_size=4,
    )
    combined = enhancement_loss(
        enhanced_image,
        curve_maps,
        input_image=input_image,
        exposure_weight=10.0,
        color_weight=5.0,
        smoothness_weight=200.0,
        spatial_weight=1.0,
        exposure_patch_size=8,
        exposure_target_mean=0.6,
        color_use_sqrt=True,
        smoothness_power=2.0,
        spatial_pool_size=4,
        use_spatial_consistency=True,
    )

    print("Example Zero-DCE losses")
    print(f"Exposure control loss: {exp_loss.item():.6f}")
    print(f"Color constancy loss: {col_loss.item():.6f}")
    print(f"Illumination smoothness loss: {smooth_loss.item():.6f}")
    print(f"Spatial consistency loss: {spa_loss.item():.6f}")
    print(f"Combined enhancement loss: {combined['total'].item():.6f}")


class EnhancementLossTests(unittest.TestCase):
    """Basic correctness tests for the enhancement loss helpers."""

    def test_each_loss_returns_scalar_tensor(self) -> None:
        input_image, enhanced_image, curve_maps = build_sample_tensors()

        losses = [
            exposure_control_loss(enhanced_image),
            color_constancy_loss(enhanced_image),
            illumination_smoothness_loss(curve_maps),
            spatial_consistency_loss(input_image, enhanced_image),
        ]

        for loss_value in losses:
            self.assertIsInstance(loss_value, torch.Tensor)
            self.assertEqual(loss_value.ndim, 0)
            self.assertTrue(torch.isfinite(loss_value).item())

    def test_color_constancy_is_zero_for_perfectly_balanced_gray(self) -> None:
        gray_image = torch.full((1, 3, 16, 16), 0.5)
        loss_value = color_constancy_loss(gray_image)
        self.assertTrue(torch.allclose(loss_value, torch.tensor(0.0), atol=1e-6))

    def test_smoothness_is_zero_for_constant_curve_maps(self) -> None:
        curve_maps = torch.zeros(1, 8, 3, 16, 16)
        loss_value = illumination_smoothness_loss(curve_maps)
        self.assertTrue(torch.allclose(loss_value, torch.tensor(0.0), atol=1e-6))

    def test_combined_loss_includes_expected_keys(self) -> None:
        input_image, enhanced_image, curve_maps = build_sample_tensors()
        loss_dict = enhancement_loss(
            enhanced_image,
            curve_maps,
            input_image=input_image,
        )

        self.assertEqual(
            set(loss_dict.keys()),
            {
                "total",
                "exposure",
                "color_constancy",
                "illumination_smoothness",
                "spatial_consistency",
            },
        )
        for loss_value in loss_dict.values():
            self.assertTrue(torch.isfinite(loss_value).item())

    def test_combined_loss_can_disable_spatial_term(self) -> None:
        _, enhanced_image, curve_maps = build_sample_tensors()
        loss_dict = enhancement_loss(
            enhanced_image,
            curve_maps,
            use_spatial_consistency=False,
        )

        self.assertTrue(torch.allclose(loss_dict["spatial_consistency"], torch.tensor(0.0)))


if __name__ == "__main__":
    example_usage()
    unittest.main(verbosity=2)
