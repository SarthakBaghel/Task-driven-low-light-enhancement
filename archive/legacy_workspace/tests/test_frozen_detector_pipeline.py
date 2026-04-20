#!/usr/bin/env python3
"""Tests for the enhancer + frozen detector evaluation pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.frozen_detector_pipeline import build_enhancer_frozen_detector_pipeline


class DummyEnhancer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.9, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enhanced = torch.clamp(x * self.scale, 0.0, 1.0)
        curve_maps = torch.zeros(
            x.shape[0],
            1,
            x.shape[1],
            x.shape[2],
            x.shape[3],
            device=x.device,
            dtype=x.dtype,
        )
        return enhanced, curve_maps


class DummyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DummyDualDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.use_dual_input = True
        self.raw_branch = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.enhanced_branch = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(8, 2)

    def forward(self, raw_x: torch.Tensor, enhanced_x: torch.Tensor | None = None) -> torch.Tensor:
        if enhanced_x is None:
            raw_x, enhanced_x = raw_x
        raw_features = self.raw_branch(raw_x)
        enhanced_features = self.enhanced_branch(enhanced_x)
        return self.classifier(torch.cat([raw_features, enhanced_features], dim=1))


class FrozenDetectorPipelineTests(unittest.TestCase):
    def test_detector_is_frozen_and_kept_in_eval_mode(self) -> None:
        enhancer = DummyEnhancer()
        detector = DummyDetector()
        pipeline = build_enhancer_frozen_detector_pipeline(
            enhancer=enhancer,
            detector=detector,
        )

        self.assertTrue(pipeline.detector_is_frozen())
        self.assertFalse(pipeline.detector.training)

        pipeline.train()
        self.assertTrue(pipeline.training)
        self.assertTrue(pipeline.enhancer.training)
        self.assertFalse(pipeline.detector.training)

    def test_forward_returns_logits_enhanced_image_and_curve_maps(self) -> None:
        pipeline = build_enhancer_frozen_detector_pipeline(
            enhancer=DummyEnhancer(),
            detector=DummyDetector(),
            return_enhanced_image_by_default=True,
            return_curve_maps_by_default=True,
        )
        inputs = torch.rand(2, 3, 32, 32)
        logits, enhanced, curve_maps = pipeline(inputs)

        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(enhanced.shape), (2, 3, 32, 32))
        self.assertEqual(tuple(curve_maps.shape), (2, 1, 3, 32, 32))

    def test_backward_updates_enhancer_but_not_detector(self) -> None:
        enhancer = DummyEnhancer()
        detector = DummyDetector()
        pipeline = build_enhancer_frozen_detector_pipeline(
            enhancer=enhancer,
            detector=detector,
            return_enhanced_image_by_default=False,
            return_curve_maps_by_default=False,
        )

        inputs = torch.rand(2, 3, 16, 16)
        logits = pipeline(inputs)
        loss = logits.sum()
        loss.backward()

        self.assertIsNotNone(enhancer.scale.grad)
        for parameter in detector.parameters():
            self.assertIsNone(parameter.grad)

    def test_dual_input_detector_receives_raw_and_enhanced_images(self) -> None:
        pipeline = build_enhancer_frozen_detector_pipeline(
            enhancer=DummyEnhancer(),
            detector=DummyDualDetector(),
            return_enhanced_image_by_default=False,
            return_curve_maps_by_default=False,
            print_frozen_layers=False,
            debug_freeze_state=False,
        )

        inputs = torch.rand(2, 3, 24, 24)
        logits = pipeline(inputs)

        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(pipeline.detector_is_frozen())


if __name__ == "__main__":
    unittest.main()
