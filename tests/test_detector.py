#!/usr/bin/env python3
"""Small smoke tests for the eye-state detector backbones."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.detector import (
    build_detector,
    count_stored_gradients,
    gradients_enabled_for_model,
    run_dummy_forward_pass,
    summarize_model,
)


def example_usage() -> None:
    """Print summaries and dummy output shapes for both detector options."""
    examples = (
        ("custom", False),
        ("mobilenetv2", True),
        ("resnet18-dual", True),
    )

    for backbone, use_pretrained in examples:
        resolved_backbone = "resnet18" if backbone == "resnet18-dual" else backbone
        use_dual_input = backbone == "resnet18-dual"
        print(f"\n=== Detector example: {backbone} ===")
        model = build_detector(
            backbone=resolved_backbone,
            use_pretrained=use_pretrained,
            allow_pretrained_fallback=True,
            use_dual_input=use_dual_input,
            print_summary=True,
        )
        outputs = run_dummy_forward_pass(model, batch_size=2, image_size=224)
        print(f"Dummy forward output shape: {tuple(outputs.shape)}")


class DetectorTests(unittest.TestCase):
    """Basic detector checks for the custom CNN and MobileNetV2 options."""

    def test_custom_detector_dummy_forward(self) -> None:
        model = build_detector(
            backbone="custom",
            use_pretrained=False,
        )
        outputs = run_dummy_forward_pass(model, batch_size=2, image_size=224)

        self.assertEqual(tuple(outputs.shape), (2, 2))
        self.assertTrue(torch.isfinite(outputs).all().item())

        summary = summarize_model(model)
        self.assertIn("Backbone: custom", summary)
        self.assertIn("Output size: (1, 2)", summary)

    def test_mobilenetv2_detector_dummy_forward(self) -> None:
        model = build_detector(
            backbone="mobilenetv2",
            use_pretrained=True,
            allow_pretrained_fallback=True,
            mobilenet_trainable_blocks=3,
        )
        outputs = run_dummy_forward_pass(model, batch_size=2, image_size=224)

        self.assertEqual(tuple(outputs.shape), (2, 2))
        self.assertTrue(torch.isfinite(outputs).all().item())

        summary = summarize_model(model)
        self.assertIn("Backbone: mobilenetv2", summary)
        self.assertIn("Trainable MobileNetV2 feature blocks: 3", summary)

        feature_flags = [parameter.requires_grad for parameter in model.model.features.parameters()]
        self.assertTrue(any(feature_flags))
        self.assertTrue(not all(feature_flags))

    def test_freeze_detector_option_disables_gradients_and_sets_eval_mode(self) -> None:
        model = build_detector(
            backbone="custom",
            use_pretrained=False,
            freeze_detector=True,
            print_frozen_layers=False,
        )

        self.assertFalse(model.training)
        self.assertFalse(gradients_enabled_for_model(model))
        self.assertEqual(count_stored_gradients(model), 0)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_dual_input_detector_dummy_forward_with_shared_backbone(self) -> None:
        model = build_detector(
            backbone="resnet18",
            use_pretrained=True,
            allow_pretrained_fallback=True,
            use_dual_input=True,
            dual_input_shared_backbone=True,
            print_frozen_layers=False,
        )
        outputs = run_dummy_forward_pass(model, batch_size=2, image_size=224)

        self.assertEqual(tuple(outputs.shape), (2, 2))
        summary = summarize_model(model)
        self.assertIn("Dual input enabled: True", summary)
        self.assertIn("shared weights", summary)

    def test_dual_input_detector_dummy_forward_with_separate_backbones(self) -> None:
        model = build_detector(
            backbone="custom",
            use_pretrained=False,
            use_dual_input=True,
            dual_input_shared_backbone=False,
            print_frozen_layers=False,
        )
        raw_inputs = torch.randn(2, 3, 224, 224)
        enhanced_inputs = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            outputs = model(raw_inputs, enhanced_inputs)

        self.assertEqual(tuple(outputs.shape), (2, 2))
        self.assertTrue(hasattr(model, "raw_backbone"))
        self.assertTrue(hasattr(model, "enhanced_backbone"))


if __name__ == "__main__":
    torch.manual_seed(13)
    example_usage()
    unittest.main(verbosity=2)
