#!/usr/bin/env python3
"""Smoke tests for the joint enhancement-detection pipeline."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.joint_model import build_joint_model, run_dummy_joint_forward_pass


def example_usage() -> None:
    """Print one example forward pass through the full joint pipeline."""
    model = build_joint_model(
        detector_backbone="custom",
        detector_kwargs={"use_pretrained": False},
        debug=True,
    )
    outputs = run_dummy_joint_forward_pass(
        model,
        batch_size=2,
        image_size=224,
        return_curve_maps=True,
        debug=True,
    )
    logits, enhanced_image, curve_maps = outputs
    print("\n=== Joint model example ===")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Enhanced image shape: {tuple(enhanced_image.shape)}")
    print(f"Curve maps shape: {tuple(curve_maps.shape)}")


class JointModelTests(unittest.TestCase):
    """Basic correctness checks for the enhancer-detector pipeline."""

    def test_joint_pipeline_shapes_with_curve_maps(self) -> None:
        model = build_joint_model(
            detector_backbone="custom",
            detector_kwargs={"use_pretrained": False},
        )
        logits, enhanced_image, curve_maps = run_dummy_joint_forward_pass(
            model,
            batch_size=2,
            image_size=224,
            return_curve_maps=True,
        )

        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(enhanced_image.shape), (2, 3, 224, 224))
        self.assertEqual(tuple(curve_maps.shape), (2, 8, 3, 224, 224))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.isfinite(enhanced_image).all().item())
        self.assertTrue(torch.isfinite(curve_maps).all().item())

    def test_joint_pipeline_can_skip_curve_maps(self) -> None:
        model = build_joint_model(
            detector_backbone="custom",
            detector_kwargs={"use_pretrained": False},
            return_curve_maps_by_default=False,
        )
        outputs = run_dummy_joint_forward_pass(
            model,
            batch_size=2,
            image_size=224,
            return_curve_maps=False,
        )

        self.assertEqual(len(outputs), 2)
        logits, enhanced_image = outputs
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(enhanced_image.shape), (2, 3, 224, 224))

    def test_detector_loss_backpropagates_into_enhancer(self) -> None:
        model = build_joint_model(
            detector_backbone="custom",
            detector_kwargs={"use_pretrained": False},
        )
        model.train()
        model.zero_grad(set_to_none=True)

        inputs = torch.rand(2, 3, 224, 224)
        targets = torch.tensor([0, 1], dtype=torch.long)

        logits, enhanced_image, curve_maps = model(inputs, return_curve_maps=True)
        self.assertEqual(tuple(enhanced_image.shape), (2, 3, 224, 224))
        self.assertEqual(tuple(curve_maps.shape), (2, 8, 3, 224, 224))

        classification_loss = F.cross_entropy(logits, targets)
        classification_loss.backward()

        enhancer_gradients = [
            parameter.grad
            for parameter in model.enhancer.parameters()
            if parameter.requires_grad
        ]
        detector_gradients = [
            parameter.grad
            for parameter in model.detector.parameters()
            if parameter.requires_grad
        ]

        self.assertTrue(any(grad is not None for grad in enhancer_gradients))
        self.assertTrue(any(grad is not None for grad in detector_gradients))
        self.assertTrue(
            any(grad is not None and torch.any(grad != 0).item() for grad in enhancer_gradients)
        )

    def test_debug_mode_prints_tensor_shapes(self) -> None:
        model = build_joint_model(
            detector_backbone="custom",
            detector_kwargs={"use_pretrained": False},
            debug=True,
        )
        input_tensor = torch.rand(1, 3, 224, 224)

        stream = io.StringIO()
        with redirect_stdout(stream):
            model(input_tensor, return_curve_maps=True, debug=True)

        debug_output = stream.getvalue()
        self.assertIn("[JointModel][DEBUG] input:", debug_output)
        self.assertIn("[JointModel][DEBUG] enhanced_image:", debug_output)
        self.assertIn("[JointModel][DEBUG] classification_logits:", debug_output)

    def test_invalid_input_shape_raises_clear_error(self) -> None:
        model = build_joint_model(
            detector_backbone="custom",
            detector_kwargs={"use_pretrained": False},
        )
        bad_input = torch.rand(2, 3, 128, 128)

        with self.assertRaisesRegex(ValueError, "expects images resized to 224x224"):
            model(bad_input)


if __name__ == "__main__":
    torch.manual_seed(17)
    example_usage()
    unittest.main(verbosity=2)
