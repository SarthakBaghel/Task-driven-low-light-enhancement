#!/usr/bin/env python3
"""Joint low-light enhancement and eye-state detection model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.detector import DEFAULT_IMAGE_SIZE, EyeStateDetector, build_detector
from models.zerodce import ZeroDCE


@dataclass
class JointModelConfig:
    """Configuration for the enhancer-detector pipeline."""

    image_size: int = DEFAULT_IMAGE_SIZE
    num_classes: int = 2
    detector_backbone: str = "custom"
    return_curve_maps_by_default: bool = True
    strict_input_size_check: bool = True
    debug: bool = False


class JointEnhanceDetectModel(nn.Module):
    """Pipeline: low-light image -> enhancer -> detector -> classification logits."""

    def __init__(
        self,
        *,
        enhancer: nn.Module | None = None,
        detector: nn.Module | None = None,
        config: JointModelConfig | None = None,
        enhancer_kwargs: dict[str, object] | None = None,
        detector_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or JointModelConfig()

        enhancer_options = dict(enhancer_kwargs or {})
        detector_options = dict(detector_kwargs or {})
        detector_options.setdefault("backbone", self.config.detector_backbone)
        detector_options.setdefault("num_classes", self.config.num_classes)
        detector_options.setdefault("image_size", self.config.image_size)

        self.enhancer = enhancer if enhancer is not None else ZeroDCE(**enhancer_options)
        self.detector = detector if detector is not None else build_detector(**detector_options)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_curve_maps: bool | None = None,
        debug: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run enhancement first, then eye-state detection on the enhanced image."""
        should_return_curve_maps = (
            self.config.return_curve_maps_by_default
            if return_curve_maps is None
            else return_curve_maps
        )
        development_mode = self.config.debug if debug is None else debug

        self._validate_input(x)
        if development_mode:
            self._debug_print("input", x)

        enhanced_image, curve_maps = self.enhancer(x)
        self._validate_enhancer_output(x, enhanced_image, curve_maps)
        if development_mode:
            self._debug_print("enhanced_image", enhanced_image)
            self._debug_print("curve_maps", curve_maps)

        classification_logits = self.detector(enhanced_image)
        self._validate_detector_output(x, classification_logits)
        if development_mode:
            self._debug_print("classification_logits", classification_logits)

        if should_return_curve_maps:
            return classification_logits, enhanced_image, curve_maps
        return classification_logits, enhanced_image

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor input, but received {type(x)!r}.")
        if x.ndim != 4:
            raise ValueError(
                "Expected input shape [batch, 3, height, width], "
                f"but received {tuple(x.shape)}."
            )
        if x.shape[1] != 3:
            raise ValueError(
                "Joint model expects RGB images with 3 channels, "
                f"but received {x.shape[1]} channels."
            )
        if self.config.strict_input_size_check and x.shape[-2:] != (
            self.config.image_size,
            self.config.image_size,
        ):
            raise ValueError(
                "Joint model expects images resized to "
                f"{self.config.image_size}x{self.config.image_size}, "
                f"but received spatial size {tuple(x.shape[-2:])}."
            )

    @staticmethod
    def _validate_enhancer_output(
        input_tensor: torch.Tensor,
        enhanced_image: torch.Tensor,
        curve_maps: torch.Tensor,
    ) -> None:
        if enhanced_image.shape != input_tensor.shape:
            raise ValueError(
                "Enhancer output shape must match the detector input shape. "
                f"Expected {tuple(input_tensor.shape)}, got {tuple(enhanced_image.shape)}."
            )
        if curve_maps.ndim != 5:
            raise ValueError(
                "Curve maps must have shape [batch, iterations, channels, height, width], "
                f"but received {tuple(curve_maps.shape)}."
            )
        if curve_maps.shape[0] != input_tensor.shape[0]:
            raise ValueError("Curve maps batch dimension does not match the input batch size.")
        if curve_maps.shape[2] != input_tensor.shape[1]:
            raise ValueError("Curve maps channel dimension must match the RGB input channels.")
        if curve_maps.shape[-2:] != input_tensor.shape[-2:]:
            raise ValueError("Curve maps spatial dimensions must match the input image size.")

    def _validate_detector_output(
        self,
        input_tensor: torch.Tensor,
        classification_logits: torch.Tensor,
    ) -> None:
        if classification_logits.ndim != 2:
            raise ValueError(
                "Detector output must have shape [batch, num_classes], "
                f"but received {tuple(classification_logits.shape)}."
            )
        if classification_logits.shape[0] != input_tensor.shape[0]:
            raise ValueError("Detector batch size does not match the input batch size.")
        expected_classes = self.config.num_classes
        if classification_logits.shape[1] != expected_classes:
            raise ValueError(
                f"Detector must output {expected_classes} class logits, "
                f"but received {classification_logits.shape[1]}."
            )

    @staticmethod
    def _debug_print(name: str, tensor: torch.Tensor) -> None:
        """Print compact tensor stats for development-time debugging."""
        detached = tensor.detach()
        if detached.numel() == 0:
            print(f"[JointModel][DEBUG] {name}: empty tensor shape={tuple(detached.shape)}")
            return

        min_value = detached.min().item()
        max_value = detached.max().item()
        mean_value = detached.mean().item()
        print(
            "[JointModel][DEBUG] "
            f"{name}: shape={tuple(detached.shape)} dtype={detached.dtype} "
            f"device={detached.device} min={min_value:.4f} "
            f"max={max_value:.4f} mean={mean_value:.4f}"
        )


def build_joint_model(
    *,
    enhancer: nn.Module | None = None,
    detector: nn.Module | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_classes: int = 2,
    detector_backbone: str = "custom",
    return_curve_maps_by_default: bool = True,
    strict_input_size_check: bool = True,
    debug: bool = False,
    enhancer_kwargs: dict[str, object] | None = None,
    detector_kwargs: dict[str, object] | None = None,
) -> JointEnhanceDetectModel:
    """Build a joint enhancer-detector pipeline."""
    config = JointModelConfig(
        image_size=image_size,
        num_classes=num_classes,
        detector_backbone=detector_backbone,
        return_curve_maps_by_default=return_curve_maps_by_default,
        strict_input_size_check=strict_input_size_check,
        debug=debug,
    )
    return JointEnhanceDetectModel(
        enhancer=enhancer,
        detector=detector,
        config=config,
        enhancer_kwargs=enhancer_kwargs,
        detector_kwargs=detector_kwargs,
    )


def run_dummy_joint_forward_pass(
    model: JointEnhanceDetectModel,
    *,
    batch_size: int = 2,
    image_size: int = DEFAULT_IMAGE_SIZE,
    return_curve_maps: bool = True,
    device: torch.device | str | None = None,
    debug: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a dummy low-light batch through the full enhancer-detector pipeline."""
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    model = model.to(target_device)
    dummy_input = torch.rand(batch_size, 3, image_size, image_size, device=target_device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input, return_curve_maps=return_curve_maps, debug=debug)
    if was_training:
        model.train()

    if isinstance(outputs, tuple):
        return tuple(output.cpu() for output in outputs)
    raise TypeError("Expected the joint model to return a tuple of tensors.")
