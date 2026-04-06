#!/usr/bin/env python3
"""Enhancer + frozen detector pipeline for evaluation and enhancer-only studies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dataset import IMAGENET_MEAN, IMAGENET_STD
from models.detector import freeze_model, print_gradient_debug_info


@dataclass
class FrozenDetectorPipelineConfig:
    """Configuration for the enhancer + frozen detector evaluation pipeline."""

    normalize_detector_input: bool = True
    keep_detector_eval: bool = True
    return_enhanced_image_by_default: bool = True
    return_curve_maps_by_default: bool = False
    print_frozen_layers: bool = True
    debug_freeze_state: bool = True


class EnhancerFrozenDetectorPipeline(nn.Module):
    """Run low-light images through an enhancer, then through a frozen detector.

    The detector is frozen and kept in evaluation mode so this module can be
    used for:
    - report-time "Enhancer + Frozen Detector" evaluation
    - enhancer-only experiments where the detector must remain fixed
    """

    def __init__(
        self,
        *,
        enhancer: nn.Module,
        detector: nn.Module,
        config: FrozenDetectorPipelineConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or FrozenDetectorPipelineConfig()
        self.enhancer = enhancer
        self.detector = detector

        self.register_buffer(
            "_detector_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_detector_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.freeze_detector()
        if self.config.keep_detector_eval:
            self.detector.eval()
        if self.config.debug_freeze_state:
            print_gradient_debug_info(self.detector, module_name="detector")

    def freeze_detector(self) -> None:
        """Freeze all detector parameters so no gradients accumulate into them."""
        freeze_model(
            self.detector,
            set_eval=self.config.keep_detector_eval,
            module_name="detector",
            print_frozen_layers=self.config.print_frozen_layers,
        )

    def detector_is_frozen(self) -> bool:
        """Return True when every detector parameter is non-trainable."""
        return all(not parameter.requires_grad for parameter in self.detector.parameters())

    def normalize_for_detector(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize enhanced images before passing them into the detector."""
        if not self.config.normalize_detector_input:
            return image
        mean = self._detector_mean.to(device=image.device, dtype=image.dtype)
        std = self._detector_std.to(device=image.device, dtype=image.dtype)
        return (image - mean) / std

    def train(self, mode: bool = True):
        """Keep the detector in eval mode even if the outer module is put in train mode."""
        super().train(mode)
        if self.config.keep_detector_eval:
            self.detector.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_enhanced_image: bool | None = None,
        return_curve_maps: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return detector logits, and optionally the enhanced image and curve maps."""
        include_enhanced = (
            self.config.return_enhanced_image_by_default
            if return_enhanced_image is None
            else return_enhanced_image
        )
        include_curve_maps = (
            self.config.return_curve_maps_by_default
            if return_curve_maps is None
            else return_curve_maps
        )

        enhanced_image, curve_maps = self.enhancer(x)
        enhanced_detector_input = self.normalize_for_detector(enhanced_image)
        if getattr(self.detector, "use_dual_input", False):
            raw_detector_input = self.normalize_for_detector(x)
            logits = self.detector(raw_detector_input, enhanced_detector_input)
        else:
            logits = self.detector(enhanced_detector_input)

        outputs: list[torch.Tensor] = [logits]
        if include_enhanced:
            outputs.append(enhanced_image)
        if include_curve_maps:
            outputs.append(curve_maps)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


def build_enhancer_frozen_detector_pipeline(
    *,
    enhancer: nn.Module,
    detector: nn.Module,
    normalize_detector_input: bool = True,
    keep_detector_eval: bool = True,
    return_enhanced_image_by_default: bool = True,
    return_curve_maps_by_default: bool = False,
    print_frozen_layers: bool = True,
    debug_freeze_state: bool = True,
) -> EnhancerFrozenDetectorPipeline:
    """Build the enhancer + frozen detector evaluation pipeline."""
    config = FrozenDetectorPipelineConfig(
        normalize_detector_input=normalize_detector_input,
        keep_detector_eval=keep_detector_eval,
        return_enhanced_image_by_default=return_enhanced_image_by_default,
        return_curve_maps_by_default=return_curve_maps_by_default,
        print_frozen_layers=print_frozen_layers,
        debug_freeze_state=debug_freeze_state,
    )
    return EnhancerFrozenDetectorPipeline(
        enhancer=enhancer,
        detector=detector,
        config=config,
    )
