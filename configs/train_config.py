#!/usr/bin/env python3
"""Training configuration helpers for enhancement, detection, and joint learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class EnhancementLossConfig:
    """Configurable weights and hyperparameters for the Zero-DCE losses."""

    exposure_weight: float = 10.0
    color_weight: float = 5.0
    smoothness_weight: float = 200.0
    spatial_weight: float = 1.0
    exposure_patch_size: int | tuple[int, int] = 16
    exposure_target_mean: float = 0.6
    color_use_sqrt: bool = False
    color_eps: float = 1e-8
    smoothness_power: float = 2.0
    spatial_pool_size: int | tuple[int, int] = 4
    use_spatial_consistency: bool = True

    def to_kwargs(self) -> dict[str, Any]:
        """Return kwargs ready to pass into enhancement_loss()."""
        return asdict(self)


@dataclass(frozen=True)
class JointLossConfig:
    """Config for the weighted joint objective.

    `enhancement_lambda` is the main experiment knob for balancing:
    total = detection + lambda * enhancement
    """

    enhancement_lambda: float = 1.0
    enhancement: EnhancementLossConfig = field(default_factory=EnhancementLossConfig)

    def __post_init__(self) -> None:
        if self.enhancement_lambda < 0.0:
            raise ValueError("'enhancement_lambda' must be non-negative.")

    def with_enhancement_lambda(self, value: float) -> "JointLossConfig":
        """Return a copy with a new lambda for easy experiment sweeps."""
        if value < 0.0:
            raise ValueError("'value' must be non-negative.")
        return replace(self, enhancement_lambda=float(value))


@dataclass(frozen=True)
class JointTrainConfig:
    """Minimal training config for end-to-end enhancer-detector experiments."""

    image_size: int = 224
    num_classes: int = 2
    detector_backbone: str = "custom"
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    device: str = "auto"
    seed: int = 42
    joint_loss: JointLossConfig = field(default_factory=JointLossConfig)

    def with_joint_loss_lambda(self, value: float) -> "JointTrainConfig":
        """Return a copy with a different lambda for quick tuning."""
        return replace(
            self,
            joint_loss=self.joint_loss.with_enhancement_lambda(value),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the nested dataclasses into a plain dictionary."""
        return asdict(self)
