#!/usr/bin/env python3
"""Joint loss for end-to-end enhancement and detection training."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from configs.train_config import JointLossConfig, JointTrainConfig
from losses.enhancement_losses import enhancement_loss


def _resolve_joint_loss_config(
    config: JointLossConfig | JointTrainConfig | None,
) -> JointLossConfig:
    """Accept either a dedicated loss config or a full training config."""
    if config is None:
        return JointLossConfig()
    if isinstance(config, JointTrainConfig):
        return config.joint_loss
    return config


def _validate_detection_inputs(
    classification_logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Ensure logits and class targets have shapes expected by CrossEntropyLoss."""
    if classification_logits.ndim != 2:
        raise ValueError(
            "classification_logits must have shape [batch, num_classes], "
            f"but received {tuple(classification_logits.shape)}."
        )
    if targets.ndim != 1:
        raise ValueError(
            "targets must have shape [batch], "
            f"but received {tuple(targets.shape)}."
        )
    if classification_logits.shape[0] != targets.shape[0]:
        raise ValueError(
            "classification_logits and targets must share the same batch size, "
            f"but received {classification_logits.shape[0]} and {targets.shape[0]}."
        )


def compute_joint_loss(
    classification_logits: torch.Tensor,
    targets: torch.Tensor,
    enhanced_image: torch.Tensor,
    curve_maps: torch.Tensor,
    *,
    input_image: torch.Tensor | None = None,
    config: JointLossConfig | JointTrainConfig | None = None,
    detection_criterion: nn.CrossEntropyLoss | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the total joint objective for the enhancer-detector pipeline.

    The weighted sum is used because detection and enhancement optimize related
    but different goals, and their raw loss magnitudes can live on different
    scales. The lambda term makes it easy to tune how strongly enhancement
    regularization should influence the end-to-end detector objective.
    """
    resolved_config = _resolve_joint_loss_config(config)
    _validate_detection_inputs(classification_logits, targets)

    criterion = detection_criterion or nn.CrossEntropyLoss()
    if not isinstance(criterion, nn.CrossEntropyLoss):
        raise TypeError("detection_criterion must be an instance of nn.CrossEntropyLoss.")
    detection = criterion(classification_logits, targets)

    enhancement_terms = enhancement_loss(
        enhanced_image,
        curve_maps,
        input_image=input_image,
        **resolved_config.enhancement.to_kwargs(),
    )
    enhancement_total = enhancement_terms["total"]
    weighted_enhancement = resolved_config.enhancement_lambda * enhancement_total
    total = detection + weighted_enhancement

    return {
        "total": total,
        "detection": detection,
        "enhancement": enhancement_total,
        "weighted_enhancement": weighted_enhancement,
        "enhancement_exposure": enhancement_terms["exposure"],
        "enhancement_color_constancy": enhancement_terms["color_constancy"],
        "enhancement_illumination_smoothness": enhancement_terms["illumination_smoothness"],
        "enhancement_spatial_consistency": enhancement_terms["spatial_consistency"],
    }


class JointTrainingLoss(nn.Module):
    """nn.Module wrapper around the joint training objective."""

    def __init__(self, config: JointLossConfig | JointTrainConfig | None = None) -> None:
        super().__init__()
        self.config = _resolve_joint_loss_config(config)
        self.detection_criterion = nn.CrossEntropyLoss()

    @property
    def enhancement_lambda(self) -> float:
        """Expose the current lambda used for weighting the enhancement loss."""
        return self.config.enhancement_lambda

    def set_enhancement_lambda(self, value: float) -> None:
        """Update lambda without rebuilding the loss object."""
        self.config = self.config.with_enhancement_lambda(value)

    def forward(
        self,
        classification_logits: torch.Tensor,
        targets: torch.Tensor,
        enhanced_image: torch.Tensor,
        curve_maps: torch.Tensor,
        *,
        input_image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return total loss plus separate detection and enhancement terms."""
        return compute_joint_loss(
            classification_logits,
            targets,
            enhanced_image,
            curve_maps,
            input_image=input_image,
            config=self.config,
            detection_criterion=self.detection_criterion,
        )


def loss_dict_to_log_items(loss_dict: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Convert a tensor-valued loss dictionary into floats for logging."""
    loggable: dict[str, float] = {}
    for key, value in loss_dict.items():
        if torch.is_tensor(value):
            loggable[key] = float(value.detach().item())
    return loggable
