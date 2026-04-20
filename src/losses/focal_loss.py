#!/usr/bin/env python3
"""Focal loss for imbalanced classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weighting."""

    def __init__(
        self,
        *,
        alpha: torch.Tensor | list[float] | tuple[float, ...] | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("'gamma' must be non-negative.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: 'none', 'mean', 'sum'.")
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is None:
            self.alpha = None
        else:
            alpha_tensor = (
                alpha if torch.is_tensor(alpha) else torch.tensor(alpha, dtype=torch.float32)
            )
            self.register_buffer("alpha", alpha_tensor.to(dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(
                "logits must have shape [batch, num_classes], "
                f"but received {tuple(logits.shape)}."
            )
        if targets.ndim != 1:
            raise ValueError(
                "targets must have shape [batch], "
                f"but received {tuple(targets.shape)}."
            )

        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_factor = torch.pow(1.0 - pt, self.gamma)

        loss = focal_factor * ce_loss
        if self.alpha is not None:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            if alpha.numel() != logits.shape[1]:
                raise ValueError(
                    "alpha must have one entry per class. "
                    f"Expected {logits.shape[1]}, got {alpha.numel()}."
                )
            loss = alpha[targets] * loss

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        return loss
