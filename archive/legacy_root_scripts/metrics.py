#!/usr/bin/env python3
"""Metrics helpers for binary eye-state classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch


def safe_divide(numerator: float, denominator: float) -> float:
    """Return a safe division result without raising on zero denominators."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def resolve_positive_label(
    class_to_idx: Mapping[str, int] | None = None,
    positive_class_name: str = "closed",
) -> int:
    """Resolve the positive class label used for precision/recall/F1."""
    if class_to_idx and positive_class_name in class_to_idx:
        return int(class_to_idx[positive_class_name])
    if class_to_idx and len(class_to_idx) == 2:
        return int(max(class_to_idx.values()))
    return 1


@dataclass
class RunningClassificationMetrics:
    """Accumulate loss and binary classification metrics across batches."""

    positive_label: int = 1
    num_classes: int = 2
    total_loss: float = 0.0
    total_samples: int = 0
    correct: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    confusion_matrix: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_classes <= 1:
            raise ValueError("'num_classes' must be greater than 1.")
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
        )

    def update(
        self,
        *,
        loss: float,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Update the running metrics using one batch of predictions."""
        preds = predictions.detach().view(-1).to(dtype=torch.int64).cpu()
        refs = targets.detach().view(-1).to(dtype=torch.int64).cpu()
        batch_size = refs.numel()
        positive = self.positive_label

        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        self.correct += int((preds == refs).sum().item())
        self.true_positives += int(((preds == positive) & (refs == positive)).sum().item())
        self.false_positives += int(((preds == positive) & (refs != positive)).sum().item())
        self.false_negatives += int(((preds != positive) & (refs == positive)).sum().item())

        matrix_indices = refs * self.num_classes + preds
        counts = torch.bincount(
            matrix_indices,
            minlength=self.num_classes * self.num_classes,
        )
        self.confusion_matrix += counts.view(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float | int | torch.Tensor]:
        """Return loss, accuracy, precision, recall, and F1 score."""
        avg_loss = safe_divide(self.total_loss, self.total_samples)
        accuracy = safe_divide(self.correct, self.total_samples)
        precision = safe_divide(
            self.true_positives,
            self.true_positives + self.false_positives,
        )
        recall = safe_divide(
            self.true_positives,
            self.true_positives + self.false_negatives,
        )
        f1 = safe_divide(2.0 * precision * recall, precision + recall)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": self.confusion_matrix.clone(),
            "num_samples": self.total_samples,
        }


def format_metric_line(name: str, metrics: Mapping[str, float]) -> str:
    """Format metrics for clean epoch-wise logging."""
    return (
        f"{name:<5} "
        f"loss: {metrics['loss']:.4f} | "
        f"accuracy: {metrics['accuracy']:.4f} | "
        f"precision: {metrics['precision']:.4f} | "
        f"recall: {metrics['recall']:.4f} | "
        f"f1: {metrics['f1']:.4f}"
    )


def extract_binary_confusion_terms(
    confusion_matrix: torch.Tensor,
    positive_label: int = 1,
) -> dict[str, int]:
    """Return TN, FP, FN, and TP values from a binary confusion matrix."""
    if confusion_matrix.shape != (2, 2):
        raise ValueError(
            "Binary confusion terms require a 2x2 confusion matrix, "
            f"but received shape {tuple(confusion_matrix.shape)}."
        )

    negative_label = 1 - positive_label
    return {
        "tn": int(confusion_matrix[negative_label, negative_label].item()),
        "fp": int(confusion_matrix[negative_label, positive_label].item()),
        "fn": int(confusion_matrix[positive_label, negative_label].item()),
        "tp": int(confusion_matrix[positive_label, positive_label].item()),
    }
