#!/usr/bin/env python3
"""Metrics and threshold-tuning helpers for binary eye-state classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import torch


DEFAULT_MIN_CLOSED_PREDICTION_RATE = 0.05
DEFAULT_MAX_CLOSED_PREDICTION_RATE = 0.95


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def resolve_positive_label(
    class_to_idx: Mapping[str, int] | None = None,
    positive_class_name: str = "closed",
) -> int:
    if class_to_idx and positive_class_name in class_to_idx:
        return int(class_to_idx[positive_class_name])
    if class_to_idx and len(class_to_idx) == 2:
        return int(max(class_to_idx.values()))
    return 1


def compute_binary_metrics(
    *,
    targets: np.ndarray,
    predictions: np.ndarray,
    positive_label: int = 1,
) -> dict[str, float | int | np.ndarray]:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    positive = int(positive_label)
    negative = 1 - positive

    tp = int(((predictions == positive) & (targets == positive)).sum())
    fp = int(((predictions == positive) & (targets != positive)).sum())
    fn = int(((predictions != positive) & (targets == positive)).sum())
    tn = int(((predictions == negative) & (targets == negative)).sum())

    accuracy = safe_divide(tp + tn, len(targets))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    specificity = safe_divide(tn, tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)
    predicted_positive_rate = safe_divide(tp + fp, len(targets))
    predicted_negative_rate = safe_divide(tn + fn, len(targets))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "closed_recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "predicted_positive_rate": predicted_positive_rate,
        "predicted_negative_rate": predicted_negative_rate,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]], dtype=np.int64),
    }


def predictions_from_closed_probability(
    closed_probabilities: np.ndarray,
    *,
    threshold: float,
    positive_label: int = 1,
) -> np.ndarray:
    closed_predictions = (closed_probabilities >= threshold).astype(np.int64)
    if positive_label == 1:
        return closed_predictions
    return 1 - closed_predictions


def evaluate_threshold_candidates(
    *,
    targets: np.ndarray,
    closed_probabilities: np.ndarray,
    positive_label: int = 1,
    thresholds: Iterable[float] | None = None,
) -> list[dict[str, float | int | np.ndarray]]:
    """Compute metrics for each candidate threshold."""
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    closed_probabilities = np.asarray(closed_probabilities, dtype=np.float32).reshape(-1)
    threshold_candidates = list(thresholds or np.linspace(0.10, 0.90, 33))

    results: list[dict[str, float | int | np.ndarray]] = []
    for threshold in threshold_candidates:
        predictions = predictions_from_closed_probability(
            closed_probabilities,
            threshold=float(threshold),
            positive_label=positive_label,
        )
        metrics = compute_binary_metrics(
            targets=targets,
            predictions=predictions,
            positive_label=positive_label,
        )
        metrics = dict(metrics)
        metrics["threshold"] = float(threshold)
        results.append(metrics)
    return results


def tune_closed_threshold(
    *,
    targets: np.ndarray,
    closed_probabilities: np.ndarray,
    positive_label: int = 1,
    thresholds: Iterable[float] | None = None,
    objective: str = "f1",
    min_positive_rate: float = DEFAULT_MIN_CLOSED_PREDICTION_RATE,
    max_positive_rate: float = DEFAULT_MAX_CLOSED_PREDICTION_RATE,
) -> tuple[float, dict[str, float | int | np.ndarray]]:
    """Search thresholds to improve closed-eye detection on validation data."""
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    closed_probabilities = np.asarray(closed_probabilities, dtype=np.float32).reshape(-1)
    threshold_metrics = evaluate_threshold_candidates(
        targets=targets,
        closed_probabilities=closed_probabilities,
        positive_label=positive_label,
        thresholds=thresholds,
    )
    if not threshold_metrics:
        raise ValueError("No threshold candidates were provided for tuning.")

    best_metrics = select_best_threshold_metrics(
        threshold_metrics,
        objective=objective,
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
    )
    best_threshold = float(best_metrics["threshold"])
    return best_threshold, best_metrics


def threshold_within_closed_rate_bounds(
    metrics: Mapping[str, float | int | np.ndarray],
    *,
    min_positive_rate: float = DEFAULT_MIN_CLOSED_PREDICTION_RATE,
    max_positive_rate: float = DEFAULT_MAX_CLOSED_PREDICTION_RATE,
) -> bool:
    """Return True when the threshold predicts both classes in a reasonable ratio.

    Without this guardrail, F1-only tuning on balanced validation data can prefer
    degenerate thresholds that predict almost everything as `closed`, which looks
    strong on recall but is not a useful operating point for the report or for
    downstream deployment.
    """
    positive_rate = float(metrics["predicted_positive_rate"])
    return min_positive_rate <= positive_rate <= max_positive_rate


def _threshold_sort_key(
    metrics: Mapping[str, float | int | np.ndarray],
    *,
    objective: str,
) -> tuple[float, ...]:
    positive_rate = float(metrics["predicted_positive_rate"])
    balance_bonus = -abs(positive_rate - 0.5)
    if objective == "recall":
        return (
            float(metrics["recall"]),
            float(metrics["f1"]),
            float(metrics["balanced_accuracy"]),
            float(metrics["precision"]),
            float(metrics["accuracy"]),
            balance_bonus,
        )
    return (
        float(metrics["f1"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["accuracy"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        balance_bonus,
    )


def select_best_threshold_metrics(
    threshold_metrics: Iterable[Mapping[str, float | int | np.ndarray]],
    *,
    objective: str = "f1",
    min_positive_rate: float = DEFAULT_MIN_CLOSED_PREDICTION_RATE,
    max_positive_rate: float = DEFAULT_MAX_CLOSED_PREDICTION_RATE,
) -> dict[str, float | int | np.ndarray]:
    """Select the best threshold while avoiding collapsed prediction regimes."""
    rows = [dict(row) for row in threshold_metrics]
    if not rows:
        raise ValueError("No threshold metrics were provided for selection.")

    rate_guarded = [
        row
        for row in rows
        if threshold_within_closed_rate_bounds(
            row,
            min_positive_rate=min_positive_rate,
            max_positive_rate=max_positive_rate,
        )
    ]
    if rate_guarded:
        candidates = rate_guarded
    else:
        candidates = [
            row
            for row in rows
            if 0.0 < float(row["predicted_positive_rate"]) < 1.0
        ] or rows

    best_metrics = max(
        candidates,
        key=lambda row: _threshold_sort_key(row, objective=objective),
    )
    return dict(best_metrics)


@dataclass
class ProbabilityAccumulator:
    """Accumulate losses, targets, and probabilities across one epoch."""

    total_loss: float = 0.0
    total_samples: int = 0
    target_batches: list[np.ndarray] = None
    closed_probability_batches: list[np.ndarray] = None
    default_prediction_batches: list[np.ndarray] = None

    def __post_init__(self) -> None:
        self.target_batches = []
        self.closed_probability_batches = []
        self.default_prediction_batches = []

    def update(
        self,
        *,
        loss: float,
        logits: torch.Tensor,
        targets: torch.Tensor,
        positive_label: int = 1,
    ) -> None:
        batch_size = targets.numel()
        probabilities = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        targets_np = targets.detach().cpu().numpy().astype(np.int64)
        closed_probs = probabilities[:, positive_label]
        default_predictions = np.argmax(probabilities, axis=1)

        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        self.target_batches.append(targets_np)
        self.closed_probability_batches.append(closed_probs)
        self.default_prediction_batches.append(default_predictions)

    def compute(
        self,
        *,
        positive_label: int = 1,
        threshold: float | None = None,
        tune_threshold: bool = False,
        threshold_objective: str = "f1",
        threshold_candidates: Iterable[float] | None = None,
        min_positive_rate: float = DEFAULT_MIN_CLOSED_PREDICTION_RATE,
        max_positive_rate: float = DEFAULT_MAX_CLOSED_PREDICTION_RATE,
    ) -> dict[str, float | int | np.ndarray]:
        targets = np.concatenate(self.target_batches) if self.target_batches else np.empty(0, dtype=np.int64)
        closed_probabilities = (
            np.concatenate(self.closed_probability_batches)
            if self.closed_probability_batches
            else np.empty(0, dtype=np.float32)
        )

        if tune_threshold:
            best_threshold, metrics = tune_closed_threshold(
                targets=targets,
                closed_probabilities=closed_probabilities,
                positive_label=positive_label,
                thresholds=threshold_candidates,
                objective=threshold_objective,
                min_positive_rate=min_positive_rate,
                max_positive_rate=max_positive_rate,
            )
        else:
            chosen_threshold = 0.5 if threshold is None else float(threshold)
            predictions = predictions_from_closed_probability(
                closed_probabilities,
                threshold=chosen_threshold,
                positive_label=positive_label,
            )
            metrics = compute_binary_metrics(
                targets=targets,
                predictions=predictions,
                positive_label=positive_label,
            )
            best_threshold = chosen_threshold

        metrics = dict(metrics)
        metrics["loss"] = safe_divide(self.total_loss, self.total_samples)
        metrics["threshold"] = best_threshold
        return metrics

    def export_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return concatenated targets and closed-class probabilities."""
        targets = np.concatenate(self.target_batches) if self.target_batches else np.empty(0, dtype=np.int64)
        closed_probabilities = (
            np.concatenate(self.closed_probability_batches)
            if self.closed_probability_batches
            else np.empty(0, dtype=np.float32)
        )
        return targets, closed_probabilities


def format_classifier_metric_line(name: str, metrics: Mapping[str, float]) -> str:
    return (
        f"{name:<8} "
        f"loss: {metrics['loss']:.4f} | "
        f"accuracy: {metrics['accuracy']:.4f} | "
        f"precision: {metrics['precision']:.4f} | "
        f"recall: {metrics['recall']:.4f} | "
        f"f1: {metrics['f1']:.4f} | "
        f"closed_recall: {metrics['closed_recall']:.4f} | "
        f"threshold: {metrics['threshold']:.2f}"
    )
