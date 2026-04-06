#!/usr/bin/env python3
"""Evaluate an enhancer followed by a frozen clean-trained detector."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import EyeStateDataset
from inference_enhancer import load_model as load_enhancer_model
from losses.focal_loss import FocalLoss
from models.detector import (
    build_detector,
    count_stored_gradients,
    gradients_enabled_for_model,
    print_gradient_debug_info,
)
from models.frozen_detector_pipeline import build_enhancer_frozen_detector_pipeline
from utils.classifier_metrics import (
    DEFAULT_MAX_CLOSED_PREDICTION_RATE,
    DEFAULT_MIN_CLOSED_PREDICTION_RATE,
    ProbabilityAccumulator,
    evaluate_threshold_candidates,
    format_classifier_metric_line,
    resolve_positive_label,
    select_best_threshold_metrics,
)
from utils.classifier_transforms import build_transfer_learning_transforms


REPORT_METRICS = ("accuracy", "precision", "recall", "f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether enhancement alone improves low-light detection by "
            "running: low-light image -> enhancer -> frozen clean-trained detector."
        ),
    )
    parser.add_argument(
        "detector_checkpoint",
        type=str,
        help="Checkpoint from the clean-trained detector.",
    )
    parser.add_argument(
        "enhancer_checkpoint",
        type=str,
        help="Checkpoint from the trained Zero-DCE enhancer.",
    )
    parser.add_argument(
        "lowlight_root",
        type=str,
        help="Low-light evaluation dataset root.",
    )
    parser.add_argument(
        "--enhanced-detector-checkpoint",
        type=str,
        help=(
            "Optional detector checkpoint fine-tuned on enhancer outputs. When provided, "
            "the script compares raw low-light, enhancer + original detector, and "
            "enhancer + fine-tuned detector."
        ),
    )
    parser.add_argument(
        "--dual-detector-checkpoint",
        type=str,
        help=(
            "Optional dual-input detector checkpoint trained on `(raw low-light, enhanced image)` "
            "pairs. When provided, the report also compares the dual-input detector."
        ),
    )
    parser.add_argument(
        "--clean-root",
        type=str,
        help="Optional clean evaluation root used as a reference baseline.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--retune-threshold-on-clean",
        action="store_true",
        help="Also evaluate the original detector on clean images with threshold tuning.",
    )
    parser.add_argument(
        "--threshold-candidates",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6],
        help="Candidate thresholds tried for each evaluation branch.",
    )
    parser.add_argument(
        "--min-closed-prediction-rate",
        type=float,
        default=DEFAULT_MIN_CLOSED_PREDICTION_RATE,
        help="Reject tuned thresholds that predict too few closed-eye samples unless no balanced option exists.",
    )
    parser.add_argument(
        "--max-closed-prediction-rate",
        type=float,
        default=DEFAULT_MAX_CLOSED_PREDICTION_RATE,
        help="Reject tuned thresholds that predict too many closed-eye samples unless no balanced option exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/evaluation_enhancer_frozen_detector",
        help="Directory where CSV, plots, and summary text will be saved.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def load_checkpoint(path: str | Path) -> dict:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def build_detector_from_checkpoint(
    checkpoint: dict,
    *,
    device: torch.device,
    freeze_detector_weights: bool,
    print_frozen_layers: bool,
) -> torch.nn.Module:
    """Rebuild a detector from checkpoint config, including dual-input variants."""
    model_config = checkpoint["model_config"]
    detector = build_detector(
        backbone=model_config["backbone"],
        num_classes=model_config["num_classes"],
        image_size=model_config["image_size"],
        mobilenet_trainable_blocks=model_config.get("mobilenet_trainable_blocks", 3),
        resnet_trainable_layers=model_config.get("resnet_trainable_layers", 1),
        use_pretrained=model_config.get("use_pretrained", True),
        allow_pretrained_fallback=True,
        freeze_detector=freeze_detector_weights,
        print_frozen_layers=print_frozen_layers,
        use_dual_input=bool(model_config.get("use_dual_input", False)),
        dual_input_shared_backbone=bool(model_config.get("dual_input_shared_backbone", True)),
    ).to(device)
    detector.load_state_dict(checkpoint["model_state_dict"])
    detector.eval()
    return detector


def build_raw_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def create_loader(
    dataset_root: str | Path,
    *,
    class_to_idx: dict[str, int],
    batch_size: int,
    num_workers: int,
    transform,
) -> tuple[EyeStateDataset, DataLoader]:
    dataset = EyeStateDataset(
        root=dataset_root,
        class_to_idx=class_to_idx,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return dataset, loader


def evaluate_detector(
    detector: torch.nn.Module,
    dataloader,
    criterion: torch.nn.Module,
    device: torch.device,
    positive_label: int,
    *,
    threshold_candidates: list[float],
    min_positive_rate: float,
    max_positive_rate: float,
) -> tuple[dict[str, float | int | np.ndarray], list[dict[str, object]]]:
    accumulator = ProbabilityAccumulator()
    detector.eval()
    print_gradient_debug_info(detector, module_name="detector-before-raw-eval")
    with torch.no_grad():
        print(f"[FreezeDebug] detector raw evaluation uses torch.no_grad={not torch.is_grad_enabled()}")
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = detector(images)
            loss = criterion(logits, labels)
            accumulator.update(
                loss=float(loss.detach().item()),
                logits=logits,
                targets=labels,
                positive_label=positive_label,
            )
    return finalize_threshold_metrics(
        accumulator,
        positive_label=positive_label,
        threshold_candidates=threshold_candidates,
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
    )


def evaluate_frozen_pipeline(
    pipeline: torch.nn.Module,
    dataloader,
    criterion: torch.nn.Module,
    device: torch.device,
    positive_label: int,
    *,
    threshold_candidates: list[float],
    min_positive_rate: float,
    max_positive_rate: float,
) -> tuple[dict[str, float | int | np.ndarray], list[dict[str, object]]]:
    accumulator = ProbabilityAccumulator()
    pipeline.eval()
    print_gradient_debug_info(pipeline.detector, module_name="detector-before-enhanced-eval")
    with torch.no_grad():
        print(f"[FreezeDebug] enhanced pipeline evaluation uses torch.no_grad={not torch.is_grad_enabled()}")
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = pipeline(
                images,
                return_enhanced_image=False,
                return_curve_maps=False,
            )
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(logits, labels)
            accumulator.update(
                loss=float(loss.detach().item()),
                logits=logits,
                targets=labels,
                positive_label=positive_label,
            )
    return finalize_threshold_metrics(
        accumulator,
        positive_label=positive_label,
        threshold_candidates=threshold_candidates,
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
    )


def finalize_threshold_metrics(
    accumulator: ProbabilityAccumulator,
    *,
    positive_label: int,
    threshold_candidates: list[float],
    min_positive_rate: float,
    max_positive_rate: float,
) -> tuple[dict[str, float | int | np.ndarray], list[dict[str, object]]]:
    """Compute metrics for each threshold and return the best guarded result."""
    targets, closed_probabilities = accumulator.export_arrays()
    threshold_rows = evaluate_threshold_candidates(
        targets=targets,
        closed_probabilities=closed_probabilities,
        positive_label=positive_label,
        thresholds=threshold_candidates,
    )
    average_loss = accumulator.total_loss / max(accumulator.total_samples, 1)
    for row in threshold_rows:
        row["loss"] = average_loss

    best_metrics = select_best_threshold_metrics(
        threshold_rows,
        objective="f1",
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
    )
    best_metrics = dict(best_metrics)
    best_metrics["loss"] = average_loss
    return best_metrics, threshold_rows


def verify_detector_is_truly_frozen(detector: torch.nn.Module, *, context: str) -> None:
    """Fail fast if the detector stops being frozen during the experiment."""
    trainable = gradients_enabled_for_model(detector)
    stored_gradients = count_stored_gradients(detector)
    print(
        f"[FreezeDebug] {context}: "
        f"requires_grad_enabled={trainable} | stored_gradients={stored_gradients}"
    )
    if trainable:
        raise RuntimeError(
            f"Detector is not fully frozen during {context}: some parameters still have "
            "requires_grad=True."
        )
    if stored_gradients != 0:
        raise RuntimeError(
            f"Detector accumulated gradients during {context}: found {stored_gradients} "
            "stored gradient tensors."
        )


def build_result_row(
    *,
    experiment: str,
    dataset: str,
    metrics: dict[str, float | int | np.ndarray],
) -> dict[str, object]:
    return {
        "experiment": experiment,
        "dataset": dataset,
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "threshold": float(metrics["threshold"]),
        "confusion_matrix": np.asarray(metrics["confusion_matrix"]).tolist(),
        "tn": int(metrics["tn"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "tp": int(metrics["tp"]),
    }


def build_threshold_row(
    *,
    experiment: str,
    dataset: str,
    metrics: dict[str, float | int | np.ndarray],
) -> dict[str, object]:
    """Return a compact per-threshold metrics row."""
    return {
        "experiment": experiment,
        "dataset": dataset,
        "threshold": float(metrics["threshold"]),
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "specificity": float(metrics["specificity"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "predicted_positive_rate": float(metrics["predicted_positive_rate"]),
        "tn": int(metrics["tn"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "tp": int(metrics["tp"]),
    }


def save_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_threshold_sweep_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Save per-threshold metrics for each experiment."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "dataset",
                "threshold",
                "loss",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "specificity",
                "balanced_accuracy",
                "predicted_positive_rate",
                "tn",
                "fp",
                "fn",
                "tp",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_report_table_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Experiment", "Dataset", "Accuracy", "Precision", "Recall", "F1"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Experiment": row["experiment"],
                    "Dataset": row["dataset"],
                    "Accuracy": f"{100.0 * float(row['accuracy']):.2f}",
                    "Precision": f"{100.0 * float(row['precision']):.2f}",
                    "Recall": f"{100.0 * float(row['recall']):.2f}",
                    "F1": f"{100.0 * float(row['f1']):.2f}",
                }
            )


def format_report_table(rows: list[dict[str, object]]) -> str:
    header = "| Experiment | Dataset | Accuracy | Precision | Recall | F1 |\n|---|---|---:|---:|---:|---:|"
    body = [
        (
            f"| {row['experiment']} | {row['dataset']} | "
            f"{100.0 * float(row['accuracy']):.2f}% | "
            f"{100.0 * float(row['precision']):.2f}% | "
            f"{100.0 * float(row['recall']):.2f}% | "
            f"{100.0 * float(row['f1']):.2f}% |"
        )
        for row in rows
    ]
    return "\n".join([header, *body])


def plot_confusion_matrices(rows: list[dict[str, object]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 5))
    if len(rows) == 1:
        axes = [axes]

    for axis, row in zip(axes, rows):
        matrix = np.asarray(row["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{row['experiment']}\n{row['dataset']}")
        axis.set_xticks([0, 1], labels=["Open", "Closed"])
        axis.set_yticks([0, 1], labels=["Open", "Closed"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                axis.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="black")

    figure.suptitle("Enhancer + Frozen Detector Evaluation", fontsize=14)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_metric_bars(rows: list[dict[str, object]], output_path: Path) -> None:
    metrics = list(REPORT_METRICS)
    x = np.arange(len(metrics))
    width = 0.8 / max(len(rows), 1)
    colors = ["#5B8FF9", "#F08C2E", "#5AD8A6", "#9270CA"]

    figure, axis = plt.subplots(figsize=(11, 5.5))
    for index, row in enumerate(rows):
        values = [100.0 * float(row[metric]) for metric in metrics]
        bars = axis.bar(
            x + index * width,
            values,
            width=width,
            label=f"{row['experiment']} ({row['dataset']})",
            color=colors[index % len(colors)],
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 1.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axis.set_xticks(x + width * (len(rows) - 1) / 2, labels=metrics)
    axis.set_ylim(0.0, 100.0)
    axis.set_ylabel("Score (%)")
    axis.set_title("Raw vs Enhanced Detector Performance on Low-light Images")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)

    raw_lowlight_row = next(
        (
            row for row in rows
            if row["experiment"] == "Original Detector" and row["dataset"] == "Low-light"
        ),
        None,
    )
    enhanced_lowlight_row = next(
        (
            row for row in rows
            if row["experiment"] == "Enhancer + Original Detector" and row["dataset"] == "Low-light"
        ),
        None,
    )
    enhanced_finetuned_row = next(
        (
            row for row in rows
            if row["experiment"] == "Enhancer + Fine-tuned Detector"
            and row["dataset"] == "Low-light"
        ),
        None,
    )
    dual_input_row = next(
        (
            row for row in rows
            if row["experiment"] == "Dual Input Detector" and row["dataset"] == "Low-light"
        ),
        None,
    )
    dual_input_row = next(
        (
            row for row in rows
            if row["experiment"] == "Dual Input Detector" and row["dataset"] == "Low-light"
        ),
        None,
    )
    dual_input_row = next(
        (
            row for row in rows
            if row["experiment"] == "Dual Input Detector" and row["dataset"] == "Low-light"
        ),
        None,
    )
    if raw_lowlight_row and enhanced_lowlight_row:
        delta_lines = []
        for metric in metrics:
            delta = 100.0 * (
                float(enhanced_lowlight_row[metric]) - float(raw_lowlight_row[metric])
            )
            delta_lines.append(f"{metric.capitalize()}: {delta:+.2f} pts")
        axis.text(
            1.02,
            0.98,
            "Enhanced vs Raw Low-light\n" + "\n".join(delta_lines),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f7f7f7", "edgecolor": "#d0d0d0"},
        )
    if raw_lowlight_row and enhanced_finetuned_row:
        delta_lines = []
        for metric in metrics:
            delta = 100.0 * (
                float(enhanced_finetuned_row[metric]) - float(raw_lowlight_row[metric])
            )
            delta_lines.append(f"{metric.capitalize()}: {delta:+.2f} pts")
        axis.text(
            1.02,
            0.48,
            "Fine-tuned vs Raw Low-light\n" + "\n".join(delta_lines),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f7f7f7", "edgecolor": "#d0d0d0"},
        )
    if raw_lowlight_row and dual_input_row:
        delta_lines = []
        for metric in metrics:
            delta = 100.0 * (float(dual_input_row[metric]) - float(raw_lowlight_row[metric]))
            delta_lines.append(f"{metric.capitalize()}: {delta:+.2f} pts")
        axis.text(
            1.02,
            0.18,
            "Dual vs Raw Low-light\n" + "\n".join(delta_lines),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f7f7f7", "edgecolor": "#d0d0d0"},
        )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def build_summary_lines(
    *,
    rows: list[dict[str, object]],
    detector_frozen: bool,
    detector_eval: bool,
    detector_checkpoint: str | Path,
    enhancer_checkpoint: str | Path,
    min_positive_rate: float,
    max_positive_rate: float,
) -> list[str]:
    lines = [
        "Enhancer + Frozen Detector Evaluation",
        "====================================",
        "",
        f"Detector checkpoint: {Path(detector_checkpoint).expanduser().resolve()}",
        f"Enhancer checkpoint: {Path(enhancer_checkpoint).expanduser().resolve()}",
        f"Detector frozen: {detector_frozen}",
        f"Detector eval mode: {detector_eval}",
        (
            "Threshold guardrail: predicted closed rate kept within "
            f"[{100.0 * min_positive_rate:.1f}%, {100.0 * max_positive_rate:.1f}%] "
            "when a non-collapsed option exists."
        ),
        "",
    ]

    lines.append("Selected thresholds:")
    for row in rows:
        lines.append(
            f"- {row['experiment']} on {row['dataset']}: {float(row['threshold']):.2f}"
        )
    lines.append("")

    raw_lowlight_row = next(
        (
            row for row in rows
            if row["experiment"] == "Original Detector" and row["dataset"] == "Low-light"
        ),
        None,
    )
    enhanced_lowlight_row = next(
        (
            row for row in rows
            if row["experiment"] == "Enhancer + Original Detector" and row["dataset"] == "Low-light"
        ),
        None,
    )
    enhanced_finetuned_row = next(
        (
            row for row in rows
            if row["experiment"] == "Enhancer + Fine-tuned Detector"
            and row["dataset"] == "Low-light"
        ),
        None,
    )
    dual_input_row = next(
        (
            row for row in rows
            if row["experiment"] == "Dual Input Detector" and row["dataset"] == "Low-light"
        ),
        None,
    )

    if raw_lowlight_row is not None:
        lines.append(
            "Raw low-light detector performance: "
            f"accuracy={100.0 * float(raw_lowlight_row['accuracy']):.2f}%, "
            f"precision={100.0 * float(raw_lowlight_row['precision']):.2f}%, "
            f"recall={100.0 * float(raw_lowlight_row['recall']):.2f}%, "
            f"f1={100.0 * float(raw_lowlight_row['f1']):.2f}%."
        )
    if enhanced_lowlight_row is not None:
        lines.append(
            "Enhanced low-light with original detector: "
            f"accuracy={100.0 * float(enhanced_lowlight_row['accuracy']):.2f}%, "
            f"precision={100.0 * float(enhanced_lowlight_row['precision']):.2f}%, "
            f"recall={100.0 * float(enhanced_lowlight_row['recall']):.2f}%, "
            f"f1={100.0 * float(enhanced_lowlight_row['f1']):.2f}%."
        )
    if enhanced_finetuned_row is not None:
        lines.append(
            "Enhanced low-light with fine-tuned detector: "
            f"accuracy={100.0 * float(enhanced_finetuned_row['accuracy']):.2f}%, "
            f"precision={100.0 * float(enhanced_finetuned_row['precision']):.2f}%, "
            f"recall={100.0 * float(enhanced_finetuned_row['recall']):.2f}%, "
            f"f1={100.0 * float(enhanced_finetuned_row['f1']):.2f}%."
        )
    if dual_input_row is not None:
        lines.append(
            "Dual-input low-light detector: "
            f"accuracy={100.0 * float(dual_input_row['accuracy']):.2f}%, "
            f"precision={100.0 * float(dual_input_row['precision']):.2f}%, "
            f"recall={100.0 * float(dual_input_row['recall']):.2f}%, "
            f"f1={100.0 * float(dual_input_row['f1']):.2f}%."
        )

    if raw_lowlight_row and enhanced_lowlight_row:
        lines.append("")
        lines.append("Improvement from enhancement with the original detector:")
        for metric in REPORT_METRICS:
            delta = 100.0 * (
                float(enhanced_lowlight_row[metric]) - float(raw_lowlight_row[metric])
            )
            word = "improvement" if delta >= 0 else "drop"
            lines.append(f"- {metric.capitalize()}: {delta:+.2f} points ({word}).")

    if raw_lowlight_row and enhanced_finetuned_row:
        lines.append("")
        lines.append("Improvement from enhancement plus detector fine-tuning:")
        for metric in REPORT_METRICS:
            delta = 100.0 * (
                float(enhanced_finetuned_row[metric]) - float(raw_lowlight_row[metric])
            )
            word = "improvement" if delta >= 0 else "drop"
            lines.append(f"- {metric.capitalize()}: {delta:+.2f} points ({word}).")

    if raw_lowlight_row and dual_input_row:
        lines.append("")
        lines.append("Improvement from dual-input detection:")
        for metric in REPORT_METRICS:
            delta = 100.0 * (float(dual_input_row[metric]) - float(raw_lowlight_row[metric]))
            word = "improvement" if delta >= 0 else "drop"
            lines.append(f"- {metric.capitalize()}: {delta:+.2f} points ({word}).")

    return lines


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    detector_checkpoint = load_checkpoint(args.detector_checkpoint)

    model_config = detector_checkpoint["model_config"]
    class_to_idx = detector_checkpoint["class_to_idx"]
    positive_label = resolve_positive_label(class_to_idx)
    train_cfg = detector_checkpoint.get("training_config", {})
    threshold_candidates = [float(value) for value in args.threshold_candidates]
    min_positive_rate = float(args.min_closed_prediction_rate)
    max_positive_rate = float(args.max_closed_prediction_rate)

    original_detector = build_detector_from_checkpoint(
        detector_checkpoint,
        device=device,
        freeze_detector_weights=True,
        print_frozen_layers=True,
    )
    print_gradient_debug_info(original_detector, module_name="detector-after-checkpoint-load")

    enhancer = load_enhancer_model(
        checkpoint_path=args.enhancer_checkpoint,
        device=device,
    )
    original_pipeline = build_enhancer_frozen_detector_pipeline(
        enhancer=enhancer,
        detector=original_detector,
        normalize_detector_input=True,
        keep_detector_eval=True,
        return_enhanced_image_by_default=True,
        return_curve_maps_by_default=False,
        print_frozen_layers=True,
        debug_freeze_state=True,
    ).to(device)
    original_pipeline.eval()

    detector_frozen = original_pipeline.detector_is_frozen()
    detector_eval = not original_pipeline.detector.training
    if not detector_frozen:
        raise RuntimeError("Detector parameters are not frozen.")
    if not detector_eval:
        raise RuntimeError("Detector must remain in eval mode inside the frozen pipeline.")
    verify_detector_is_truly_frozen(original_pipeline.detector, context="pipeline-setup")

    enhanced_detector = None
    enhanced_pipeline = None
    if args.enhanced_detector_checkpoint:
        enhanced_detector_checkpoint = load_checkpoint(args.enhanced_detector_checkpoint)
        enhanced_detector = build_detector_from_checkpoint(
            enhanced_detector_checkpoint,
            device=device,
            freeze_detector_weights=True,
            print_frozen_layers=False,
        )
        print_gradient_debug_info(enhanced_detector, module_name="enhanced-detector-after-load")

        enhanced_pipeline = build_enhancer_frozen_detector_pipeline(
            enhancer=enhancer,
            detector=enhanced_detector,
            normalize_detector_input=True,
            keep_detector_eval=True,
            return_enhanced_image_by_default=True,
            return_curve_maps_by_default=False,
            print_frozen_layers=False,
            debug_freeze_state=True,
        ).to(device)
        enhanced_pipeline.eval()
        verify_detector_is_truly_frozen(
            enhanced_pipeline.detector,
            context="enhanced-detector-pipeline-setup",
        )

    dual_detector = None
    dual_pipeline = None
    if args.dual_detector_checkpoint:
        dual_detector_checkpoint = load_checkpoint(args.dual_detector_checkpoint)
        dual_detector = build_detector_from_checkpoint(
            dual_detector_checkpoint,
            device=device,
            freeze_detector_weights=True,
            print_frozen_layers=False,
        )
        print_gradient_debug_info(dual_detector, module_name="dual-detector-after-load")

        dual_pipeline = build_enhancer_frozen_detector_pipeline(
            enhancer=enhancer,
            detector=dual_detector,
            normalize_detector_input=True,
            keep_detector_eval=True,
            return_enhanced_image_by_default=True,
            return_curve_maps_by_default=False,
            print_frozen_layers=False,
            debug_freeze_state=True,
        ).to(device)
        dual_pipeline.eval()
        verify_detector_is_truly_frozen(
            dual_pipeline.detector,
            context="dual-detector-pipeline-setup",
        )

    criterion = FocalLoss(
        alpha=train_cfg.get("alpha"),
        gamma=float(train_cfg.get("focal_gamma", 2.0)),
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
    )
    image_size = int(model_config["image_size"])
    raw_transform = build_raw_transform(image_size)
    _, normalized_transform = build_transfer_learning_transforms(image_size=image_size)

    rows: list[dict[str, object]] = []
    threshold_sweep_rows: list[dict[str, object]] = []

    if args.clean_root:
        _, clean_loader = create_loader(
            args.clean_root,
            class_to_idx=class_to_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            transform=normalized_transform,
        )
        clean_metrics, clean_threshold_rows = evaluate_detector(
            original_pipeline.detector,
            clean_loader,
            criterion,
            device,
            positive_label,
            threshold_candidates=threshold_candidates,
            min_positive_rate=min_positive_rate,
            max_positive_rate=max_positive_rate,
        )
        rows.append(
            build_result_row(
                experiment="Original Detector",
                dataset="Clean",
                metrics=clean_metrics,
            )
        )
        for threshold_row in clean_threshold_rows:
            threshold_sweep_rows.append(
                build_threshold_row(
                    experiment="Original Detector",
                    dataset="Clean",
                    metrics=threshold_row,
                )
            )
        verify_detector_is_truly_frozen(original_pipeline.detector, context="clean-reference-evaluation")
        print(format_classifier_metric_line("Clean", clean_metrics))

    _, lowlight_detector_loader = create_loader(
        args.lowlight_root,
        class_to_idx=class_to_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=normalized_transform,
    )
    raw_lowlight_metrics, raw_threshold_rows = evaluate_detector(
        original_pipeline.detector,
        lowlight_detector_loader,
        criterion,
        device,
        positive_label,
        threshold_candidates=threshold_candidates,
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
    )
    rows.append(
        build_result_row(
            experiment="Original Detector",
            dataset="Low-light",
            metrics=raw_lowlight_metrics,
        )
    )
    for threshold_row in raw_threshold_rows:
        threshold_sweep_rows.append(
            build_threshold_row(
                experiment="Original Detector",
                dataset="Low-light",
                metrics=threshold_row,
            )
        )
    verify_detector_is_truly_frozen(original_pipeline.detector, context="raw-lowlight-evaluation")

    _, lowlight_enhancer_loader = create_loader(
        args.lowlight_root,
        class_to_idx=class_to_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=raw_transform,
    )
    enhanced_metrics, enhanced_threshold_rows = evaluate_frozen_pipeline(
        original_pipeline,
        lowlight_enhancer_loader,
        criterion,
        device,
        positive_label,
        threshold_candidates=threshold_candidates,
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
    )
    rows.append(
        build_result_row(
            experiment="Enhancer + Original Detector",
            dataset="Low-light",
            metrics=enhanced_metrics,
        )
    )
    for threshold_row in enhanced_threshold_rows:
        threshold_sweep_rows.append(
            build_threshold_row(
                experiment="Enhancer + Original Detector",
                dataset="Low-light",
                metrics=threshold_row,
            )
        )
    verify_detector_is_truly_frozen(original_pipeline.detector, context="enhanced-lowlight-evaluation")

    enhanced_detector_metrics = None
    if enhanced_pipeline is not None:
        enhanced_detector_metrics, finetuned_threshold_rows = evaluate_frozen_pipeline(
            enhanced_pipeline,
            lowlight_enhancer_loader,
            criterion,
            device,
            positive_label,
            threshold_candidates=threshold_candidates,
            min_positive_rate=min_positive_rate,
            max_positive_rate=max_positive_rate,
        )
        rows.append(
            build_result_row(
                experiment="Enhancer + Fine-tuned Detector",
                dataset="Low-light",
                metrics=enhanced_detector_metrics,
            )
        )
        for threshold_row in finetuned_threshold_rows:
            threshold_sweep_rows.append(
                build_threshold_row(
                    experiment="Enhancer + Fine-tuned Detector",
                    dataset="Low-light",
                    metrics=threshold_row,
                )
            )
        verify_detector_is_truly_frozen(
            enhanced_pipeline.detector,
            context="enhanced-finetuned-lowlight-evaluation",
        )

    print(f"Evaluating on device: {device}")
    print(f"Classes: {class_to_idx}")
    print(
        "Detector freeze check: "
        f"frozen={detector_frozen} | eval_mode={detector_eval}"
    )
    print(format_classifier_metric_line("LowLight", raw_lowlight_metrics))
    print(format_classifier_metric_line("Enhanced", enhanced_metrics))
    if enhanced_detector_metrics is not None:
        print(format_classifier_metric_line("Enh+FT", enhanced_detector_metrics))

    dual_detector_metrics = None
    if dual_pipeline is not None:
        dual_detector_metrics, dual_threshold_rows = evaluate_frozen_pipeline(
            dual_pipeline,
            lowlight_enhancer_loader,
            criterion,
            device,
            positive_label,
            threshold_candidates=threshold_candidates,
            min_positive_rate=min_positive_rate,
            max_positive_rate=max_positive_rate,
        )
        rows.append(
            build_result_row(
                experiment="Dual Input Detector",
                dataset="Low-light",
                metrics=dual_detector_metrics,
            )
        )
        for threshold_row in dual_threshold_rows:
            threshold_sweep_rows.append(
                build_threshold_row(
                    experiment="Dual Input Detector",
                    dataset="Low-light",
                    metrics=threshold_row,
                )
            )
        verify_detector_is_truly_frozen(
            dual_pipeline.detector,
            context="dual-input-lowlight-evaluation",
        )
        print(format_classifier_metric_line("Dual", dual_detector_metrics))

    output_dir = Path(args.output_dir).expanduser().resolve()
    save_report_table_csv(rows, output_dir / "experiment_results.csv")
    save_csv(rows, output_dir / "evaluation_results.csv")
    save_threshold_sweep_csv(threshold_sweep_rows, output_dir / "threshold_sweep.csv")
    plot_confusion_matrices(rows, output_dir / "confusion_matrices.png")
    plot_metric_bars(rows, output_dir / "metric_comparison.png")

    summary_lines = build_summary_lines(
        rows=rows,
        detector_frozen=detector_frozen,
        detector_eval=detector_eval,
        detector_checkpoint=args.detector_checkpoint,
        enhancer_checkpoint=args.enhancer_checkpoint,
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
    )
    (output_dir / "evaluation_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\nReport Table")
    print(format_report_table(rows))
    print(f"\nSaved evaluation artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
