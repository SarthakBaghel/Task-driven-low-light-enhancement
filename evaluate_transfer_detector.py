#!/usr/bin/env python3
"""Evaluate one trained detector on clean and low-light datasets.

This script is designed for baseline degradation analysis in the project report:
the same trained detector checkpoint is applied to clean and degraded
low-light data, and the resulting metric drop is exported in both tabular and
figure form.
"""

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

from dataset import EyeStateDataset
from losses.focal_loss import FocalLoss
from models.detector import build_detector
from utils.classifier_metrics import (
    DEFAULT_MAX_CLOSED_PREDICTION_RATE,
    DEFAULT_MIN_CLOSED_PREDICTION_RATE,
    ProbabilityAccumulator,
    format_classifier_metric_line,
    resolve_positive_label,
)
from utils.classifier_transforms import build_transfer_learning_transforms


REPORT_METRICS = ("accuracy", "precision", "recall", "f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a transfer-learning eye-state detector.")
    parser.add_argument("checkpoint", type=str, help="Path to a saved detector checkpoint.")
    parser.add_argument("clean_root", type=str, help="Clean evaluation dataset root.")
    parser.add_argument(
        "--lowlight-root",
        type=str,
        help="Optional low-light evaluation root using the same class mapping.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--retune-threshold-on-clean",
        action="store_true",
        help="Tune the closed-eye threshold on the clean evaluation set instead of using the saved threshold.",
    )
    parser.add_argument(
        "--min-closed-prediction-rate",
        type=float,
        default=DEFAULT_MIN_CLOSED_PREDICTION_RATE,
        help="Reject retuned thresholds that predict too few closed-eye samples unless no balanced option exists.",
    )
    parser.add_argument(
        "--max-closed-prediction-rate",
        type=float,
        default=DEFAULT_MAX_CLOSED_PREDICTION_RATE,
        help="Reject retuned thresholds that predict too many closed-eye samples unless no balanced option exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/evaluation_transfer",
        help="Directory where CSV and plots will be written.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_checkpoint(path: str | Path) -> dict:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def create_loader(
    dataset_root: str | Path,
    *,
    class_to_idx: dict[str, int],
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> tuple[EyeStateDataset, DataLoader]:
    _, val_transform = build_transfer_learning_transforms(image_size=image_size)
    dataset = EyeStateDataset(root=dataset_root, class_to_idx=class_to_idx, transform=val_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return dataset, loader


def evaluate_loader(
    model: torch.nn.Module,
    dataloader,
    criterion: torch.nn.Module,
    device: torch.device,
    positive_label: int,
    *,
    threshold: float | None = None,
    tune_threshold: bool = False,
    threshold_objective: str = "f1",
    min_positive_rate: float = DEFAULT_MIN_CLOSED_PREDICTION_RATE,
    max_positive_rate: float = DEFAULT_MAX_CLOSED_PREDICTION_RATE,
) -> dict[str, float]:
    accumulator = ProbabilityAccumulator()
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            accumulator.update(
                loss=float(loss.detach().item()),
                logits=logits,
                targets=labels,
                positive_label=positive_label,
            )
    return accumulator.compute(
        positive_label=positive_label,
        threshold=threshold,
        tune_threshold=tune_threshold,
        threshold_objective=threshold_objective,
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
    )


def save_results_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report_table_rows(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    """Return a compact report-ready table with percentage metrics."""
    return [
        {
            "Dataset": str(row["dataset"]),
            "Accuracy": f"{100.0 * float(row['accuracy']):.2f}",
            "Precision": f"{100.0 * float(row['precision']):.2f}",
            "Recall": f"{100.0 * float(row['recall']):.2f}",
            "F1": f"{100.0 * float(row['f1']):.2f}",
        }
        for row in rows
    ]


def save_experiment_results_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Save the report table required by the project write-up."""
    table_rows = build_report_table_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Dataset", "Accuracy", "Precision", "Recall", "F1"],
        )
        writer.writeheader()
        writer.writerows(table_rows)


def format_report_table(rows: list[dict[str, object]]) -> str:
    """Return a markdown-style summary table for console output."""
    table_rows = build_report_table_rows(rows)
    header = "| Dataset | Accuracy | Precision | Recall | F1 |\n|---|---:|---:|---:|---:|"
    body = [
        (
            f"| {row['Dataset']} | {row['Accuracy']}% | {row['Precision']}% | "
            f"{row['Recall']}% | {row['F1']}% |"
        )
        for row in table_rows
    ]
    return "\n".join([header, *body])


def build_drop_summary(rows: list[dict[str, object]]) -> list[str]:
    """Summarize the performance change from clean to low-light."""
    if len(rows) < 2:
        return ["Only one dataset was evaluated, so no degradation comparison was computed."]

    clean_row = next((row for row in rows if str(row["dataset"]).lower() == "clean"), rows[0])
    lowlight_row = next(
        (row for row in rows if str(row["dataset"]).lower() == "low-light"),
        rows[1],
    )

    lines = [
        (
            "Low-light degradation summary: "
            f"accuracy {100.0 * float(clean_row['accuracy']):.2f}% -> "
            f"{100.0 * float(lowlight_row['accuracy']):.2f}%."
        )
    ]
    for metric in REPORT_METRICS:
        delta = 100.0 * (float(lowlight_row[metric]) - float(clean_row[metric]))
        direction = "drop" if delta < 0 else "change"
        lines.append(f"{metric.capitalize()}: {delta:+.2f} points ({direction}).")
    return lines


def plot_confusion_matrices(rows: list[dict[str, object]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 5))
    if len(rows) == 1:
        axes = [axes]
    for axis, row in zip(axes, rows):
        matrix = np.asarray(row["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{row['dataset']} Confusion Matrix")
        axis.set_xticks([0, 1], labels=["Open", "Closed"])
        axis.set_yticks([0, 1], labels=["Open", "Closed"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                axis.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="black")
    figure.suptitle("Detector Confusion Matrices", fontsize=14)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_metric_bars(rows: list[dict[str, object]], output_path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(rows), 1)
    figure, axis = plt.subplots(figsize=(10, 5))
    colors = ["#5B8FF9", "#F08C2E", "#5AD8A6", "#9270CA"]
    for index, row in enumerate(rows):
        values = [100.0 * float(row[metric]) for metric in metrics]
        bars = axis.bar(
            x + index * width,
            values,
            width=width,
            label=row["dataset"],
            color=colors[index % len(colors)],
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 1.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axis.set_xticks(x + width * (len(rows) - 1) / 2, labels=metrics)
    axis.set_ylim(0.0, 100.0)
    axis.set_ylabel("Score (%)")
    axis.set_title("Detector Performance: Clean vs Low-light")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    if len(rows) >= 2:
        clean_row = next((row for row in rows if str(row["dataset"]).lower() == "clean"), rows[0])
        lowlight_row = next(
            (row for row in rows if str(row["dataset"]).lower() == "low-light"),
            rows[1],
        )
        drop_lines = []
        for metric in metrics:
            delta = 100.0 * (float(lowlight_row[metric]) - float(clean_row[metric]))
            drop_lines.append(f"{metric.capitalize()}: {delta:+.2f} pts")
        axis.text(
            1.02,
            0.98,
            "Low-light vs Clean\n" + "\n".join(drop_lines),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f7f7f7", "edgecolor": "#d0d0d0"},
        )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    device = resolve_device(args.device)

    model_config = checkpoint["model_config"]
    class_to_idx = checkpoint["class_to_idx"]
    positive_label = resolve_positive_label(class_to_idx)

    model = build_detector(
        backbone=model_config["backbone"],
        num_classes=model_config["num_classes"],
        image_size=model_config["image_size"],
        mobilenet_trainable_blocks=model_config.get("mobilenet_trainable_blocks", 3),
        resnet_trainable_layers=model_config.get("resnet_trainable_layers", 1),
        use_pretrained=model_config.get("use_pretrained", True),
        allow_pretrained_fallback=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_cfg = checkpoint.get("training_config", {})
    alpha = train_cfg.get("alpha")
    criterion = FocalLoss(
        alpha=alpha,
        gamma=float(train_cfg.get("focal_gamma", 2.0)),
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
    )

    _, clean_loader = create_loader(
        args.clean_root,
        class_to_idx=class_to_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=int(model_config["image_size"]),
    )
    clean_metrics = evaluate_loader(
        model,
        clean_loader,
        criterion,
        device,
        positive_label,
        threshold=None if args.retune_threshold_on_clean else float(checkpoint.get("best_threshold", 0.5)),
        tune_threshold=args.retune_threshold_on_clean,
        threshold_objective=str(train_cfg.get("threshold_objective", "f1")),
        min_positive_rate=float(train_cfg.get("min_closed_prediction_rate", args.min_closed_prediction_rate)),
        max_positive_rate=float(train_cfg.get("max_closed_prediction_rate", args.max_closed_prediction_rate)),
    )
    threshold = float(clean_metrics["threshold"])

    rows = [
        {
            "dataset": "Clean",
            "root": str(Path(args.clean_root).expanduser().resolve()),
            **clean_metrics,
        }
    ]

    print(f"Evaluating on device: {device}")
    print(f"Classes: {class_to_idx}")
    print(format_classifier_metric_line("Clean", clean_metrics))

    if args.lowlight_root:
        _, lowlight_loader = create_loader(
            args.lowlight_root,
            class_to_idx=class_to_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=int(model_config["image_size"]),
        )
        lowlight_metrics = evaluate_loader(
            model,
            lowlight_loader,
            criterion,
            device,
            positive_label,
            threshold=threshold,
        )
        rows.append(
            {
                "dataset": "Low-light",
                "root": str(Path(args.lowlight_root).expanduser().resolve()),
                **lowlight_metrics,
            }
        )
        print(format_classifier_metric_line("LowLight", lowlight_metrics))

    output_dir = Path(args.output_dir).expanduser().resolve()
    summary_lines = [
        f"Using threshold={threshold:.2f} for the closed-eye class.",
        f"Closed-eye recall on clean data: {clean_metrics['closed_recall']:.4f}.",
    ]
    if len(rows) > 1:
        summary_lines.append(
            f"Closed-eye recall on low-light data: {rows[1]['closed_recall']:.4f}."
        )
        summary_lines.extend(build_drop_summary(rows))

    csv_rows = []
    for row in rows:
        csv_row = dict(row)
        csv_row["confusion_matrix"] = np.asarray(csv_row["confusion_matrix"]).tolist()
        csv_rows.append(csv_row)

    save_experiment_results_csv(rows, output_dir / "experiment_results.csv")
    print(f"Saved evaluation report to: {output_dir}")
    save_results_csv(csv_rows, output_dir / "evaluation_results.csv")
    plot_confusion_matrices(rows, output_dir / "confusion_matrices.png")
    plot_metric_bars(rows, output_dir / "metric_comparison.png")
    (output_dir / "evaluation_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nReport Table")
    print(format_report_table(rows))
    if len(rows) > 1:
        print("\nPerformance Drop")
        for line in build_drop_summary(rows):
            print(line)
    print(f"\nSaved compact CSV to: {output_dir / 'experiment_results.csv'}")
    print(f"Saved detailed CSV to: {output_dir / 'evaluation_results.csv'}")


if __name__ == "__main__":
    main()
