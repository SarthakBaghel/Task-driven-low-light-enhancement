#!/usr/bin/env python3
"""Compare a trained baseline CNN on clean and low-light eye-state datasets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import EyeStateDataset, build_default_transform
from metrics import (
    RunningClassificationMetrics,
    extract_binary_confusion_terms,
    format_metric_line,
    resolve_positive_label,
)
from models.baseline_cnn import BaselineCNN


COMPARISON_METRICS = ("accuracy", "precision", "recall", "f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained baseline CNN on clean and low-light datasets, "
            "then export report-ready metrics and plots."
        ),
    )
    parser.add_argument("checkpoint", type=str, help="Path to the saved model checkpoint.")
    parser.add_argument(
        "clean_data_root",
        type=str,
        help="Dataset root for the clean evaluation set.",
    )
    parser.add_argument(
        "--lowlight-data-root",
        type=str,
        help="Dataset root for the degraded low-light evaluation set.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Optional subfolder to use when a dataset root contains split folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_report",
        help="Directory where the CSV, plots, and summary text will be saved.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Fallback input image size after resizing.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader worker processes. Use 0 for maximum compatibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Evaluation device: auto, cpu, cuda, or a specific CUDA device like cuda:0.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested evaluation device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_eval_root(data_root: str | Path, split: str | None) -> Path:
    """Resolve the directory containing the class folders to evaluate."""
    root_path = Path(data_root).expanduser().resolve()
    if split:
        split_root = root_path / split
        if split_root.is_dir():
            return split_root
    return root_path


def load_checkpoint_and_model(
    checkpoint_path: str | Path,
    device: torch.device,
    fallback_image_size: int,
) -> tuple[dict[str, object], BaselineCNN, dict[str, int], list[str], int]:
    """Load the evaluation checkpoint and reconstruct the trained model."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", {"open": 0, "closed": 1})
    if not isinstance(class_to_idx, dict) or len(class_to_idx) != 2:
        raise ValueError(
            "BaselineCNN expects a checkpoint with exactly 2 classes, "
            f"but found {class_to_idx}."
        )

    model_config = checkpoint.get("model_config", {})
    image_size = int(model_config.get("image_size", fallback_image_size))
    classes = checkpoint.get(
        "classes",
        [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])],
    )

    model = BaselineCNN(
        num_classes=int(model_config.get("num_classes", 2)),
        dropout_rate=float(model_config.get("dropout_rate", 0.3)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint, model, class_to_idx, list(classes), image_size


def create_eval_dataloader(
    data_root: str | Path,
    *,
    split: str | None,
    class_to_idx: dict[str, int],
    batch_size: int,
    image_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[Path, EyeStateDataset, DataLoader]:
    """Create the dataset and dataloader for one evaluation root."""
    eval_root = resolve_eval_root(data_root, split)
    dataset = EyeStateDataset(
        root=eval_root,
        class_to_idx=class_to_idx,
        transform=build_default_transform(image_size),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    return eval_root, dataset, dataloader


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    positive_label: int,
    num_classes: int,
) -> dict[str, float | int | torch.Tensor]:
    """Run evaluation over one dataset split."""
    metrics = RunningClassificationMetrics(
        positive_label=positive_label,
        num_classes=num_classes,
    )
    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            metrics.update(
                loss=loss.item(),
                predictions=predictions,
                targets=labels,
            )

    return metrics.compute()


def evaluate_dataset(
    *,
    dataset_name: str,
    data_root: str | Path,
    split: str | None,
    class_to_idx: dict[str, int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    model: nn.Module,
    criterion: nn.Module,
    positive_label: int,
) -> dict[str, object]:
    """Evaluate one dataset and return metrics formatted for reporting."""
    eval_root, dataset, dataloader = create_eval_dataloader(
        data_root=data_root,
        split=split,
        class_to_idx=class_to_idx,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        device=device,
    )
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        positive_label=positive_label,
        num_classes=len(class_to_idx),
    )
    confusion_matrix = metrics["confusion_matrix"]
    confusion_terms = extract_binary_confusion_terms(
        confusion_matrix=confusion_matrix,
        positive_label=positive_label,
    )

    return {
        "dataset": dataset_name,
        "eval_root": str(eval_root),
        "num_samples": int(metrics["num_samples"]),
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "confusion_matrix": confusion_matrix,
        **confusion_terms,
        "class_distribution": dataset.get_class_distribution(),
    }


def format_confusion_matrix(confusion_matrix: torch.Tensor, class_names: list[str]) -> str:
    """Return a plain-text confusion matrix table."""
    header = " " * 14 + "".join(f"{name:>12}" for name in class_names)
    rows = [header]

    for row_index, class_name in enumerate(class_names):
        counts = "".join(
            f"{int(confusion_matrix[row_index, column_index].item()):>12}"
            for column_index in range(len(class_names))
        )
        rows.append(f"{class_name:<14}{counts}")

    return "\n".join(rows)


def save_results_csv(results: list[dict[str, object]], csv_path: Path) -> None:
    """Save evaluation results to a CSV file."""
    fieldnames = [
        "dataset",
        "eval_root",
        "num_samples",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "accuracy_pct",
        "precision_pct",
        "recall_pct",
        "f1_pct",
        "tn",
        "fp",
        "fn",
        "tp",
        "confusion_matrix",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            confusion_matrix = result["confusion_matrix"]
            writer.writerow(
                {
                    "dataset": result["dataset"],
                    "eval_root": result["eval_root"],
                    "num_samples": result["num_samples"],
                    "loss": f"{result['loss']:.4f}",
                    "accuracy": f"{result['accuracy']:.4f}",
                    "precision": f"{result['precision']:.4f}",
                    "recall": f"{result['recall']:.4f}",
                    "f1": f"{result['f1']:.4f}",
                    "accuracy_pct": f"{result['accuracy'] * 100.0:.2f}",
                    "precision_pct": f"{result['precision'] * 100.0:.2f}",
                    "recall_pct": f"{result['recall'] * 100.0:.2f}",
                    "f1_pct": f"{result['f1'] * 100.0:.2f}",
                    "tn": result["tn"],
                    "fp": result["fp"],
                    "fn": result["fn"],
                    "tp": result["tp"],
                    "confusion_matrix": str(confusion_matrix.tolist()),
                }
            )


def plot_confusion_matrices(
    results: list[dict[str, object]],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Plot clean and low-light confusion matrices side by side."""
    figure, axes = plt.subplots(
        1,
        len(results),
        figsize=(6 * len(results), 5),
        constrained_layout=True,
    )
    if len(results) == 1:
        axes = [axes]

    heatmap = None
    for axis, result in zip(axes, results):
        confusion_matrix = result["confusion_matrix"].cpu().numpy()
        heatmap = axis.imshow(confusion_matrix, cmap="Blues")
        axis.set_title(
            f"{result['dataset']} Confusion Matrix\n"
            f"Accuracy: {result['accuracy'] * 100.0:.2f}%"
        )
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(range(len(class_names)))
        axis.set_xticklabels([name.title() for name in class_names])
        axis.set_yticks(range(len(class_names)))
        axis.set_yticklabels([name.title() for name in class_names])

        max_value = max(confusion_matrix.max(), 1)
        for row_index in range(confusion_matrix.shape[0]):
            for column_index in range(confusion_matrix.shape[1]):
                value = int(confusion_matrix[row_index, column_index])
                color = "white" if value > max_value / 2 else "black"
                axis.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=11,
                    fontweight="bold",
                )

    if heatmap is not None:
        figure.colorbar(heatmap, ax=axes, shrink=0.85, label="Count")

    figure.suptitle("Baseline CNN Confusion Matrices", fontsize=14, fontweight="bold")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_metric_comparison(results: list[dict[str, object]], output_path: Path) -> None:
    """Plot a bar chart comparing clean and low-light metrics."""
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    positions = list(range(len(COMPARISON_METRICS)))
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)

    series_count = len(results)
    width = 0.8 / max(series_count, 1)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    for series_index, result in enumerate(results):
        offset = (series_index - (series_count - 1) / 2.0) * width
        values = [result[metric] * 100.0 for metric in COMPARISON_METRICS]
        bars = axis.bar(
            [position + offset for position in positions],
            values,
            width=width,
            label=result["dataset"],
            color=colors[series_index % len(colors)],
        )
        for bar in bars:
            height = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1.0,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Score (%)")
    axis.set_title("Clean vs Low-Light Performance Comparison", fontweight="bold")
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    axis.legend()

    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def describe_metric_change(metric_name: str, clean_value: float, lowlight_value: float) -> str:
    """Describe the change in one metric between clean and low-light data."""
    change = (lowlight_value - clean_value) * 100.0
    if abs(change) < 1e-9:
        return f"{metric_name} was unchanged"
    if change < 0.0:
        return f"{metric_name} dropped by {abs(change):.2f} percentage points"
    return f"{metric_name} improved by {change:.2f} percentage points"


def build_report_summary(
    clean_result: dict[str, object],
    lowlight_result: dict[str, object] | None,
    *,
    positive_class_name: str,
    negative_class_name: str,
) -> str:
    """Build a short report-style summary describing performance changes."""
    clean_line = (
        f"On clean images, the baseline CNN achieved "
        f"{clean_result['accuracy'] * 100.0:.2f}% accuracy, "
        f"{clean_result['precision'] * 100.0:.2f}% precision, "
        f"{clean_result['recall'] * 100.0:.2f}% recall, and "
        f"{clean_result['f1'] * 100.0:.2f}% F1."
    )

    if lowlight_result is None:
        return clean_line

    lowlight_line = (
        f"On low-light images, performance changed to "
        f"{lowlight_result['accuracy'] * 100.0:.2f}% accuracy, "
        f"{lowlight_result['precision'] * 100.0:.2f}% precision, "
        f"{lowlight_result['recall'] * 100.0:.2f}% recall, and "
        f"{lowlight_result['f1'] * 100.0:.2f}% F1."
    )

    metric_changes = ", ".join(
        [
            describe_metric_change("accuracy", clean_result["accuracy"], lowlight_result["accuracy"]),
            describe_metric_change(
                "precision",
                clean_result["precision"],
                lowlight_result["precision"],
            ),
            describe_metric_change("recall", clean_result["recall"], lowlight_result["recall"]),
            describe_metric_change("F1", clean_result["f1"], lowlight_result["f1"]),
        ]
    )

    confusion_notes = []
    if lowlight_result["fn"] > clean_result["fn"]:
        confusion_notes.append(
            f"missed {positive_class_name} detections increased from "
            f"{clean_result['fn']} to {lowlight_result['fn']}"
        )
    if lowlight_result["fp"] > clean_result["fp"]:
        confusion_notes.append(
            f"false {positive_class_name} alarms on {negative_class_name} images increased "
            f"from {clean_result['fp']} to {lowlight_result['fp']}"
        )
    if not confusion_notes:
        confusion_notes.append(
            "the confusion matrices show a similar error profile across lighting conditions"
        )

    return f"{clean_line} {lowlight_line} Relative to clean data, {metric_changes}. In the confusion matrices, {' and '.join(confusion_notes)}."


def write_summary_text(summary: str, output_path: Path) -> None:
    """Write the report summary to disk."""
    output_path.write_text(summary + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    _, model, class_to_idx, class_names, image_size = load_checkpoint_and_model(
        checkpoint_path=args.checkpoint,
        device=device,
        fallback_image_size=args.image_size,
    )
    positive_label = resolve_positive_label(class_to_idx)
    positive_class_name = next(
        class_name for class_name, index in class_to_idx.items() if index == positive_label
    )
    negative_class_name = next(
        class_name for class_name, index in class_to_idx.items() if index != positive_label
    )

    criterion = nn.CrossEntropyLoss()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        evaluate_dataset(
            dataset_name="Clean",
            data_root=args.clean_data_root,
            split=args.split,
            class_to_idx=class_to_idx,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            model=model,
            criterion=criterion,
            positive_label=positive_label,
        )
    ]

    if args.lowlight_data_root:
        results.append(
            evaluate_dataset(
                dataset_name="Low-light",
                data_root=args.lowlight_data_root,
                split=args.split,
                class_to_idx=class_to_idx,
                image_size=image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                model=model,
                criterion=criterion,
                positive_label=positive_label,
            )
        )

    csv_path = output_dir / "evaluation_results.csv"
    confusion_plot_path = output_dir / "confusion_matrices.png"
    metric_plot_path = output_dir / "metric_comparison.png"
    summary_path = output_dir / "evaluation_summary.txt"

    save_results_csv(results, csv_path)
    plot_confusion_matrices(results, class_names, confusion_plot_path)
    plot_metric_comparison(results, metric_plot_path)

    lowlight_result = results[1] if len(results) > 1 else None
    summary = build_report_summary(
        clean_result=results[0],
        lowlight_result=lowlight_result,
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name,
    )
    write_summary_text(summary, summary_path)

    print(f"Evaluating on device: {device}")
    print(f"Classes: {class_to_idx}")
    for result in results:
        print(f"\n{result['dataset']} dataset")
        print(f"Root: {result['eval_root']}")
        print(f"Samples: {result['num_samples']}")
        print(format_metric_line(str(result["dataset"]), result))
        print("Confusion matrix (rows=true, cols=pred):")
        print(format_confusion_matrix(result["confusion_matrix"], class_names))

    print("\nReport summary:")
    print(summary)
    print(f"\nSaved CSV report to: {csv_path}")
    print(f"Saved confusion matrix figure to: {confusion_plot_path}")
    print(f"Saved metric comparison figure to: {metric_plot_path}")
    print(f"Saved summary text to: {summary_path}")


if __name__ == "__main__":
    main()
