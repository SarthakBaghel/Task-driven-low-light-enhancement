#!/usr/bin/env python3
"""Generate report-ready comparison tables and figures for the project."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import EyeStateDataset, IMAGENET_MEAN, IMAGENET_STD, build_default_transform
from inference_enhancer import load_model as load_enhancer_model
from metrics import RunningClassificationMetrics, extract_binary_confusion_terms, resolve_positive_label
from models.baseline_cnn import BaselineCNN
from models.joint_model import build_joint_model


REPORT_METRICS = ("accuracy", "precision", "recall", "f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report-ready experiment comparisons and figures.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        required=True,
        help="Baseline CNN checkpoint used for clean, low-light, and enhancer-only evaluation.",
    )
    parser.add_argument(
        "--clean-root",
        type=str,
        required=True,
        help="Clean dataset root or parent root containing a split subfolder.",
    )
    parser.add_argument(
        "--lowlight-root",
        type=str,
        required=True,
        help="Low-light dataset root or parent root containing a split subfolder.",
    )
    parser.add_argument(
        "--enhancer-checkpoint",
        type=str,
        help="Optional trained enhancer checkpoint used for the enhancer-only experiment.",
    )
    parser.add_argument(
        "--joint-checkpoint",
        type=str,
        help="Optional joint enhancer-detector checkpoint for end-to-end evaluation.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Optional dataset split subfolder to use when roots contain train/val folders.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Evaluation device: auto, cpu, cuda, or a specific device like cuda:0.",
    )
    parser.add_argument(
        "--fps-warmup-batches",
        type=int,
        default=2,
        help="Warmup batches skipped before FPS timing starts.",
    )
    parser.add_argument(
        "--fps-max-batches",
        type=int,
        default=20,
        help="Maximum timed batches used for FPS benchmarking.",
    )
    parser.add_argument(
        "--baseline-history",
        type=str,
        help="Optional training-history JSON/CSV file for the baseline model.",
    )
    parser.add_argument(
        "--enhancer-history",
        type=str,
        help="Optional training-history JSON/CSV file for the enhancer model.",
    )
    parser.add_argument(
        "--joint-history",
        type=str,
        help="Optional training-history JSON/CSV file for the joint model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where report-ready tables and figures will be saved.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="Figure DPI used for PPT/report-ready image exports.",
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


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def safe_torch_load(path: str | Path, *, map_location: str | torch.device = "cpu"):
    resolved_path = Path(path).expanduser().resolve()
    try:
        return torch.load(resolved_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(resolved_path, map_location=map_location)


def resolve_eval_root(data_root: str | Path, split: str | None) -> Path:
    root_path = Path(data_root).expanduser().resolve()
    if split:
        split_root = root_path / split
        if split_root.is_dir():
            return split_root
    return root_path


def build_raw_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def create_loader(
    root: str | Path,
    *,
    split: str | None,
    class_to_idx: dict[str, int],
    batch_size: int,
    num_workers: int,
    transform,
    device: torch.device,
) -> tuple[Path, EyeStateDataset, DataLoader]:
    eval_root = resolve_eval_root(root, split)
    dataset = EyeStateDataset(
        root=eval_root,
        class_to_idx=class_to_idx,
        transform=transform,
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


def load_baseline_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    model_config = checkpoint.get("model_config", {})
    model = BaselineCNN(
        num_classes=int(model_config.get("num_classes", 2)),
        dropout_rate=float(model_config.get("dropout_rate", 0.3)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    image_size = int(model_config.get("image_size", 224))
    return checkpoint, model, class_to_idx, image_size


def load_joint_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    model = build_joint_model(
        image_size=int(model_config.get("image_size", 224)),
        num_classes=int(model_config.get("num_classes", 2)),
        detector_backbone=str(model_config.get("detector_backbone", "custom")),
        return_curve_maps_by_default=bool(model_config.get("return_curve_maps_by_default", True)),
        strict_input_size_check=bool(model_config.get("strict_input_size_check", True)),
        debug=bool(model_config.get("debug", False)),
        enhancer_kwargs=dict(model_config.get("enhancer_kwargs", {})),
        detector_kwargs=dict(model_config.get("detector_kwargs", {})),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model, checkpoint["class_to_idx"], int(model_config.get("image_size", 224))


def normalize_for_imagenet(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


def benchmark_fps(
    dataloader: DataLoader,
    device: torch.device,
    pipeline_fn: Callable[[torch.Tensor], object],
    *,
    warmup_batches: int,
    max_batches: int,
) -> float:
    timed_batches = 0
    total_samples = 0
    total_time = 0.0

    with torch.no_grad():
        for batch_index, (images, _) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            synchronize_device(device)
            start_time = time.perf_counter()
            _ = pipeline_fn(images)
            synchronize_device(device)
            elapsed = time.perf_counter() - start_time

            if batch_index >= warmup_batches:
                timed_batches += 1
                total_samples += images.shape[0]
                total_time += elapsed
                if timed_batches >= max_batches:
                    break

    if total_time <= 0.0:
        return 0.0
    return total_samples / total_time


def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    positive_label: int,
) -> dict[str, object]:
    metrics = RunningClassificationMetrics(positive_label=positive_label)
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)
            metrics.update(loss=float(loss.detach().item()), predictions=predictions, targets=labels)
    return metrics.compute()


def evaluate_enhancer_detector(
    enhancer: nn.Module,
    detector: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    positive_label: int,
) -> dict[str, object]:
    metrics = RunningClassificationMetrics(positive_label=positive_label)
    enhancer.eval()
    detector.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            enhanced_images, _ = enhancer(images)
            logits = detector(normalize_for_imagenet(enhanced_images))
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)
            metrics.update(loss=float(loss.detach().item()), predictions=predictions, targets=labels)
    return metrics.compute()


def evaluate_joint_model(
    joint_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    positive_label: int,
) -> dict[str, object]:
    metrics = RunningClassificationMetrics(positive_label=positive_label)
    criterion = nn.CrossEntropyLoss()
    joint_model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = joint_model(images, return_curve_maps=False)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)
            metrics.update(loss=float(loss.detach().item()), predictions=predictions, targets=labels)
    return metrics.compute()


def row_from_metrics(
    experiment: str,
    dataset_name: str,
    metrics: dict[str, object],
    fps: float,
    *,
    positive_label: int = 1,
) -> dict[str, object]:
    confusion_matrix = metrics["confusion_matrix"]
    if isinstance(confusion_matrix, torch.Tensor):
        confusion_matrix = confusion_matrix.detach().cpu()
    confusion_terms = extract_binary_confusion_terms(confusion_matrix, positive_label=positive_label)
    return {
        "experiment": experiment,
        "dataset": dataset_name,
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "fps": float(fps),
        "tn": confusion_terms["tn"],
        "fp": confusion_terms["fp"],
        "fn": confusion_terms["fn"],
        "tp": confusion_terms["tp"],
        "confusion_matrix": confusion_matrix.tolist() if isinstance(confusion_matrix, torch.Tensor) else confusion_matrix,
    }


def save_comparison_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["experiment", "dataset", "accuracy", "precision", "recall", "f1", "fps", "tn", "fp", "fn", "tp", "confusion_matrix"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_markdown_table(rows: list[dict[str, object]], output_path: Path) -> None:
    header = "| Experiment | Dataset | Accuracy | Precision | Recall | F1 | FPS |\n|---|---:|---:|---:|---:|---:|---:|\n"
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['dataset']} | "
            f"{100.0 * float(row['accuracy']):.2f}% | "
            f"{100.0 * float(row['precision']):.2f}% | "
            f"{100.0 * float(row['recall']):.2f}% | "
            f"{100.0 * float(row['f1']):.2f}% | "
            f"{float(row['fps']):.2f} |\n"
        )
    output_path.write_text("".join(lines), encoding="utf-8")


def save_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    best_accuracy_row = max(rows, key=lambda row: float(row["accuracy"]))
    best_f1_row = max(rows, key=lambda row: float(row["f1"]))
    fastest_row = max(rows, key=lambda row: float(row["fps"]))
    lines = [
        "Project Report Summary",
        "======================",
        "",
        (
            f"Best accuracy: {best_accuracy_row['experiment']} on {best_accuracy_row['dataset']} "
            f"with {100.0 * float(best_accuracy_row['accuracy']):.2f}%."
        ),
        (
            f"Best F1-score: {best_f1_row['experiment']} on {best_f1_row['dataset']} "
            f"with {100.0 * float(best_f1_row['f1']):.2f}%."
        ),
        (
            f"Fastest inference: {fastest_row['experiment']} on {fastest_row['dataset']} "
            f"with {float(fastest_row['fps']):.2f} FPS."
        ),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_high_res_figure(figure: plt.Figure, output_path: Path, *, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


def plot_comparison_table(rows: list[dict[str, object]], output_path: Path, *, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(12, 0.9 + 0.6 * max(len(rows), 1)))
    axis.axis("off")

    columns = ["Experiment", "Dataset", "Accuracy", "Precision", "Recall", "F1", "FPS"]
    table_data = [
        [
            row["experiment"],
            row["dataset"],
            f"{100.0 * float(row['accuracy']):.2f}%",
            f"{100.0 * float(row['precision']):.2f}%",
            f"{100.0 * float(row['recall']):.2f}%",
            f"{100.0 * float(row['f1']):.2f}%",
            f"{float(row['fps']):.2f}",
        ]
        for row in rows
    ]
    table = axis.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)
    axis.set_title("Experiment Comparison Table", fontsize=14, pad=14)

    for (row_index, column_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#d9e8fb")
            cell.set_text_props(weight="bold")
        elif row_index % 2 == 1:
            cell.set_facecolor("#f7f9fc")

    figure.tight_layout()
    save_high_res_figure(figure, output_path, dpi=dpi)
    plt.close(figure)


def plot_confusion_matrices(rows: list[dict[str, object]], output_path: Path, *, dpi: int) -> None:
    columns = 2
    rows_needed = int(np.ceil(len(rows) / columns))
    figure, axes = plt.subplots(rows_needed, columns, figsize=(12, 5 * rows_needed))
    axes = np.atleast_1d(axes).reshape(rows_needed, columns)

    for axis in axes.flat:
        axis.axis("off")

    for axis, row in zip(axes.flat, rows):
        matrix = np.asarray(row["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{row['experiment']}\n{row['dataset']}", fontsize=12)
        axis.set_xticks([0, 1], labels=["Open", "Closed"])
        axis.set_yticks([0, 1], labels=["Open", "Closed"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        max_value = max(float(matrix.max()), 1.0)
        for i in range(2):
            for j in range(2):
                value = int(matrix[i, j])
                axis.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value >= max_value * 0.5 else "black",
                    fontsize=12,
                    weight="bold",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.axis("on")

    figure.suptitle("Confusion Matrices", fontsize=16)
    figure.tight_layout()
    save_high_res_figure(figure, output_path, dpi=dpi)
    plt.close(figure)


def coerce_numeric(value: str) -> object:
    try:
        if value.strip() == "":
            return value
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except Exception:
        return value


def load_history_file(path: str | Path) -> list[dict[str, object]]:
    history_path = Path(path).expanduser().resolve()
    if history_path.suffix.lower() == ".json":
        return json.loads(history_path.read_text(encoding="utf-8"))
    if history_path.suffix.lower() == ".csv":
        with history_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [{key: coerce_numeric(value) for key, value in row.items()} for row in reader]
    raise ValueError(f"Unsupported history file format: {history_path}")


def get_history_value(entry: dict[str, object], key_path: list[str]) -> float | None:
    current: object = entry
    for key in key_path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            flat_keys = [
                ".".join(key_path),
                "_".join(key_path),
                f"{key_path[0]}_{key_path[1]}" if len(key_path) == 2 else None,
            ]
            for candidate in flat_keys:
                if candidate and isinstance(entry, dict) and candidate in entry:
                    current = entry[candidate]
                    break
            else:
                return None
            break
    if isinstance(current, (int, float)):
        return float(current)
    return None


def plot_training_curves(
    history_specs: list[tuple[str, list[dict[str, object]]]],
    output_path: Path,
    *,
    dpi: int,
) -> bool:
    if not history_specs:
        return False

    figure, axes = plt.subplots(len(history_specs), 2, figsize=(14, 4.5 * len(history_specs)))
    axes = np.atleast_2d(axes)

    for row_axes, (label, history_entries) in zip(axes, history_specs):
        epochs = [get_history_value(entry, ["epoch"]) for entry in history_entries]
        epochs = [int(epoch) for epoch in epochs if epoch is not None]
        if not epochs:
            continue

        loss_axis, score_axis = row_axes

        train_loss = [
            get_history_value(entry, ["train", "loss"])
            or get_history_value(entry, ["train", "total_loss"])
            for entry in history_entries
        ]
        val_loss = [
            get_history_value(entry, ["val", "loss"])
            or get_history_value(entry, ["val", "total_loss"])
            for entry in history_entries
        ]
        train_accuracy = [get_history_value(entry, ["train", "accuracy"]) for entry in history_entries]
        val_accuracy = [get_history_value(entry, ["val", "accuracy"]) for entry in history_entries]
        train_f1 = [get_history_value(entry, ["train", "f1"]) for entry in history_entries]
        val_f1 = [get_history_value(entry, ["val", "f1"]) for entry in history_entries]

        loss_has_series = False
        if any(value is not None for value in train_loss):
            loss_axis.plot(epochs, train_loss, marker="o", label="Train loss")
            loss_has_series = True
        if any(value is not None for value in val_loss):
            loss_axis.plot(epochs, val_loss, marker="o", label="Val loss")
            loss_has_series = True
        if any(get_history_value(entry, ["train", "detection_loss"]) is not None for entry in history_entries):
            loss_axis.plot(
                epochs,
                [get_history_value(entry, ["train", "detection_loss"]) for entry in history_entries],
                linestyle="--",
                label="Train detection loss",
            )
            loss_has_series = True
        if any(get_history_value(entry, ["train", "enhancement_loss"]) is not None for entry in history_entries):
            loss_axis.plot(
                epochs,
                [get_history_value(entry, ["train", "enhancement_loss"]) for entry in history_entries],
                linestyle=":",
                label="Train enhancement loss",
            )
            loss_has_series = True
        loss_axis.set_title(f"{label}: Loss Curves")
        loss_axis.set_xlabel("Epoch")
        loss_axis.set_ylabel("Loss")
        loss_axis.grid(alpha=0.25)
        if loss_has_series:
            loss_axis.legend()
        else:
            loss_axis.text(
                0.5,
                0.5,
                "No loss history available",
                ha="center",
                va="center",
                transform=loss_axis.transAxes,
            )

        score_has_series = False
        if any(value is not None for value in train_accuracy):
            score_axis.plot(epochs, train_accuracy, marker="o", label="Train accuracy")
            score_has_series = True
        if any(value is not None for value in val_accuracy):
            score_axis.plot(epochs, val_accuracy, marker="o", label="Val accuracy")
            score_has_series = True
        if any(value is not None for value in train_f1):
            score_axis.plot(epochs, train_f1, linestyle="--", label="Train F1")
            score_has_series = True
        if any(value is not None for value in val_f1):
            score_axis.plot(epochs, val_f1, linestyle="--", label="Val F1")
            score_has_series = True
        score_axis.set_title(f"{label}: Accuracy / F1")
        score_axis.set_xlabel("Epoch")
        score_axis.set_ylabel("Score")
        score_axis.set_ylim(0.0, 1.05)
        score_axis.grid(alpha=0.25)
        if score_has_series:
            score_axis.legend()
        else:
            score_axis.text(
                0.5,
                0.5,
                "No classification metrics logged\nfor this training run",
                ha="center",
                va="center",
                transform=score_axis.transAxes,
            )

    figure.tight_layout()
    save_high_res_figure(figure, output_path, dpi=dpi)
    plt.close(figure)
    return True


def resolve_default_history_path(checkpoint_path: str | Path, kind: str) -> Path | None:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if kind == "baseline":
        candidate = checkpoint.with_name(f"{checkpoint.stem}_history.json")
        return candidate if candidate.is_file() else None
    if kind == "joint":
        candidate = checkpoint.parent / "joint_training_history.json"
        return candidate if candidate.is_file() else None
    return None


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_checkpoint, baseline_model, baseline_class_to_idx, baseline_image_size = load_baseline_model(
        args.baseline_checkpoint,
        device,
    )
    baseline_positive_label = resolve_positive_label(baseline_class_to_idx)
    baseline_criterion = nn.CrossEntropyLoss()

    clean_root, _, clean_loader = create_loader(
        args.clean_root,
        split=args.split,
        class_to_idx=baseline_class_to_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=build_default_transform(baseline_image_size),
        device=device,
    )
    lowlight_root, _, lowlight_loader = create_loader(
        args.lowlight_root,
        split=args.split,
        class_to_idx=baseline_class_to_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=build_default_transform(baseline_image_size),
        device=device,
    )
    raw_lowlight_root, _, raw_lowlight_loader = create_loader(
        args.lowlight_root,
        split=args.split,
        class_to_idx=baseline_class_to_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=build_raw_transform(baseline_image_size),
        device=device,
    )

    rows: list[dict[str, object]] = []

    clean_metrics = evaluate_classifier(
        baseline_model,
        clean_loader,
        baseline_criterion,
        device,
        baseline_positive_label,
    )
    clean_fps = benchmark_fps(
        clean_loader,
        device,
        lambda images: baseline_model(images),
        warmup_batches=args.fps_warmup_batches,
        max_batches=args.fps_max_batches,
    )
    rows.append(
        row_from_metrics(
            "Baseline CNN",
            "Clean",
            clean_metrics,
            clean_fps,
            positive_label=baseline_positive_label,
        )
    )

    lowlight_metrics = evaluate_classifier(
        baseline_model,
        lowlight_loader,
        baseline_criterion,
        device,
        baseline_positive_label,
    )
    lowlight_fps = benchmark_fps(
        lowlight_loader,
        device,
        lambda images: baseline_model(images),
        warmup_batches=args.fps_warmup_batches,
        max_batches=args.fps_max_batches,
    )
    rows.append(
        row_from_metrics(
            "Baseline CNN",
            "Low-light",
            lowlight_metrics,
            lowlight_fps,
            positive_label=baseline_positive_label,
        )
    )

    if args.enhancer_checkpoint:
        enhancer_model = load_enhancer_model(
            checkpoint_path=args.enhancer_checkpoint,
            device=device,
            hidden_channels=32,
            num_iterations=8,
        )
        enhancer_metrics = evaluate_enhancer_detector(
            enhancer_model,
            baseline_model,
            raw_lowlight_loader,
            baseline_criterion,
            device,
            baseline_positive_label,
        )
        enhancer_fps = benchmark_fps(
            raw_lowlight_loader,
            device,
            lambda images: baseline_model(normalize_for_imagenet(enhancer_model(images)[0])),
            warmup_batches=args.fps_warmup_batches,
            max_batches=args.fps_max_batches,
        )
        rows.append(
            row_from_metrics(
                "Enhancer Only",
                "Low-light",
                enhancer_metrics,
                enhancer_fps,
                positive_label=baseline_positive_label,
            )
        )
    else:
        print("[WARN] No enhancer checkpoint provided. Skipping enhancer-only row.")

    if args.joint_checkpoint:
        joint_checkpoint, joint_model, joint_class_to_idx, joint_image_size = load_joint_model(
            args.joint_checkpoint,
            device,
        )
        joint_positive_label = resolve_positive_label(joint_class_to_idx)
        joint_root, _, joint_loader = create_loader(
            args.lowlight_root,
            split=args.split,
            class_to_idx=joint_class_to_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            transform=build_raw_transform(joint_image_size),
            device=device,
        )
        joint_metrics = evaluate_joint_model(
            joint_model,
            joint_loader,
            device,
            joint_positive_label,
        )
        joint_fps = benchmark_fps(
            joint_loader,
            device,
            lambda images: joint_model(images, return_curve_maps=False)[0],
            warmup_batches=args.fps_warmup_batches,
            max_batches=args.fps_max_batches,
        )
        rows.append(
            row_from_metrics(
                "Joint Enhancer-Detector",
                "Low-light",
                joint_metrics,
                joint_fps,
                positive_label=joint_positive_label,
            )
        )
    else:
        print("[WARN] No joint checkpoint provided. Skipping joint-model row.")

    if not rows:
        raise ValueError("No experiment rows were generated.")

    comparison_csv = output_dir / "experiment_comparison.csv"
    comparison_md = output_dir / "experiment_comparison.md"
    comparison_table_png = output_dir / "comparison_table.png"
    confusion_png = output_dir / "confusion_matrices.png"
    training_curves_png = output_dir / "training_curves.png"
    summary_txt = output_dir / "summary.txt"

    save_comparison_csv(rows, comparison_csv)
    save_markdown_table(rows, comparison_md)
    plot_comparison_table(rows, comparison_table_png, dpi=args.dpi)
    plot_confusion_matrices(rows, confusion_png, dpi=args.dpi)
    save_summary(rows, summary_txt)

    history_specs: list[tuple[str, list[dict[str, object]]]] = []

    baseline_history_path = (
        Path(args.baseline_history).expanduser().resolve()
        if args.baseline_history
        else resolve_default_history_path(args.baseline_checkpoint, "baseline")
    )
    if baseline_history_path and baseline_history_path.is_file():
        history_specs.append(("Baseline CNN", load_history_file(baseline_history_path)))

    if args.enhancer_history:
        enhancer_history_path = Path(args.enhancer_history).expanduser().resolve()
        if enhancer_history_path.is_file():
            history_specs.append(("Enhancer", load_history_file(enhancer_history_path)))

    joint_history_path = (
        Path(args.joint_history).expanduser().resolve()
        if args.joint_history
        else resolve_default_history_path(args.joint_checkpoint, "joint") if args.joint_checkpoint else None
    )
    if joint_history_path and joint_history_path.is_file():
        history_specs.append(("Joint Model", load_history_file(joint_history_path)))

    if plot_training_curves(history_specs, training_curves_png, dpi=args.dpi):
        print(f"Saved training curves to: {training_curves_png}")
    else:
        print("[WARN] No training history files were found. Skipping training-curves figure.")

    print(f"Saved comparison CSV to: {comparison_csv}")
    print(f"Saved Markdown table to: {comparison_md}")
    print(f"Saved high-resolution table to: {comparison_table_png}")
    print(f"Saved high-resolution confusion matrices to: {confusion_png}")
    print(f"Saved summary to: {summary_txt}")


if __name__ == "__main__":
    main()
