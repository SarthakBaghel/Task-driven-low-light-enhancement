#!/usr/bin/env python3
"""Train a lightweight CNN baseline for binary eye-state classification."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from dataloader import create_dataloaders
from metrics import (
    RunningClassificationMetrics,
    format_metric_line,
    resolve_positive_label,
)
from models.baseline_cnn import BaselineCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight CNN for binary eye-state classification.",
    )
    parser.add_argument(
        "data_root",
        type=str,
        help="Dataset root. Supports class folders directly or train/val split folders.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for Adam.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=5,
        help="Step size for the learning rate scheduler.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Learning rate decay factor for the scheduler.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate used before the classifier head.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size after resizing.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio when train/val folders are not predefined.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader worker processes.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, cpu, cuda, or a specific CUDA device like cuda:0.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="checkpoints/baseline_cnn_best.pt",
        help="Path to save the best model checkpoint.",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        help=(
            "Optional JSON file where epoch-wise training and validation metrics "
            "will be saved for report plots."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested training device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    positive_label: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Run one training or validation epoch."""
    is_training = optimizer is not None
    metrics = RunningClassificationMetrics(positive_label=positive_label)

    model.train(mode=is_training)
    context = nullcontext() if is_training else torch.no_grad()

    with context:
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            metrics.update(
                loss=loss.item(),
                predictions=predictions,
                targets=labels,
            )

    return metrics.compute()


def save_checkpoint(
    save_path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    class_to_idx: dict[str, int],
    classes: list[str],
    metrics: dict[str, float],
    image_size: int,
    dropout: float,
) -> None:
    """Save the best-performing model checkpoint."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "class_to_idx": class_to_idx,
        "classes": classes,
        "model_config": {
            "num_classes": 2,
            "dropout_rate": dropout,
            "image_size": image_size,
        },
        "val_metrics": metrics,
    }
    torch.save(checkpoint, save_path)


def serialize_metrics(metrics: dict[str, object]) -> dict[str, object]:
    """Convert metrics into a JSON-friendly structure."""
    serialized: dict[str, object] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            serialized[key] = value.detach().cpu().tolist()
        elif isinstance(value, (int, float)):
            serialized[key] = float(value)
        else:
            serialized[key] = value
    return serialized


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    data_bundle = create_dataloaders(
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    if len(data_bundle.classes) != 2:
        raise ValueError(
            "BaselineCNN expects exactly 2 classes, "
            f"but found {len(data_bundle.classes)}: {data_bundle.classes}"
        )
    positive_label = resolve_positive_label(data_bundle.class_to_idx)

    model = BaselineCNN(num_classes=2, dropout_rate=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
    )

    best_val_accuracy = -1.0
    best_epoch = 0
    save_path = Path(args.save_path).expanduser().resolve()
    history_path = (
        Path(args.history_path).expanduser().resolve()
        if args.history_path
        else save_path.with_name(f"{save_path.stem}_history.json")
    )
    history_entries: list[dict[str, object]] = []

    print(f"Training on device: {device}")
    print(f"Classes: {data_bundle.class_to_idx}")
    print(
        f"Samples: train={len(data_bundle.train_dataset)} | "
        f"val={len(data_bundle.val_dataset)}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=data_bundle.train_loader,
            criterion=criterion,
            device=device,
            positive_label=positive_label,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=data_bundle.val_loader,
            criterion=criterion,
            device=device,
            positive_label=positive_label,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs} | lr: {current_lr:.6f}")
        print(format_metric_line("Train", train_metrics))
        print(format_metric_line("Val", val_metrics))

        history_entries.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train": serialize_metrics(train_metrics),
                "val": serialize_metrics(val_metrics),
            }
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            save_checkpoint(
                save_path=save_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                class_to_idx=data_bundle.class_to_idx,
                classes=data_bundle.classes,
                metrics=val_metrics,
                image_size=args.image_size,
                dropout=args.dropout,
            )
            print(f"Saved best model to: {save_path}")

        scheduler.step()

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(history_entries, indent=2),
        encoding="utf-8",
    )
    print(f"Saved training history to: {history_path}")

    print(
        f"\nBest validation accuracy: {best_val_accuracy:.4f} "
        f"(epoch {best_epoch})"
    )


if __name__ == "__main__":
    main()
