#!/usr/bin/env python3
"""Train the joint enhancer-detector pipeline on low-light eye-state images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from configs.train_config import EnhancementLossConfig, JointLossConfig, JointTrainConfig
from dataloader import create_dataloaders
from losses.joint_loss import JointTrainingLoss
from models.joint_model import build_joint_model
from utils.checkpoints import save_training_checkpoint
from utils.metrics import (
    RunningJointMetrics,
    format_joint_metric_line,
    resolve_positive_label,
)
from validate_joint import autocast_context, build_joint_transform, run_validation_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a joint Zero-DCE enhancer and eye-state detector pipeline.",
    )
    parser.add_argument(
        "data_root",
        type=str,
        help="Dataset root with low-light images. Supports either train/val folders or flat class folders.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for Adam.",
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
        help="Validation split ratio when the dataset is not already split.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader workers.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, cpu, cuda, or a specific CUDA device.",
    )
    parser.add_argument(
        "--detector-backbone",
        choices=("custom", "mobilenetv2"),
        default="custom",
        help="Detector backbone used after the enhancer.",
    )
    parser.add_argument(
        "--mobilenet-trainable-blocks",
        type=int,
        default=3,
        help="Number of final MobileNetV2 feature blocks to fine-tune.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained MobileNetV2 weights.",
    )
    parser.add_argument(
        "--enhancer-hidden-channels",
        type=int,
        default=32,
        help="Hidden channels used by the Zero-DCE enhancer.",
    )
    parser.add_argument(
        "--enhancer-iterations",
        type=int,
        default=8,
        help="Number of Zero-DCE enhancement iterations.",
    )
    parser.add_argument(
        "--enhancement-lambda",
        type=float,
        default=1.0,
        help="Weight applied to the enhancement loss inside the total joint loss.",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
        help="Clip gradient norm to this value. Set to 0 to disable clipping.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable mixed precision even if CUDA is available.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/joint",
        help="Directory where last and best checkpoints will be saved.",
    )
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Also save numbered epoch checkpoints in addition to last/best.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable joint-model debug prints for the first batch of each epoch.",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        help=(
            "Optional JSON file where epoch-wise train/val metrics are saved "
            "for report-ready training curves."
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
    """Resolve the requested runtime device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def create_grad_scaler(enabled: bool):
    """Create a GradScaler in a way that works across PyTorch AMP APIs."""
    amp_namespace = getattr(torch, "amp", None)
    grad_scaler_cls = getattr(amp_namespace, "GradScaler", None) if amp_namespace else None
    if grad_scaler_cls is not None:
        try:
            return grad_scaler_cls("cuda", enabled=enabled)
        except TypeError:
            return grad_scaler_cls(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_training_config(args: argparse.Namespace) -> JointTrainConfig:
    """Create the nested training config dataclasses used by the joint loss."""
    enhancement_config = EnhancementLossConfig()
    joint_loss_config = JointLossConfig(
        enhancement_lambda=args.enhancement_lambda,
        enhancement=enhancement_config,
    )
    return JointTrainConfig(
        image_size=args.image_size,
        num_classes=2,
        detector_backbone=args.detector_backbone,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        joint_loss=joint_loss_config,
    )


def build_joint_model_config(
    args: argparse.Namespace,
    *,
    num_classes: int,
) -> dict[str, object]:
    """Return a serializable configuration used to rebuild the joint model."""
    detector_kwargs = {
        "backbone": args.detector_backbone,
        "num_classes": num_classes,
        "image_size": args.image_size,
        "mobilenet_trainable_blocks": args.mobilenet_trainable_blocks,
        "use_pretrained": not args.no_pretrained,
        "allow_pretrained_fallback": True,
        "print_summary": False,
    }
    enhancer_kwargs = {
        "hidden_channels": args.enhancer_hidden_channels,
        "num_iterations": args.enhancer_iterations,
    }
    return {
        "image_size": args.image_size,
        "num_classes": num_classes,
        "detector_backbone": args.detector_backbone,
        "return_curve_maps_by_default": True,
        "strict_input_size_check": True,
        "debug": args.debug,
        "enhancer_kwargs": enhancer_kwargs,
        "detector_kwargs": detector_kwargs,
    }


def run_training_epoch(
    model: nn.Module,
    dataloader,
    criterion: JointTrainingLoss,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    positive_label: int,
    *,
    use_amp: bool = False,
    gradient_clip_norm: float = 0.0,
    debug: bool = False,
) -> dict[str, float]:
    """Run one full training epoch for the joint model."""
    metrics = RunningJointMetrics(positive_label=positive_label)
    model.train()

    for batch_index, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, use_amp):
            logits, enhanced_image, curve_maps = model(
                images,
                return_curve_maps=True,
                debug=debug and batch_index == 0,
            )
            loss_dict = criterion(
                logits,
                labels,
                enhanced_image,
                curve_maps,
                input_image=images,
            )
            total_loss = loss_dict["total"]

        if scaler.is_enabled():
            scaler.scale(total_loss).backward()
            if gradient_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if gradient_clip_norm > 0.0:
                clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        metrics.update(
            total_loss=float(loss_dict["total"].detach().item()),
            detection_loss=float(loss_dict["detection"].detach().item()),
            enhancement_loss=float(loss_dict["enhancement"].detach().item()),
            predictions=predictions,
            targets=labels,
        )

    return metrics.compute()


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
    use_amp = torch.cuda.is_available() and device.type == "cuda" and not args.disable_amp
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()

    transform = build_joint_transform(image_size=args.image_size)
    data_bundle = create_dataloaders(
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        train_transform=transform,
        val_transform=transform,
    )
    if len(data_bundle.classes) != 2:
        raise ValueError(
            "Joint training expects exactly 2 classes, "
            f"but found {len(data_bundle.classes)}: {data_bundle.classes}"
        )
    positive_label = resolve_positive_label(data_bundle.class_to_idx)

    train_config = build_training_config(args)
    model_config = build_joint_model_config(args, num_classes=len(data_bundle.classes))
    model = build_joint_model(
        image_size=args.image_size,
        num_classes=len(data_bundle.classes),
        detector_backbone=args.detector_backbone,
        debug=args.debug,
        enhancer_kwargs=dict(model_config["enhancer_kwargs"]),
        detector_kwargs=dict(model_config["detector_kwargs"]),
    ).to(device)

    criterion = JointTrainingLoss(train_config)
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = create_grad_scaler(enabled=use_amp)

    best_val_accuracy = -1.0
    best_epoch = 0
    last_checkpoint_path = checkpoint_dir / "joint_last.pt"
    best_checkpoint_path = checkpoint_dir / "joint_best.pt"
    history_path = (
        Path(args.history_path).expanduser().resolve()
        if args.history_path
        else checkpoint_dir / "joint_training_history.json"
    )
    history_entries: list[dict[str, object]] = []

    print(f"Training on device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"Classes: {data_bundle.class_to_idx}")
    print(
        f"Samples: train={len(data_bundle.train_dataset)} | "
        f"val={len(data_bundle.val_dataset)}"
    )
    print(
        "Joint loss: "
        f"total = detection + {train_config.joint_loss.enhancement_lambda:.4f} * enhancement"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_training_epoch(
            model=model,
            dataloader=data_bundle.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            positive_label=positive_label,
            use_amp=use_amp,
            gradient_clip_norm=args.gradient_clip_norm,
            debug=args.debug,
        )
        val_metrics = run_validation_epoch(
            model=model,
            dataloader=data_bundle.val_loader,
            criterion=criterion,
            device=device,
            positive_label=positive_label,
            use_amp=use_amp,
            debug=args.debug,
        )

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(format_joint_metric_line("Train", train_metrics))
        print(format_joint_metric_line("Val", val_metrics))

        history_entries.append(
            {
                "epoch": epoch,
                "train": serialize_metrics(train_metrics),
                "val": serialize_metrics(val_metrics),
            }
        )

        save_training_checkpoint(
            save_path=last_checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            model_config=model_config,
            train_config=train_config,
            class_to_idx=data_bundle.class_to_idx,
            classes=data_bundle.classes,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_val_accuracy=max(best_val_accuracy, val_metrics["accuracy"]),
            extra_state={
                "best_epoch": best_epoch,
                "gradient_clip_norm": args.gradient_clip_norm,
                "amp_enabled": use_amp,
            },
        )

        if args.save_every_epoch:
            save_training_checkpoint(
                save_path=checkpoint_dir / f"joint_epoch_{epoch:03d}.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                model_config=model_config,
                train_config=train_config,
                class_to_idx=data_bundle.class_to_idx,
                classes=data_bundle.classes,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_val_accuracy=max(best_val_accuracy, val_metrics["accuracy"]),
                extra_state={
                    "best_epoch": best_epoch,
                    "gradient_clip_norm": args.gradient_clip_norm,
                    "amp_enabled": use_amp,
                },
            )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = float(val_metrics["accuracy"])
            best_epoch = epoch
            save_training_checkpoint(
                save_path=best_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                model_config=model_config,
                train_config=train_config,
                class_to_idx=data_bundle.class_to_idx,
                classes=data_bundle.classes,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_val_accuracy=best_val_accuracy,
                extra_state={
                    "best_epoch": best_epoch,
                    "gradient_clip_norm": args.gradient_clip_norm,
                    "amp_enabled": use_amp,
                },
            )
            print(
                f"Saved new best checkpoint at {best_checkpoint_path} "
                f"(val_accuracy={best_val_accuracy:.4f})"
            )

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(history_entries, indent=2),
        encoding="utf-8",
    )
    print(f"Saved training history to: {history_path}")

    print(f"\nTraining finished. Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}")
    print(f"Last checkpoint: {last_checkpoint_path}")
    print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
