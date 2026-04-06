#!/usr/bin/env python3
"""Validation helpers for the joint enhancer-detector pipeline."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torchvision import transforms

from configs.train_config import EnhancementLossConfig, JointLossConfig, JointTrainConfig
from dataloader import create_dataloaders
from losses.joint_loss import JointTrainingLoss
from models.joint_model import build_joint_model
from utils.checkpoints import load_checkpoint
from utils.metrics import (
    RunningJointMetrics,
    format_joint_metric_line,
    resolve_positive_label,
)


def build_joint_transform(image_size: int = 224) -> transforms.Compose:
    """Build the image transform used for joint enhancement-detection training.

    Zero-DCE style enhancement expects image values in the [0, 1] range, so we
    intentionally avoid ImageNet normalization here.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested runtime device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def autocast_context(device: torch.device, enabled: bool):
    """Return an AMP autocast context only when CUDA mixed precision is available."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def run_validation_epoch(
    model: torch.nn.Module,
    dataloader,
    criterion: JointTrainingLoss,
    device: torch.device,
    positive_label: int,
    *,
    use_amp: bool = False,
    debug: bool = False,
) -> dict[str, float]:
    """Run one validation epoch for the joint model."""
    metrics = RunningJointMetrics(positive_label=positive_label)
    model.eval()

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

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

            predictions = torch.argmax(logits, dim=1)
            metrics.update(
                total_loss=float(loss_dict["total"].detach().item()),
                detection_loss=float(loss_dict["detection"].detach().item()),
                enhancement_loss=float(loss_dict["enhancement"].detach().item()),
                predictions=predictions,
                targets=labels,
            )

    return metrics.compute()


def build_joint_loss_from_checkpoint(train_config_dict: dict) -> JointTrainingLoss:
    """Reconstruct the joint loss configuration stored in a training checkpoint."""
    joint_loss_dict = dict(train_config_dict.get("joint_loss", {}))
    enhancement_dict = dict(joint_loss_dict.get("enhancement", {}))

    enhancement_config = EnhancementLossConfig(**enhancement_dict)
    joint_loss_config = JointLossConfig(
        enhancement_lambda=float(joint_loss_dict.get("enhancement_lambda", 1.0)),
        enhancement=enhancement_config,
    )
    train_config = JointTrainConfig(
        image_size=int(train_config_dict.get("image_size", 224)),
        num_classes=int(train_config_dict.get("num_classes", 2)),
        detector_backbone=str(train_config_dict.get("detector_backbone", "custom")),
        batch_size=int(train_config_dict.get("batch_size", 32)),
        epochs=int(train_config_dict.get("epochs", 1)),
        learning_rate=float(train_config_dict.get("learning_rate", 1e-4)),
        weight_decay=float(train_config_dict.get("weight_decay", 1e-4)),
        num_workers=int(train_config_dict.get("num_workers", 2)),
        device=str(train_config_dict.get("device", "auto")),
        seed=int(train_config_dict.get("seed", 42)),
        joint_loss=joint_loss_config,
    )
    return JointTrainingLoss(train_config)


def build_model_from_checkpoint(model_config: dict) -> torch.nn.Module:
    """Rebuild the joint model architecture saved in a checkpoint."""
    return build_joint_model(
        image_size=int(model_config.get("image_size", 224)),
        num_classes=int(model_config.get("num_classes", 2)),
        detector_backbone=str(model_config.get("detector_backbone", "custom")),
        return_curve_maps_by_default=bool(
            model_config.get("return_curve_maps_by_default", True)
        ),
        strict_input_size_check=bool(model_config.get("strict_input_size_check", True)),
        debug=bool(model_config.get("debug", False)),
        enhancer_kwargs=dict(model_config.get("enhancer_kwargs", {})),
        detector_kwargs=dict(model_config.get("detector_kwargs", {})),
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for validating a saved joint model."""
    parser = argparse.ArgumentParser(description="Validate a saved joint enhancer-detector model.")
    parser.add_argument("checkpoint", type=str, help="Path to the saved training checkpoint.")
    parser.add_argument(
        "data_root",
        type=str,
        help="Dataset root containing low-light training/validation images.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Validation batch size.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio if unsplit.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Validation device: auto, cpu, cuda, or a specific CUDA device.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable mixed precision even if CUDA is available.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug prints for the first validation batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model_config = checkpoint["model_config"]
    train_config_dict = checkpoint["train_config"]
    device = resolve_device(args.device)

    image_size = int(model_config.get("image_size", 224))
    transform = build_joint_transform(image_size=image_size)
    data_bundle = create_dataloaders(
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        image_size=image_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        train_transform=transform,
        val_transform=transform,
    )
    positive_label = resolve_positive_label(data_bundle.class_to_idx)

    model = build_model_from_checkpoint(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = build_joint_loss_from_checkpoint(train_config_dict)

    use_amp = torch.cuda.is_available() and device.type == "cuda" and not args.disable_amp
    val_metrics = run_validation_epoch(
        model=model,
        dataloader=data_bundle.val_loader,
        criterion=criterion,
        device=device,
        positive_label=positive_label,
        use_amp=use_amp,
        debug=args.debug,
    )

    print(f"Validation device: {device}")
    print(f"Classes: {data_bundle.class_to_idx}")
    print(format_joint_metric_line("Val", val_metrics))


if __name__ == "__main__":
    main()
