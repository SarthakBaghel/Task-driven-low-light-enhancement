#!/usr/bin/env python3
"""Train a standalone Zero-DCE enhancer on low-light images.

This script is the missing "enhancer only" training stage for the project:

low-light image -> Zero-DCE enhancer -> enhancement losses

The dataset can still use the same folder structure as the classifiers
(`train/open`, `train/closed`, `val/open`, `val/closed`), but the class labels
are ignored here because Zero-DCE is trained with unsupervised image-quality
losses rather than class supervision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from configs.train_config import EnhancementLossConfig
from dataloader import create_dataloaders, seed_worker
from losses.enhancement_losses import enhancement_loss
from models.zerodce import ZeroDCE
from validate_joint import autocast_context, build_joint_transform


LOSS_KEYS = (
    "loss",
    "exposure",
    "color_constancy",
    "illumination_smoothness",
    "spatial_consistency",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Zero-DCE enhancer on low-light images.",
    )
    parser.add_argument(
        "data_root",
        type=str,
        help=(
            "Dataset root containing low-light images. Supports either train/val "
            "folders or a flat folder with class subdirectories."
        ),
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
        default=0,
        help="Number of dataloader workers. Default is 0 for Colab/sandbox friendliness.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, cpu, cuda, mps, or an explicit device like cuda:0.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=32,
        help="Hidden channel width used by Zero-DCE.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=8,
        help="Number of iterative enhancement curves predicted by Zero-DCE.",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="Factor used by ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=3,
        help="Epoch patience before the scheduler reduces learning rate.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=6,
        help="Stop if validation loss does not improve for this many epochs.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation-loss improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=0.0,
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
        default="checkpoints/enhancer",
        help="Directory where enhancer_last.pt and enhancer_best.pt will be saved.",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        help="Optional JSON path for saving epoch-wise training history.",
    )
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Also save numbered checkpoints for each epoch.",
    )
    parser.add_argument(
        "--exposure-weight",
        type=float,
        default=10.0,
        help="Weight for exposure control loss.",
    )
    parser.add_argument(
        "--color-weight",
        type=float,
        default=5.0,
        help="Weight for color constancy loss.",
    )
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        default=200.0,
        help="Weight for illumination smoothness loss.",
    )
    parser.add_argument(
        "--spatial-weight",
        type=float,
        default=1.0,
        help="Weight for optional spatial consistency loss.",
    )
    parser.add_argument(
        "--exposure-patch-size",
        type=int,
        default=16,
        help="Patch size used by the exposure control loss.",
    )
    parser.add_argument(
        "--exposure-target-mean",
        type=float,
        default=0.6,
        help="Target local mean brightness used by the exposure control loss.",
    )
    parser.add_argument(
        "--smoothness-power",
        type=float,
        default=2.0,
        help="Power applied inside the illumination smoothness regularizer.",
    )
    parser.add_argument(
        "--spatial-pool-size",
        type=int,
        default=4,
        help="Pooling size used by the spatial consistency loss.",
    )
    parser.add_argument(
        "--color-use-sqrt",
        action="store_true",
        help="Use the sqrt variant of the color constancy loss.",
    )
    parser.add_argument(
        "--color-eps",
        type=float,
        default=1e-8,
        help="Numerical stability epsilon used by the optional sqrt color loss.",
    )
    parser.add_argument(
        "--disable-spatial-consistency",
        action="store_true",
        help="Disable spatial consistency loss inside the enhancement objective.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set seeds for reproducible training."""
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
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def create_grad_scaler(enabled: bool):
    """Create a GradScaler that works across PyTorch AMP API variants."""
    amp_namespace = getattr(torch, "amp", None)
    grad_scaler_cls = getattr(amp_namespace, "GradScaler", None) if amp_namespace else None
    if grad_scaler_cls is not None:
        try:
            return grad_scaler_cls("cuda", enabled=enabled)
        except TypeError:
            return grad_scaler_cls(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_loss_config(args: argparse.Namespace) -> EnhancementLossConfig:
    """Create a reusable Zero-DCE loss config from CLI arguments."""
    return EnhancementLossConfig(
        exposure_weight=args.exposure_weight,
        color_weight=args.color_weight,
        smoothness_weight=args.smoothness_weight,
        spatial_weight=args.spatial_weight,
        exposure_patch_size=args.exposure_patch_size,
        exposure_target_mean=args.exposure_target_mean,
        color_use_sqrt=args.color_use_sqrt,
        color_eps=args.color_eps,
        smoothness_power=args.smoothness_power,
        spatial_pool_size=args.spatial_pool_size,
        use_spatial_consistency=not args.disable_spatial_consistency,
    )


def build_train_config(
    args: argparse.Namespace,
    *,
    loss_config: EnhancementLossConfig,
) -> dict[str, Any]:
    """Build a serializable config snapshot for checkpoints."""
    return {
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "val_ratio": args.val_ratio,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": args.device,
        "hidden_channels": args.hidden_channels,
        "num_iterations": args.num_iterations,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "gradient_clip_norm": args.gradient_clip_norm,
        "use_amp": not args.disable_amp,
        "enhancement_loss": loss_config.to_kwargs(),
    }


def detach_metrics(loss_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """Convert the enhancement-loss dictionary into detached Python floats."""
    return {
        "loss": float(loss_dict["total"].detach().item()),
        "exposure": float(loss_dict["exposure"].detach().item()),
        "color_constancy": float(loss_dict["color_constancy"].detach().item()),
        "illumination_smoothness": float(
            loss_dict["illumination_smoothness"].detach().item()
        ),
        "spatial_consistency": float(loss_dict["spatial_consistency"].detach().item()),
    }


def average_metric_totals(totals: dict[str, float], batches: int) -> dict[str, float]:
    """Return mean metrics over an epoch."""
    if batches <= 0:
        raise ValueError("Cannot average epoch metrics with zero batches.")
    return {key: value / batches for key, value in totals.items()}


def format_metric_line(prefix: str, metrics: dict[str, float]) -> str:
    """Format a compact one-line metric summary."""
    return (
        f"{prefix} loss={metrics['loss']:.4f} | "
        f"exp={metrics['exposure']:.4f} | "
        f"color={metrics['color_constancy']:.4f} | "
        f"smooth={metrics['illumination_smoothness']:.4f} | "
        f"spatial={metrics['spatial_consistency']:.4f}"
    )


def write_history(history: list[dict[str, Any]], path: str | Path) -> Path:
    """Persist the epoch-wise training history as JSON."""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return output_path


def build_checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler,
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_val_loss: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a checkpoint payload compatible with the existing inference scripts."""
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "model_config": model_config,
        "train_config": train_config,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_val_loss": best_val_loss,
        "history": history,
    }


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a checkpoint to disk."""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def run_epoch(
    model: ZeroDCE,
    dataloader,
    device: torch.device,
    loss_config: EnhancementLossConfig,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    use_amp: bool = False,
    gradient_clip_norm: float = 0.0,
    epoch_index: int = 1,
    total_epochs: int = 1,
    stage: str = "train",
) -> dict[str, float]:
    """Run one train or validation epoch for the standalone enhancer."""
    is_training = optimizer is not None
    model.train(mode=is_training)
    totals = {key: 0.0 for key in LOSS_KEYS}
    batch_count = 0
    progress = tqdm(
        dataloader,
        desc=f"{stage.capitalize()} {epoch_index:02d}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )

    for images, _ in progress:
        batch_count += 1
        images = images.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, use_amp):
            enhanced_image, curve_maps = model(images)
            loss_dict = enhancement_loss(
                enhanced_image,
                curve_maps,
                input_image=images,
                **loss_config.to_kwargs(),
            )

        loss_value = loss_dict["total"]
        if is_training:
            if scaler is not None and use_amp:
                scaler.scale(loss_value).backward()
                if gradient_clip_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_value.backward()
                if gradient_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

        batch_metrics = detach_metrics(loss_dict)
        for key, value in batch_metrics.items():
            totals[key] += value

        averaged_so_far = average_metric_totals(totals, batch_count)
        progress.set_postfix(
            loss=f"{averaged_so_far['loss']:.4f}",
            exp=f"{averaged_so_far['exposure']:.4f}",
            smooth=f"{averaged_so_far['illumination_smoothness']:.4f}",
        )

    return average_metric_totals(totals, batch_count)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    use_amp = device.type == "cuda" and not args.disable_amp
    scaler = create_grad_scaler(use_amp)

    transform = build_joint_transform(image_size=args.image_size)
    data_bundle = create_dataloaders(
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        train_transform=transform,
        val_transform=transform,
    )

    model = ZeroDCE(
        hidden_channels=args.hidden_channels,
        num_iterations=args.num_iterations,
    ).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    loss_config = build_loss_config(args)
    train_config = build_train_config(args, loss_config=loss_config)
    model_config = {
        "hidden_channels": args.hidden_channels,
        "num_iterations": args.num_iterations,
    }

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = (
        Path(args.history_path).expanduser().resolve()
        if args.history_path
        else checkpoint_dir / "enhancer_training_history.json"
    )
    best_checkpoint_path = checkpoint_dir / "enhancer_best.pt"
    last_checkpoint_path = checkpoint_dir / "enhancer_last.pt"

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    print(f"Training device: {device}")
    print(f"Loaded classes (ignored for enhancer losses): {data_bundle.class_to_idx}")
    print(f"Train images: {len(data_bundle.train_dataset)}")
    print(f"Val images: {len(data_bundle.val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_metrics = run_epoch(
            model=model,
            dataloader=data_bundle.train_loader,
            device=device,
            loss_config=loss_config,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            gradient_clip_norm=args.gradient_clip_norm,
            epoch_index=epoch,
            total_epochs=args.epochs,
            stage="train",
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=data_bundle.val_loader,
            device=device,
            loss_config=loss_config,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
            gradient_clip_norm=0.0,
            epoch_index=epoch,
            total_epochs=args.epochs,
            stage="val",
        )
        scheduler.step(val_metrics["loss"])

        epoch_seconds = time.perf_counter() - epoch_start
        history_entry = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(history_entry)
        write_history(history, history_path)

        improved = val_metrics["loss"] < (best_val_loss - args.min_delta)
        if improved:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0

            best_payload = build_checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                model_config=model_config,
                train_config=train_config,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_val_loss=best_val_loss,
                history=history,
            )
            save_checkpoint(best_payload, best_checkpoint_path)
        else:
            epochs_without_improvement += 1

        payload = build_checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_config=model_config,
            train_config=train_config,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_val_loss=best_val_loss,
            history=history,
        )
        save_checkpoint(payload, last_checkpoint_path)

        if args.save_every_epoch:
            epoch_path = checkpoint_dir / f"enhancer_epoch_{epoch:03d}.pt"
            save_checkpoint(payload, epoch_path)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | lr={current_lr:.6f} | "
            f"time={epoch_seconds:.1f}s"
        )
        print(format_metric_line("Train", train_metrics))
        print(format_metric_line("Val  ", val_metrics))
        if improved:
            print(
                f"Best checkpoint updated: epoch={epoch} "
                f"val_loss={best_val_loss:.4f}"
            )
        else:
            print(
                f"No validation-loss improvement for {epochs_without_improvement} "
                f"epoch(s). Best is epoch {best_epoch} with loss {best_val_loss:.4f}."
            )

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                "Early stopping triggered because validation loss stopped improving."
            )
            break

    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Last checkpoint: {last_checkpoint_path}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
