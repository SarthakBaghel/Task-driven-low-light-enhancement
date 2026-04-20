#!/usr/bin/env python3
"""Train a stronger eye-state classifier with transfer learning and Colab support."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataloader import create_dataloaders, seed_worker
from dataset import EyeStateDataset
from inference_enhancer import load_model as load_enhancer_model
from losses.focal_loss import FocalLoss
from models.detector import (
    build_detector,
    count_stored_gradients,
    freeze_model,
    gradients_enabled_for_model,
    print_gradient_debug_info,
)
from utils.classifier_metrics import (
    DEFAULT_MAX_CLOSED_PREDICTION_RATE,
    DEFAULT_MIN_CLOSED_PREDICTION_RATE,
    ProbabilityAccumulator,
    format_classifier_metric_line,
    resolve_positive_label,
)
from utils.classifier_transforms import (
    build_transfer_learning_raw_transforms,
    build_transfer_learning_transforms,
    normalize_tensor_for_detector,
)
from utils.colab_runtime import (
    copy_checkpoints_to_directory,
    mount_google_drive,
    resolve_runtime,
    resolve_runtime_path,
    resolve_workspace_root,
)


@dataclass
class EarlyStoppingState:
    best_score: float = float("-inf")
    best_epoch: int = 0
    patience_counter: int = 0


DEFAULT_TRANSFER_BEST_PATH = "checkpoints/transfer_detector_best.pt"
DEFAULT_TRANSFER_LAST_PATH = "checkpoints/transfer_detector_last.pt"
DEFAULT_ENHANCED_BEST_PATH = "checkpoints/detector_enhanced.pth"
DEFAULT_ENHANCED_LAST_PATH = "checkpoints/detector_enhanced_last.pth"
DEFAULT_DUAL_BEST_PATH = "checkpoints/detector_dual_input.pth"
DEFAULT_DUAL_LAST_PATH = "checkpoints/detector_dual_input_last.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a transfer-learning eye-state classifier.",
    )
    parser.add_argument("data_root", type=str, help="Dataset root with train/val or flat class folders.")
    parser.add_argument(
        "--lowlight-val-root",
        type=str,
        help="Optional low-light validation root evaluated after each epoch.",
    )
    parser.add_argument(
        "--backbone",
        choices=("mobilenetv2", "resnet18"),
        default="resnet18",
        help="Transfer-learning backbone.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-batch-size", type=int, default=4)
    parser.add_argument(
        "--disable-auto-batch-size",
        action="store_true",
        help="Disable automatic batch-size reduction when memory is limited.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable more deterministic training behavior for reproducibility.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--threshold-objective",
        choices=("f1", "recall"),
        default="f1",
        help="Metric optimized when tuning the closed-eye threshold on validation data.",
    )
    parser.add_argument(
        "--monitor",
        choices=("accuracy", "f1", "closed_recall"),
        default="f1",
        help="Validation metric used for early stopping and best-checkpoint selection.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--mobilenet-trainable-blocks", type=int, default=3)
    parser.add_argument("--resnet-trainable-layers", type=int, default=1)
    parser.add_argument("--save-path", type=str, default=DEFAULT_TRANSFER_BEST_PATH)
    parser.add_argument("--save-last-path", type=str, default=DEFAULT_TRANSFER_LAST_PATH)
    parser.add_argument(
        "--drive-checkpoint-dir",
        type=str,
        help="Optional secondary checkpoint directory, useful for Google Drive backups.",
    )
    parser.add_argument("--report-lowlight-every-epoch", action="store_true")
    parser.add_argument("--debug-summary", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--enhancer-checkpoint",
        type=str,
        help=(
            "Optional pretrained enhancer checkpoint. When provided, each image is "
            "first enhanced and only the detector is trained."
        ),
    )
    parser.add_argument(
        "--detector-input-mode",
        choices=("auto", "raw", "enhanced", "dual"),
        default="auto",
        help=(
            "Which detector inputs to use. 'raw' trains on the low-light image only, "
            "'enhanced' trains on enhancer outputs only, and 'dual' trains on both "
            "raw and enhanced images together."
        ),
    )
    parser.add_argument(
        "--dual-input-separate-backbones",
        action="store_true",
        help="Use separate backbone copies instead of shared weights for dual-input training.",
    )
    parser.add_argument(
        "--enhancer-hidden-channels",
        type=int,
        default=32,
        help="Fallback hidden channel width when the enhancer checkpoint has no config.",
    )
    parser.add_argument(
        "--enhancer-iterations",
        type=int,
        default=8,
        help="Fallback Zero-DCE iteration count when the checkpoint has no config.",
    )
    parser.add_argument(
        "--init-detector-checkpoint",
        type=str,
        help="Optional clean-trained detector checkpoint used as initialization.",
    )
    parser.add_argument(
        "--threshold-candidates",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6],
        help="Candidate thresholds used for validation-time threshold tuning.",
    )
    parser.add_argument(
        "--min-closed-prediction-rate",
        type=float,
        default=DEFAULT_MIN_CLOSED_PREDICTION_RATE,
        help=(
            "When tuning thresholds, reject candidates that predict fewer than this "
            "fraction of samples as the closed-eye class unless no balanced option exists."
        ),
    )
    parser.add_argument(
        "--max-closed-prediction-rate",
        type=float,
        default=DEFAULT_MAX_CLOSED_PREDICTION_RATE,
        help=(
            "When tuning thresholds, reject candidates that predict more than this "
            "fraction of samples as the closed-eye class unless no balanced option exists."
        ),
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--runtime",
        choices=("auto", "local", "colab"),
        default="auto",
        help="Select local Mac paths or Colab paths automatically.",
    )
    parser.add_argument(
        "--local-workspace-root",
        type=str,
        default=".",
        help="Workspace root used when running locally.",
    )
    parser.add_argument(
        "--colab-workspace-root",
        type=str,
        default="/content/drive/MyDrive/task-driven-low-light-enhancement",
        help="Workspace root used when running in Colab.",
    )
    parser.add_argument(
        "--mount-drive",
        action="store_true",
        help="Mount Google Drive from inside the training script when running in Colab.",
    )
    parser.add_argument(
        "--drive-mount-point",
        type=str,
        default="/content/drive",
        help="Mount point used for Google Drive in Colab.",
    )
    parser.add_argument(
        "--force-remount-drive",
        action="store_true",
        help="Force-remount Google Drive when --mount-drive is enabled.",
    )
    return parser.parse_args()


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def resolve_device(device_name: str) -> torch.device:
    """Resolve the best available training device."""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def compute_focal_alpha(targets: list[int], num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(targets, dtype=torch.int64), minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    inverse = counts.sum() / counts
    return inverse / inverse.sum() * num_classes


def recommend_batch_size(
    requested_batch_size: int,
    *,
    device: torch.device,
    auto_batch_size: bool,
) -> int:
    """Choose a safer starting batch size for smaller Colab GPUs."""
    if not auto_batch_size or device.type != "cuda":
        return requested_batch_size

    total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    suggested_batch_size = requested_batch_size
    if total_memory_gb <= 6:
        suggested_batch_size = min(suggested_batch_size, 8)
    elif total_memory_gb <= 10:
        suggested_batch_size = min(suggested_batch_size, 16)
    elif total_memory_gb <= 16:
        suggested_batch_size = min(suggested_batch_size, 24)

    return max(1, suggested_batch_size)


def is_out_of_memory_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def reduce_batch_size(current_batch_size: int, min_batch_size: int) -> int:
    reduced = max(min_batch_size, current_batch_size // 2)
    return max(1, reduced)


def create_lowlight_loader(
    dataset_root: str | Path,
    *,
    batch_size: int,
    image_size: int,
    class_to_idx: dict[str, int],
    num_workers: int,
    seed: int,
    device: torch.device,
    transform,
) -> tuple[EyeStateDataset, DataLoader]:
    dataset = EyeStateDataset(
        root=dataset_root,
        class_to_idx=class_to_idx,
        transform=transform,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return dataset, loader


def resolve_detector_input_mode(args: argparse.Namespace) -> str:
    """Resolve automatic detector input mode selection and validate enhancer usage."""
    requested_mode = str(args.detector_input_mode).lower()
    if requested_mode == "auto":
        return "enhanced" if args.enhancer_checkpoint else "raw"
    if requested_mode in {"enhanced", "dual"} and not args.enhancer_checkpoint:
        raise ValueError(
            f"--detector-input-mode {requested_mode} requires --enhancer-checkpoint."
        )
    return requested_mode


def build_training_data(
    *,
    dataset_root: Path,
    lowlight_val_root: Path | None,
    batch_size: int,
    image_size: int,
    val_ratio: float,
    seed: int,
    num_workers: int,
    device: torch.device,
    detector_input_mode: str,
) -> tuple[object, EyeStateDataset | None, DataLoader | None]:
    if detector_input_mode in {"enhanced", "dual"}:
        train_transform, val_transform = build_transfer_learning_raw_transforms(
            image_size=image_size
        )
    else:
        train_transform, val_transform = build_transfer_learning_transforms(
            image_size=image_size
        )
    data_bundle = create_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    lowlight_dataset = None
    lowlight_loader = None
    if lowlight_val_root is not None:
        lowlight_dataset, lowlight_loader = create_lowlight_loader(
            lowlight_val_root,
            batch_size=batch_size,
            image_size=image_size,
            class_to_idx=data_bundle.class_to_idx,
            num_workers=num_workers,
            seed=seed,
            device=device,
            transform=val_transform,
        )

    return data_bundle, lowlight_dataset, lowlight_loader


def resolve_save_paths(
    args: argparse.Namespace,
    *,
    workspace_root: Path,
    detector_input_mode: str,
) -> tuple[Path, Path]:
    """Resolve output checkpoint paths, using enhanced defaults when needed."""
    save_path_arg = args.save_path
    save_last_path_arg = args.save_last_path

    if detector_input_mode == "dual":
        if save_path_arg == DEFAULT_TRANSFER_BEST_PATH:
            save_path_arg = DEFAULT_DUAL_BEST_PATH
        if save_last_path_arg == DEFAULT_TRANSFER_LAST_PATH:
            save_last_path_arg = DEFAULT_DUAL_LAST_PATH
    elif detector_input_mode == "enhanced":
        if save_path_arg == DEFAULT_TRANSFER_BEST_PATH:
            save_path_arg = DEFAULT_ENHANCED_BEST_PATH
        if save_last_path_arg == DEFAULT_TRANSFER_LAST_PATH:
            save_last_path_arg = DEFAULT_ENHANCED_LAST_PATH

    best_save_path = resolve_runtime_path(save_path_arg, workspace_root=workspace_root)
    last_save_path = resolve_runtime_path(save_last_path_arg, workspace_root=workspace_root)
    return best_save_path, last_save_path


def load_detector_initialization_checkpoint(path: str | Path) -> dict[str, object]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Detector initialization checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def initialize_detector_from_checkpoint(
    model: nn.Module,
    checkpoint: dict[str, object],
) -> None:
    """Load a clean-trained detector checkpoint as detector initialization."""
    checkpoint_config = checkpoint.get("model_config", {})
    checkpoint_backbone = checkpoint_config.get("backbone")
    target_backbone = getattr(model, "backbone", None)
    if checkpoint_backbone is not None and target_backbone is not None:
        if str(checkpoint_backbone) != str(target_backbone):
            raise ValueError(
                "Initialization checkpoint backbone does not match the requested detector "
                f"backbone: checkpoint={checkpoint_backbone}, requested={target_backbone}."
            )
    checkpoint_uses_dual_input = bool(checkpoint_config.get("use_dual_input", False))
    target_uses_dual_input = bool(getattr(model, "use_dual_input", False))
    if checkpoint_uses_dual_input == target_uses_dual_input:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return

    if target_uses_dual_input and not checkpoint_uses_dual_input:
        initialize_dual_detector_from_single_input_checkpoint(
            model,
            checkpoint["model_state_dict"],
        )
        return

    raise ValueError(
        "Initialization checkpoint input mode is incompatible with the requested detector: "
        f"checkpoint use_dual_input={checkpoint_uses_dual_input}, "
        f"requested use_dual_input={target_uses_dual_input}."
    )


def _extract_feature_state_from_single_input_checkpoint(
    checkpoint_state_dict: dict[str, torch.Tensor],
    target_backbone: nn.Module,
) -> dict[str, torch.Tensor]:
    """Return checkpoint weights that match a dual-input feature backbone."""
    target_keys = set(target_backbone.state_dict().keys())
    extracted: dict[str, torch.Tensor] = {}
    for key, value in checkpoint_state_dict.items():
        if not key.startswith("model."):
            continue
        trimmed_key = key[len("model.") :]
        if trimmed_key in target_keys:
            extracted[trimmed_key] = value
    if not extracted:
        raise ValueError(
            "Could not extract any feature-backbone weights from the initialization checkpoint."
        )
    return extracted


def initialize_dual_detector_from_single_input_checkpoint(
    model: nn.Module,
    checkpoint_state_dict: dict[str, torch.Tensor],
) -> None:
    """Initialize dual-input branches from a compatible single-input detector checkpoint."""
    if not getattr(model, "use_dual_input", False):
        raise ValueError("Dual-input initialization was requested for a single-input detector.")

    if getattr(model, "dual_input_shared_backbone", False):
        shared_backbone = getattr(model, "shared_backbone")
        feature_state = _extract_feature_state_from_single_input_checkpoint(
            checkpoint_state_dict,
            shared_backbone,
        )
        shared_backbone.load_state_dict(feature_state, strict=False)
    else:
        raw_backbone = getattr(model, "raw_backbone")
        enhanced_backbone = getattr(model, "enhanced_backbone")
        raw_state = _extract_feature_state_from_single_input_checkpoint(
            checkpoint_state_dict,
            raw_backbone,
        )
        enhanced_state = _extract_feature_state_from_single_input_checkpoint(
            checkpoint_state_dict,
            enhanced_backbone,
        )
        raw_backbone.load_state_dict(raw_state, strict=False)
        enhanced_backbone.load_state_dict(enhanced_state, strict=False)


def prepare_detector_inputs(
    images: torch.Tensor,
    *,
    enhancer: nn.Module | None,
    detector_input_mode: str,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return detector inputs, optionally passing through a frozen enhancer."""
    if detector_input_mode == "raw":
        return images
    if enhancer is None:
        raise ValueError(
            f"Detector input mode '{detector_input_mode}' requires a frozen enhancer."
        )
    with torch.no_grad():
        enhanced_images, _ = enhancer(images)
    if detector_input_mode == "enhanced":
        return normalize_tensor_for_detector(enhanced_images)
    if detector_input_mode == "dual":
        return (
            normalize_tensor_for_detector(images),
            normalize_tensor_for_detector(enhanced_images),
        )
    raise ValueError(f"Unsupported detector input mode: {detector_input_mode}")


def verify_module_stays_frozen(module: nn.Module, *, module_name: str, context: str) -> None:
    """Fail loudly if a supposedly frozen module starts storing gradients."""
    trainable = gradients_enabled_for_model(module)
    stored_gradients = count_stored_gradients(module)
    print(
        f"[FreezeDebug] {module_name}-{context}: "
        f"requires_grad_enabled={trainable} | stored_gradients={stored_gradients}"
    )
    if trainable:
        raise RuntimeError(
            f"{module_name} is not frozen during {context}: found parameters with requires_grad=True."
        )
    if stored_gradients != 0:
        raise RuntimeError(
            f"{module_name} stored gradients during {context}: found {stored_gradients} tensors."
        )


def run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    positive_label: int,
    *,
    enhancer: nn.Module | None = None,
    detector_input_mode: str = "raw",
    optimizer: torch.optim.Optimizer | None = None,
    threshold: float | None = None,
    tune_threshold: bool = False,
    threshold_objective: str = "f1",
    threshold_candidates: Iterable[float] | None = None,
    min_positive_rate: float = DEFAULT_MIN_CLOSED_PREDICTION_RATE,
    max_positive_rate: float = DEFAULT_MAX_CLOSED_PREDICTION_RATE,
    description: str = "",
    show_progress: bool = True,
) -> dict[str, float]:
    is_training = optimizer is not None
    accumulator = ProbabilityAccumulator()
    model.train(mode=is_training)

    progress_bar = tqdm(
        dataloader,
        desc=description,
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )

    context = torch.enable_grad() if is_training else torch.no_grad()
    try:
        with context:
            for images, labels in progress_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                detector_inputs = prepare_detector_inputs(
                    images,
                    enhancer=enhancer,
                    detector_input_mode=detector_input_mode,
                )

                if is_training:
                    optimizer.zero_grad(set_to_none=True)

                logits = model(detector_inputs)
                loss = criterion(logits, labels)

                if is_training:
                    loss.backward()
                    optimizer.step()

                accumulator.update(
                    loss=float(loss.detach().item()),
                    logits=logits,
                    targets=labels,
                    positive_label=positive_label,
                )

                if show_progress:
                    average_loss = accumulator.total_loss / max(accumulator.total_samples, 1)
                    progress_bar.set_postfix(loss=f"{average_loss:.4f}")
    finally:
        progress_bar.close()

    return accumulator.compute(
        positive_label=positive_label,
        threshold=threshold,
        tune_threshold=tune_threshold,
        threshold_objective=threshold_objective,
        threshold_candidates=threshold_candidates,
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
    )


def metric_value(metrics: dict[str, float], monitor: str) -> float:
    return float(metrics[monitor])


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    class_to_idx: dict[str, int],
    classes: list[str],
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    lowlight_metrics: dict[str, float] | None,
    model_config: dict[str, object],
    training_config: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_to_idx": class_to_idx,
            "classes": classes,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "lowlight_metrics": lowlight_metrics,
            "model_config": model_config,
            "training_config": training_config,
            "best_threshold": val_metrics["threshold"],
        },
        path,
    )


def backup_checkpoints_if_requested(
    *,
    checkpoint_paths: list[Path],
    drive_checkpoint_dir: Path | None,
) -> None:
    if drive_checkpoint_dir is None:
        return
    copied = copy_checkpoints_to_directory(checkpoint_paths, destination_dir=drive_checkpoint_dir)
    if copied:
        copied_string = ", ".join(str(path) for path in copied)
        print(f"Backed up checkpoint(s) to: {copied_string}")


def main() -> None:
    args = parse_args()
    runtime = resolve_runtime(args.runtime)
    detector_input_mode = resolve_detector_input_mode(args)

    if args.mount_drive:
        mounted_path = mount_google_drive(
            mount_point=args.drive_mount_point,
            force_remount=args.force_remount_drive,
        )
        if mounted_path is not None:
            print(f"Google Drive mounted at: {mounted_path}")

    workspace_root = resolve_workspace_root(
        runtime=runtime,
        local_workspace_root=args.local_workspace_root,
        colab_workspace_root=args.colab_workspace_root,
    )
    data_root = resolve_runtime_path(args.data_root, workspace_root=workspace_root)
    lowlight_val_root = (
        resolve_runtime_path(args.lowlight_val_root, workspace_root=workspace_root)
        if args.lowlight_val_root
        else None
    )
    best_save_path, last_save_path = resolve_save_paths(
        args,
        workspace_root=workspace_root,
        detector_input_mode=detector_input_mode,
    )
    drive_checkpoint_dir = (
        resolve_runtime_path(args.drive_checkpoint_dir, workspace_root=workspace_root)
        if args.drive_checkpoint_dir
        else None
    )

    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    effective_batch_size = recommend_batch_size(
        args.batch_size,
        device=device,
        auto_batch_size=not args.disable_auto_batch_size,
    )
    if effective_batch_size != args.batch_size:
        print(
            f"Adjusted initial batch size from {args.batch_size} to {effective_batch_size} "
            f"based on available GPU memory."
        )

    data_bundle, lowlight_dataset, lowlight_loader = build_training_data(
        dataset_root=data_root,
        lowlight_val_root=lowlight_val_root,
        batch_size=effective_batch_size,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        device=device,
        detector_input_mode=detector_input_mode,
    )

    if len(data_bundle.classes) != 2:
        raise ValueError(
            "This training pipeline expects exactly 2 classes, "
            f"but found {len(data_bundle.classes)}: {data_bundle.classes}"
        )
    positive_label = resolve_positive_label(data_bundle.class_to_idx)

    alpha = compute_focal_alpha(data_bundle.train_dataset.targets, len(data_bundle.classes))
    criterion = FocalLoss(
        alpha=alpha,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    model = build_detector(
        backbone=args.backbone,
        num_classes=len(data_bundle.classes),
        image_size=args.image_size,
        mobilenet_trainable_blocks=args.mobilenet_trainable_blocks,
        resnet_trainable_layers=args.resnet_trainable_layers,
        use_pretrained=not args.no_pretrained,
        allow_pretrained_fallback=True,
        use_dual_input=detector_input_mode == "dual",
        dual_input_shared_backbone=not args.dual_input_separate_backbones,
        print_summary=args.debug_summary,
    ).to(device)
    init_detector_checkpoint = None
    if args.init_detector_checkpoint:
        init_detector_checkpoint = load_detector_initialization_checkpoint(
            resolve_runtime_path(args.init_detector_checkpoint, workspace_root=workspace_root)
        )
        initialize_detector_from_checkpoint(model, init_detector_checkpoint)

    enhancer = None
    if detector_input_mode in {"enhanced", "dual"}:
        enhancer = load_enhancer_model(
            checkpoint_path=resolve_runtime_path(args.enhancer_checkpoint, workspace_root=workspace_root),
            device=device,
            hidden_channels=args.enhancer_hidden_channels,
            num_iterations=args.enhancer_iterations,
        )
        freeze_model(
            enhancer,
            set_eval=True,
            module_name="enhancer",
            print_frozen_layers=False,
        )
        print_gradient_debug_info(enhancer, module_name="enhancer")

    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    early_stopping = EarlyStoppingState()
    training_config = {
        "epochs": args.epochs,
        "requested_batch_size": args.batch_size,
        "effective_batch_size": effective_batch_size,
        "min_batch_size": args.min_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "focal_gamma": args.focal_gamma,
        "label_smoothing": args.label_smoothing,
        "detector_input_mode": detector_input_mode,
        "use_dual_input": detector_input_mode == "dual",
        "dual_input_shared_backbone": not args.dual_input_separate_backbones,
        "threshold_objective": args.threshold_objective,
        "monitor": args.monitor,
        "early_stopping_patience": args.early_stopping_patience,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "min_delta": args.min_delta,
        "alpha": alpha.tolist(),
        "lowlight_val_root": str(lowlight_val_root) if lowlight_val_root is not None else None,
        "runtime": runtime,
        "workspace_root": str(workspace_root),
        "data_root": str(data_root),
        "drive_checkpoint_dir": str(drive_checkpoint_dir) if drive_checkpoint_dir else None,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "init_detector_checkpoint": (
            str(resolve_runtime_path(args.init_detector_checkpoint, workspace_root=workspace_root))
            if args.init_detector_checkpoint
            else None
        ),
        "enhancer_checkpoint": (
            str(resolve_runtime_path(args.enhancer_checkpoint, workspace_root=workspace_root))
            if args.enhancer_checkpoint
            else None
        ),
        "threshold_candidates": [float(value) for value in args.threshold_candidates],
        "min_closed_prediction_rate": float(args.min_closed_prediction_rate),
        "max_closed_prediction_rate": float(args.max_closed_prediction_rate),
    }
    model_config = {
        "backbone": args.backbone,
        "num_classes": len(data_bundle.classes),
        "image_size": args.image_size,
        "mobilenet_trainable_blocks": args.mobilenet_trainable_blocks,
        "resnet_trainable_layers": args.resnet_trainable_layers,
        "use_pretrained": not args.no_pretrained,
        "trained_on_enhanced_images": detector_input_mode == "enhanced",
        "detector_input_mode": detector_input_mode,
        "use_dual_input": detector_input_mode == "dual",
        "dual_input_shared_backbone": not args.dual_input_separate_backbones,
    }

    print(f"Runtime: {runtime}")
    print(f"Workspace root: {workspace_root}")
    print(f"Training on device: {device}")
    print(f"Classes: {data_bundle.class_to_idx}")
    print(
        f"Samples: train={len(data_bundle.train_dataset)} | "
        f"val={len(data_bundle.val_dataset)}"
    )
    if lowlight_dataset is not None:
        print(f"Low-light validation samples: {len(lowlight_dataset)}")
    print(f"Focal alpha: {alpha.tolist()}")
    print(f"Best checkpoint path: {best_save_path}")
    print(f"Last checkpoint path: {last_save_path}")
    if init_detector_checkpoint is not None:
        print("Detector initialization: loaded clean-trained detector checkpoint")
    if enhancer is not None:
        print(
            f"Detector input mode '{detector_input_mode}': using a frozen pretrained enhancer "
            "to prepare detector inputs."
        )
        verify_module_stays_frozen(enhancer, module_name="enhancer", context="setup")
    if drive_checkpoint_dir is not None:
        print(f"Drive backup directory: {drive_checkpoint_dir}")

    epoch = 1
    while epoch <= args.epochs:
        try:
            train_metrics = run_epoch(
                model=model,
                dataloader=data_bundle.train_loader,
                criterion=criterion,
                device=device,
                positive_label=positive_label,
                enhancer=enhancer,
                detector_input_mode=detector_input_mode,
                optimizer=optimizer,
                threshold=0.5,
                description=f"Train {epoch}/{args.epochs}",
                show_progress=not args.disable_tqdm,
            )
            val_metrics = run_epoch(
                model=model,
                dataloader=data_bundle.val_loader,
                criterion=criterion,
                device=device,
                positive_label=positive_label,
                enhancer=enhancer,
                detector_input_mode=detector_input_mode,
                tune_threshold=True,
                threshold_objective=args.threshold_objective,
                threshold_candidates=args.threshold_candidates,
                min_positive_rate=args.min_closed_prediction_rate,
                max_positive_rate=args.max_closed_prediction_rate,
                description=f"Val {epoch}/{args.epochs}",
                show_progress=not args.disable_tqdm,
            )
            lowlight_metrics = None
            if lowlight_loader is not None and args.report_lowlight_every_epoch:
                lowlight_metrics = run_epoch(
                    model=model,
                    dataloader=lowlight_loader,
                    criterion=criterion,
                    device=device,
                    positive_label=positive_label,
                    enhancer=enhancer,
                    detector_input_mode=detector_input_mode,
                    threshold=val_metrics["threshold"],
                    description=f"Low-light {epoch}/{args.epochs}",
                    show_progress=not args.disable_tqdm,
                )
        except RuntimeError as exc:
            if (
                args.disable_auto_batch_size
                or not is_out_of_memory_error(exc)
                or effective_batch_size <= args.min_batch_size
            ):
                raise

            new_batch_size = reduce_batch_size(effective_batch_size, args.min_batch_size)
            if new_batch_size >= effective_batch_size:
                raise

            print(
                f"[WARN] Memory limit reached at batch size {effective_batch_size}. "
                f"Retrying epoch {epoch} with batch size {new_batch_size}."
            )
            effective_batch_size = new_batch_size
            training_config["effective_batch_size"] = effective_batch_size

            if device.type == "cuda":
                torch.cuda.empty_cache()

            data_bundle, lowlight_dataset, lowlight_loader = build_training_data(
                dataset_root=data_root,
                lowlight_val_root=lowlight_val_root,
                batch_size=effective_batch_size,
                image_size=args.image_size,
                val_ratio=args.val_ratio,
                seed=args.seed,
                num_workers=args.num_workers,
                device=device,
                detector_input_mode=detector_input_mode,
            )
            continue

        current_score = metric_value(val_metrics, args.monitor)
        scheduler.step(current_score)
        if enhancer is not None:
            verify_module_stays_frozen(enhancer, module_name="enhancer", context=f"epoch-{epoch}")

        print(f"\nEpoch {epoch}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.6f}")
        print(format_classifier_metric_line("Train", train_metrics))
        print(format_classifier_metric_line("Val", val_metrics))
        if lowlight_metrics is not None:
            print(format_classifier_metric_line("LowLight", lowlight_metrics))

        save_checkpoint(
            last_save_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            class_to_idx=data_bundle.class_to_idx,
            classes=data_bundle.classes,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lowlight_metrics=lowlight_metrics,
            model_config=model_config,
            training_config=training_config,
        )
        backup_checkpoints_if_requested(
            checkpoint_paths=[last_save_path],
            drive_checkpoint_dir=drive_checkpoint_dir,
        )

        if current_score > early_stopping.best_score + args.min_delta:
            early_stopping.best_score = current_score
            early_stopping.best_epoch = epoch
            early_stopping.patience_counter = 0
            save_checkpoint(
                best_save_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                class_to_idx=data_bundle.class_to_idx,
                classes=data_bundle.classes,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lowlight_metrics=lowlight_metrics,
                model_config=model_config,
                training_config=training_config,
            )
            backup_checkpoints_if_requested(
                checkpoint_paths=[best_save_path],
                drive_checkpoint_dir=drive_checkpoint_dir,
            )
            print(
                f"Saved new best model at {best_save_path} "
                f"(monitor={args.monitor}, score={current_score:.4f})"
            )
        else:
            early_stopping.patience_counter += 1

        if early_stopping.patience_counter >= args.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best epoch was {early_stopping.best_epoch}."
            )
            break

        epoch += 1

    print(f"\nBest {args.monitor}: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}")
    print(f"Best checkpoint: {best_save_path}")
    print(f"Last checkpoint: {last_save_path}")


if __name__ == "__main__":
    main()
