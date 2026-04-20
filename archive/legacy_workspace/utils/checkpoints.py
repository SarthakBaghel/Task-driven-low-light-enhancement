#!/usr/bin/env python3
"""Checkpoint helpers for joint enhancer-detector training."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _serialize_config(config: Any) -> Any:
    """Convert dataclasses into plain dictionaries before saving."""
    if config is None:
        return None
    if is_dataclass(config):
        return asdict(config)
    return config


def build_checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
    train_config: Any,
    class_to_idx: dict[str, int],
    classes: list[str],
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    best_val_accuracy: float,
    scaler: torch.amp.GradScaler | torch.cuda.amp.GradScaler | None = None,
    extra_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a serializable checkpoint dictionary."""
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "model_config": _serialize_config(model_config),
        "train_config": _serialize_config(train_config),
        "class_to_idx": class_to_idx,
        "classes": classes,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_val_accuracy": best_val_accuracy,
    }
    if extra_state:
        payload["extra_state"] = extra_state
    return payload


def save_checkpoint(payload: dict[str, Any], save_path: str | Path) -> Path:
    """Write a checkpoint payload to disk."""
    path = Path(save_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def save_training_checkpoint(
    *,
    save_path: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
    train_config: Any,
    class_to_idx: dict[str, int],
    classes: list[str],
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    best_val_accuracy: float,
    scaler: torch.amp.GradScaler | torch.cuda.amp.GradScaler | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    """Build and save a training checkpoint in one call."""
    payload = build_checkpoint_payload(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        train_config=train_config,
        class_to_idx=class_to_idx,
        classes=classes,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        best_val_accuracy=best_val_accuracy,
        scaler=scaler,
        extra_state=extra_state,
    )
    return save_checkpoint(payload, save_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint dictionary from disk."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
