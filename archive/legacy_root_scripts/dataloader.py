#!/usr/bin/env python3
"""Dataloader helpers for eye-state classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import EyeStateDataset, build_default_transform, discover_class_folders


@dataclass
class DataBundle:
    train_dataset: EyeStateDataset
    val_dataset: EyeStateDataset
    train_loader: DataLoader
    val_loader: DataLoader
    classes: list[str]
    class_to_idx: dict[str, int]


def seed_worker(worker_id: int) -> None:
    """Seed numpy and python random inside each DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def has_predefined_split(dataset_root: str | Path) -> bool:
    """Return True if the root already contains train/ and val/ folders."""
    root_path = Path(dataset_root).expanduser().resolve()
    train_root = root_path / "train"
    val_root = root_path / "val"

    if not train_root.is_dir() or not val_root.is_dir():
        return False

    try:
        discover_class_folders(train_root)
        discover_class_folders(val_root)
    except ValueError:
        return False

    return True


def stratified_split_samples(
    samples: list[tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Split samples into train and validation sets while preserving class balance."""
    rng = random.Random(seed)
    grouped: dict[int, list[tuple[Path, int]]] = {}
    for sample in samples:
        grouped.setdefault(sample[1], []).append(sample)

    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []

    for class_index in sorted(grouped):
        class_samples = list(grouped[class_index])
        rng.shuffle(class_samples)

        if len(class_samples) == 1:
            train_samples.extend(class_samples)
            continue

        val_count = int(round(len(class_samples) * val_ratio))
        val_count = max(1, val_count)
        val_count = min(val_count, len(class_samples) - 1)

        val_samples.extend(class_samples[:val_count])
        train_samples.extend(class_samples[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def create_datasets(
    dataset_root: str | Path,
    *,
    image_size: int | tuple[int, int] = 224,
    val_ratio: float = 0.2,
    seed: int = 42,
    train_transform=None,
    val_transform=None,
) -> tuple[EyeStateDataset, EyeStateDataset]:
    """Create train and validation datasets from either a split or unsplit root."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("'val_ratio' must be between 0 and 1.")

    root_path = Path(dataset_root).expanduser().resolve()
    train_transform = train_transform or build_default_transform(image_size)
    val_transform = val_transform or build_default_transform(image_size)

    if has_predefined_split(root_path):
        train_dataset = EyeStateDataset(root=root_path / "train", transform=train_transform)
        val_dataset = EyeStateDataset(
            root=root_path / "val",
            class_to_idx=train_dataset.class_to_idx,
            transform=val_transform,
        )
        return train_dataset, val_dataset

    full_dataset = EyeStateDataset(root=root_path, transform=train_transform)
    train_samples, val_samples = stratified_split_samples(
        samples=full_dataset.samples,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_dataset = EyeStateDataset(
        samples=train_samples,
        class_to_idx=full_dataset.class_to_idx,
        transform=train_transform,
    )
    val_dataset = EyeStateDataset(
        samples=val_samples,
        class_to_idx=full_dataset.class_to_idx,
        transform=val_transform,
    )
    return train_dataset, val_dataset


def create_dataloaders(
    dataset_root: str | Path,
    *,
    batch_size: int = 32,
    image_size: int | tuple[int, int] = 224,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 2,
    pin_memory: bool | None = None,
    worker_init_fn=None,
    train_transform=None,
    val_transform=None,
) -> DataBundle:
    """Create train and validation dataloaders."""
    if batch_size <= 0:
        raise ValueError("'batch_size' must be greater than 0.")
    if num_workers < 0:
        raise ValueError("'num_workers' must be 0 or greater.")

    train_dataset, val_dataset = create_datasets(
        dataset_root=dataset_root,
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    use_pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent_workers,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
    )

    return DataBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        classes=train_dataset.classes,
        class_to_idx=train_dataset.class_to_idx,
    )
