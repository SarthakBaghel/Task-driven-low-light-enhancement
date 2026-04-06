#!/usr/bin/env python3
"""Utility functions for inspecting and visualizing eye-state datasets."""

from __future__ import annotations

from collections import Counter
import random

import matplotlib.pyplot as plt
import torch

from dataset import IMAGENET_MEAN, IMAGENET_STD


def get_class_distribution(dataset) -> dict[str, int]:
    """Return the class distribution for a dataset with classes and targets."""
    counts = Counter(dataset.targets)
    return {
        class_name: counts[dataset.class_to_idx[class_name]]
        for class_name in dataset.classes
    }


def print_dataset_summary(train_loader, val_loader) -> None:
    """Print class counts, total sample counts, and one batch shape."""
    if len(train_loader.dataset) == 0:
        raise ValueError("The training dataset is empty.")

    train_distribution = get_class_distribution(train_loader.dataset)
    val_distribution = get_class_distribution(val_loader.dataset)

    total_distribution = {
        class_name: train_distribution.get(class_name, 0) + val_distribution.get(class_name, 0)
        for class_name in train_loader.dataset.classes
    }

    sample_images, sample_labels = next(iter(train_loader))

    print("=== Dataset Summary ===")
    print(f"Class to index: {train_loader.dataset.class_to_idx}")
    print(f"Train distribution: {train_distribution}")
    print(f"Validation distribution: {val_distribution}")
    print(f"Total distribution: {total_distribution}")
    print(
        "Total samples: "
        f"train={len(train_loader.dataset)} | val={len(val_loader.dataset)} | "
        f"all={len(train_loader.dataset) + len(val_loader.dataset)}"
    )
    print(
        "Sample batch shape: "
        f"images={tuple(sample_images.shape)} | labels={tuple(sample_labels.shape)}"
    )


def denormalize_image(
    image_tensor: torch.Tensor,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> torch.Tensor:
    """Undo ImageNet normalization for visualization."""
    mean_tensor = torch.tensor(mean, dtype=image_tensor.dtype, device=image_tensor.device)
    std_tensor = torch.tensor(std, dtype=image_tensor.dtype, device=image_tensor.device)
    mean_tensor = mean_tensor.view(-1, 1, 1)
    std_tensor = std_tensor.view(-1, 1, 1)
    return image_tensor * std_tensor + mean_tensor


def visualize_random_images(dataset, num_images: int = 8, seed: int | None = None) -> None:
    """Visualize random images from a dataset with their class labels."""
    if len(dataset) == 0:
        raise ValueError("Cannot visualize images from an empty dataset.")

    rng = random.Random(seed)
    num_images = min(num_images, len(dataset))
    selected_indices = rng.sample(range(len(dataset)), k=num_images)

    rows = 2
    cols = 4
    figure, axes = plt.subplots(rows, cols, figsize=(16, 8))
    axes = axes.flatten()

    idx_to_class = {index: class_name for class_name, index in dataset.class_to_idx.items()}

    for axis, sample_index in zip(axes, selected_indices):
        image_tensor, label = dataset[sample_index]
        image_tensor = denormalize_image(image_tensor).clamp(0.0, 1.0)
        image_array = image_tensor.permute(1, 2, 0).cpu().numpy()

        axis.imshow(image_array)
        axis.set_title(f"{idx_to_class[label]} ({label})")
        axis.axis("off")

    for axis in axes[num_images:]:
        axis.axis("off")

    plt.tight_layout()
    plt.show()
