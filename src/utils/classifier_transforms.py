#!/usr/bin/env python3
"""Augmentation and preprocessing helpers for transfer-learning classifiers."""

from __future__ import annotations

from typing import Sequence

import torch
from torchvision import transforms

from dataset import IMAGENET_MEAN, IMAGENET_STD


class AddGaussianNoise:
    """Add Gaussian noise to a tensor image in [0, 1]."""

    def __init__(self, std_range: tuple[float, float] = (0.0, 0.05), p: float = 0.5) -> None:
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return tensor
        std = torch.empty(1).uniform_(self.std_range[0], self.std_range[1]).item()
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0.0, 1.0)


def normalize_tensor_for_detector(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a BCHW image tensor using ImageNet statistics."""
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(
            "Expected a BCHW RGB tensor for detector normalization, "
            f"but received {tuple(tensor.shape)}."
        )
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std


def build_transfer_learning_raw_transforms(
    image_size: int | Sequence[int] = 224,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return transforms that keep images in [0, 1] for enhancer-first training.

    These transforms mirror the transfer-learning augmentations but skip the
    final ImageNet normalization because the enhancer expects raw image values.
    """
    size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)

    train_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.45, contrast=0.45),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.8))],
                p=0.3,
            ),
            transforms.ToTensor(),
            AddGaussianNoise(std_range=(0.01, 0.06), p=0.5),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )
    return train_transform, val_transform


def build_transfer_learning_transforms(
    image_size: int | Sequence[int] = 224,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return train/validation transforms for robust eye-state classification."""
    size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)

    train_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.45, contrast=0.45),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.8))],
                p=0.3,
            ),
            transforms.ToTensor(),
            AddGaussianNoise(std_range=(0.01, 0.06), p=0.5),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, val_transform
