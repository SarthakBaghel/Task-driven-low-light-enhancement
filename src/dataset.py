#!/usr/bin/env python3
"""Dataset utilities for binary eye-state classification."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Sequence
import warnings

from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PREFERRED_CLASS_ORDER = ("open", "closed")


def build_default_transform(image_size: int | tuple[int, int] = 224) -> transforms.Compose:
    """Return the default image preprocessing pipeline."""
    size = (image_size, image_size) if isinstance(image_size, int) else image_size
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def is_image_file(path: Path) -> bool:
    """Return True when the path points to a supported image file."""
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def discover_class_folders(root: str | Path) -> list[str]:
    """Detect class folders directly under the dataset root."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

    class_names = []
    for child in sorted(root_path.iterdir()):
        if child.is_dir() and any(is_image_file(path) for path in child.rglob("*")):
            class_names.append(child.name)

    if not class_names:
        raise ValueError(
            f"No class folders containing images were found under: {root_path}"
        )

    preferred = [name for name in PREFERRED_CLASS_ORDER if name in class_names]
    remaining = sorted(name for name in class_names if name not in preferred)
    return preferred + remaining


def build_class_to_idx(class_names: Sequence[str]) -> dict[str, int]:
    """Build a deterministic mapping from class names to integer labels."""
    return {class_name: index for index, class_name in enumerate(class_names)}


def make_dataset(
    root: str | Path,
    class_to_idx: dict[str, int],
) -> list[tuple[Path, int]]:
    """Scan the dataset root and return a list of image paths with labels."""
    root_path = Path(root).expanduser().resolve()
    samples: list[tuple[Path, int]] = []

    for class_name, class_index in sorted(class_to_idx.items(), key=lambda item: item[1]):
        class_dir = root_path / class_name
        if not class_dir.is_dir():
            continue

        for image_path in sorted(class_dir.rglob("*")):
            if is_image_file(image_path):
                samples.append((image_path, class_index))

    if not samples:
        raise ValueError(f"No images found under class folders in: {root_path}")

    return samples


class EyeStateDataset(Dataset):
    """PyTorch dataset for folder-based eye-state classification."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        samples: Sequence[tuple[str | Path, int]] | None = None,
        class_to_idx: dict[str, int] | None = None,
        transform: Callable | None = None,
        image_size: int | tuple[int, int] = 224,
    ) -> None:
        if root is None and samples is None:
            raise ValueError("Either 'root' or 'samples' must be provided.")

        self.root = Path(root).expanduser().resolve() if root is not None else None
        self.transform = transform or build_default_transform(image_size)

        if samples is None:
            detected_classes = discover_class_folders(self.root)
            self.class_to_idx = class_to_idx or build_class_to_idx(detected_classes)
            self.classes = sorted(self.class_to_idx, key=self.class_to_idx.get)
            self.samples = make_dataset(self.root, self.class_to_idx)
        else:
            if class_to_idx is None:
                raise ValueError("'class_to_idx' is required when 'samples' are provided.")

            self.class_to_idx = dict(class_to_idx)
            self.classes = sorted(self.class_to_idx, key=self.class_to_idx.get)
            self.samples = [(Path(path), int(label)) for path, label in samples]
            if not self.samples:
                raise ValueError("The provided sample list is empty.")

        self.targets = [label for _, label in self.samples]
        self._bad_indices: set[int] = set()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self.samples):
            raise IndexError(f"Index out of range: {index}")
        return self._load_with_fallback(index)

    def _load_with_fallback(self, index: int):
        current_index = index
        attempts = 0

        while attempts < len(self.samples):
            image_path, label = self.samples[current_index]
            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                image_tensor = self.transform(image) if self.transform is not None else image
                return image_tensor, label
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                if current_index not in self._bad_indices:
                    self._bad_indices.add(current_index)
                    warnings.warn(
                        f"Skipping unreadable image: {image_path} ({exc})",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                current_index = (current_index + 1) % len(self.samples)
                attempts += 1

        raise RuntimeError("No readable images are available in the dataset.")

    def get_class_distribution(self) -> dict[str, int]:
        """Return a class-count dictionary."""
        counts = Counter(self.targets)
        return {
            class_name: counts[self.class_to_idx[class_name]]
            for class_name in self.classes
        }

    @property
    def num_corrupted_files(self) -> int:
        """Return the number of files marked as unreadable during loading."""
        return len(self._bad_indices)
