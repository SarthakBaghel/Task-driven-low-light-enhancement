#!/usr/bin/env python3
"""Create a smaller balanced image subset for fast experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import shutil


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ClassGroup:
    """Represents one balanced sampling group such as train/open vs train/closed."""

    group_name: str
    class_name: str
    class_root: Path
    images: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a smaller balanced subset of an image dataset."
    )
    parser.add_argument(
        "input_root",
        type=str,
        help="Dataset root. Supports either class folders directly or split/class folders.",
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="New directory where the sampled subset will be copied.",
    )
    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument(
        "--samples-per-class",
        type=int,
        help="Fixed number of images to copy per class.",
    )
    selection_group.add_argument(
        "--percent-per-class",
        type=float,
        help=(
            "Percentage of each class to sample. To keep the subset balanced, "
            "the final count per class becomes the minimum requested count across classes."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling images.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="selected_files.csv",
        help="Filename used for the CSV manifest inside the output folder.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def discover_top_level_class_dirs(root: Path) -> list[Path]:
    class_dirs = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and any(is_image_file(path) for path in child.rglob("*")):
            class_dirs.append(child)
    return class_dirs


def collect_grouped_class_dirs(root: Path) -> dict[str, list[Path]]:
    """Collect class directories either directly under root or inside split folders."""
    split_groups: dict[str, list[Path]] = {}
    for split_name in SPLIT_NAMES:
        split_root = root / split_name
        if not split_root.is_dir():
            continue
        class_dirs = discover_top_level_class_dirs(split_root)
        if class_dirs:
            split_groups[split_name] = class_dirs

    if split_groups:
        return split_groups

    class_dirs = discover_top_level_class_dirs(root)
    if class_dirs:
        return {"root": class_dirs}

    raise ValueError(
        "Could not find class folders under the dataset root. Expected either "
        "`open/closed` style folders or `train/open`, `train/closed` style folders."
    )


def collect_groups(root: Path) -> list[ClassGroup]:
    groups: list[ClassGroup] = []
    grouped_dirs = collect_grouped_class_dirs(root)
    for group_name, class_dirs in grouped_dirs.items():
        for class_dir in class_dirs:
            images = tuple(sorted(path for path in class_dir.rglob("*") if is_image_file(path)))
            if not images:
                continue
            groups.append(
                ClassGroup(
                    group_name=group_name,
                    class_name=class_dir.name,
                    class_root=class_dir,
                    images=images,
                )
            )

    if not groups:
        raise ValueError(f"No image files found under: {root}")
    return groups


def resolve_target_count(
    grouped_classes: list[ClassGroup],
    *,
    samples_per_class: int | None,
    percent_per_class: float | None,
) -> int:
    if samples_per_class is not None:
        if samples_per_class <= 0:
            raise ValueError("--samples-per-class must be greater than 0.")
        smallest_class = min(len(group.images) for group in grouped_classes)
        if samples_per_class > smallest_class:
            raise ValueError(
                f"Requested {samples_per_class} images per class, but the smallest class in this "
                f"group has only {smallest_class} images."
            )
        return samples_per_class

    if percent_per_class is None:
        raise ValueError("Either samples_per_class or percent_per_class must be provided.")
    if not 0.0 < percent_per_class <= 100.0:
        raise ValueError("--percent-per-class must be in the range (0, 100].")

    fraction = percent_per_class / 100.0
    requested_counts = [max(1, int(len(group.images) * fraction)) for group in grouped_classes]
    target_count = min(requested_counts)
    if target_count <= 0:
        raise ValueError("The requested percentage is too small to select any images.")
    return target_count


def ensure_output_root_is_safe(output_root: Path) -> None:
    if not output_root.exists():
        return
    if any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_root}\n"
            "Please choose a fresh output folder."
        )


def sample_group(
    grouped_classes: list[ClassGroup],
    *,
    target_count: int,
    rng: random.Random,
) -> dict[str, list[Path]]:
    sampled: dict[str, list[Path]] = {}
    for group in grouped_classes:
        selected = rng.sample(list(group.images), k=target_count)
        sampled[group.class_name] = sorted(selected)
    return sampled


def copy_selected_images(
    *,
    selected_paths: list[Path],
    input_root: Path,
    output_root: Path,
) -> list[tuple[Path, Path]]:
    copied: list[tuple[Path, Path]] = []
    for source_path in selected_paths:
        relative_path = source_path.relative_to(input_root)
        destination_path = output_root / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        copied.append((source_path, destination_path))
    return copied


def group_by_container(groups: list[ClassGroup]) -> dict[str, list[ClassGroup]]:
    grouped: dict[str, list[ClassGroup]] = {}
    for group in groups:
        grouped.setdefault(group.group_name, []).append(group)
    return grouped


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input dataset root does not exist: {input_root}")

    ensure_output_root_is_safe(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    groups = collect_groups(input_root)
    grouped_containers = group_by_container(groups)

    csv_rows: list[dict[str, object]] = []
    final_counts: dict[str, dict[str, int]] = {}

    for group_name, grouped_classes in sorted(grouped_containers.items()):
        target_count = resolve_target_count(
            grouped_classes,
            samples_per_class=args.samples_per_class,
            percent_per_class=args.percent_per_class,
        )
        sampled_by_class = sample_group(grouped_classes, target_count=target_count, rng=rng)

        for class_group in grouped_classes:
            selected_paths = sampled_by_class[class_group.class_name]
            copied_pairs = copy_selected_images(
                selected_paths=selected_paths,
                input_root=input_root,
                output_root=output_root,
            )
            final_counts.setdefault(group_name, {})[class_group.class_name] = len(copied_pairs)
            for source_path, destination_path in copied_pairs:
                csv_rows.append(
                    {
                        "group": group_name,
                        "class_name": class_group.class_name,
                        "source_path": str(source_path),
                        "destination_path": str(destination_path),
                    }
                )

    csv_path = output_root / args.csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group", "class_name", "source_path", "destination_path"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved subset to: {output_root}")
    print(f"Saved CSV manifest to: {csv_path}")
    print("Final class counts:")
    for group_name, class_counts in sorted(final_counts.items()):
        count_parts = ", ".join(
            f"{class_name}={count}"
            for class_name, count in sorted(class_counts.items())
        )
        label = "root" if group_name == "root" else group_name
        print(f"  {label}: {count_parts}")


if __name__ == "__main__":
    main()
