#!/usr/bin/env python3
"""Prepare a balanced train/validation dataset from labeled images."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import shutil
import sys


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ("open", "closed")


@dataclass
class SplitStats:
    train_open: int = 0
    train_closed: int = 0
    val_open: int = 0
    val_closed: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a balanced train/validation split for eye-state images."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory containing labeled images or extracted frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where dataset/train and dataset/val will be created.",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        help=(
            "Optional CSV file mapping image paths to labels. "
            "Supported columns: path,label"
        ),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of images per class to place in validation. Default: 0.2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling. Default: 42.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "move"),
        default="copy",
        help="Whether to copy or move images into the output dataset. Default: copy.",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable downsampling to equal class counts before splitting.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the output directory before creating the new split.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if args.labels_file is not None and not args.labels_file.exists():
        raise FileNotFoundError(f"Labels file does not exist: {args.labels_file}")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")


def iter_image_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_labels_csv(labels_file: Path, input_dir: Path) -> dict[Path, str]:
    with labels_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Labels file is empty: {labels_file}")

    start_index = 0
    first_row = [value.strip().lower() for value in rows[0]]
    if len(first_row) >= 2 and first_row[0] == "path" and first_row[1] == "label":
        start_index = 1

    labels: dict[Path, str] = {}
    for row_number, row in enumerate(rows[start_index:], start=start_index + 1):
        if len(row) < 2:
            raise ValueError(
                f"Invalid labels row {row_number}: expected at least 2 columns."
            )

        raw_path = row[0].strip()
        raw_label = row[1].strip().lower()
        if raw_label not in CLASS_NAMES:
            raise ValueError(
                f"Invalid label '{raw_label}' on row {row_number}. "
                f"Expected one of: {', '.join(CLASS_NAMES)}"
            )

        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = input_dir / image_path
        image_path = image_path.resolve()
        labels[image_path] = raw_label

    return labels


def infer_label_from_path(image_path: Path) -> str | None:
    for part in image_path.parts:
        lowered = part.lower()
        if lowered in CLASS_NAMES:
            return lowered
    return None


def collect_labeled_images(
    input_dir: Path,
    labels_file: Path | None,
) -> dict[str, list[Path]]:
    image_paths = iter_image_files(input_dir)
    if not image_paths:
        raise ValueError(f"No image files found under {input_dir}")

    labeled_images: dict[str, list[Path]] = {label: [] for label in CLASS_NAMES}

    if labels_file is not None:
        label_map = parse_labels_csv(labels_file, input_dir)
        for image_path in image_paths:
            resolved = image_path.resolve()
            label = label_map.get(resolved)
            if label is not None:
                labeled_images[label].append(image_path)
    else:
        for image_path in image_paths:
            label = infer_label_from_path(image_path.relative_to(input_dir))
            if label is not None:
                labeled_images[label].append(image_path)

    return labeled_images


def balance_classes(
    labeled_images: dict[str, list[Path]],
    balance: bool,
    rng: random.Random,
) -> dict[str, list[Path]]:
    working = {label: list(paths) for label, paths in labeled_images.items()}

    for paths in working.values():
        rng.shuffle(paths)

    if not balance:
        return working

    counts = [len(working[label]) for label in CLASS_NAMES]
    if min(counts) == 0:
        return working

    target_count = min(counts)
    return {label: working[label][:target_count] for label in CLASS_NAMES}


def split_class_paths(
    paths: list[Path],
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[Path], list[Path]]:
    items = list(paths)
    rng.shuffle(items)

    if len(items) <= 1:
        return items, []

    val_count = int(round(len(items) * val_ratio))
    val_count = max(1, val_count)
    val_count = min(val_count, len(items) - 1)

    val_paths = items[:val_count]
    train_paths = items[val_count:]
    return train_paths, val_paths


def sanitize_relative_name(relative_path: Path) -> str:
    parent_prefix = "__".join(relative_path.parts[:-1])
    if not parent_prefix:
        return relative_path.name
    return f"{parent_prefix}__{relative_path.name}"


def destination_path(
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    split_name: str,
    label: str,
) -> Path:
    relative_path = image_path.relative_to(input_dir)
    filename = sanitize_relative_name(relative_path)
    return output_dir / split_name / label / filename


def transfer_file(source: Path, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(source, destination)
    else:
        shutil.move(source, destination)


def build_output_dataset(
    labeled_images: dict[str, list[Path]],
    input_dir: Path,
    output_dir: Path,
    val_ratio: float,
    copy_mode: str,
    rng: random.Random,
) -> SplitStats:
    stats = SplitStats()

    for label in CLASS_NAMES:
        train_paths, val_paths = split_class_paths(labeled_images[label], val_ratio, rng)

        for image_path in train_paths:
            transfer_file(
                source=image_path,
                destination=destination_path(
                    image_path=image_path,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    split_name="train",
                    label=label,
                ),
                copy_mode=copy_mode,
            )

        for image_path in val_paths:
            transfer_file(
                source=image_path,
                destination=destination_path(
                    image_path=image_path,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    split_name="val",
                    label=label,
                ),
                copy_mode=copy_mode,
            )

        if label == "open":
            stats.train_open = len(train_paths)
            stats.val_open = len(val_paths)
        else:
            stats.train_closed = len(train_paths)
            stats.val_closed = len(val_paths)

    return stats


def print_stats(
    original_counts: dict[str, int],
    balanced_counts: dict[str, int],
    stats: SplitStats,
    balance_enabled: bool,
) -> None:
    print("\n=== Dataset Stats ===")
    print(
        "Original labeled counts: "
        f"open={original_counts['open']} | closed={original_counts['closed']}"
    )
    if balance_enabled:
        print(
            "Balanced counts used: "
            f"open={balanced_counts['open']} | closed={balanced_counts['closed']}"
        )
    else:
        print(
            "Counts used without balancing: "
            f"open={balanced_counts['open']} | closed={balanced_counts['closed']}"
        )
    print(
        "Train split: "
        f"open={stats.train_open} | closed={stats.train_closed} | "
        f"total={stats.train_open + stats.train_closed}"
    )
    print(
        "Val split: "
        f"open={stats.val_open} | closed={stats.val_closed} | "
        f"total={stats.val_open + stats.val_closed}"
    )
    print(
        "Total final dataset: "
        f"{stats.train_open + stats.train_closed + stats.val_open + stats.val_closed}"
    )


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.clear_output and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    try:
        labeled_images = collect_labeled_images(args.input_dir, args.labels_file)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    original_counts = {label: len(paths) for label, paths in labeled_images.items()}
    if any(count == 0 for count in original_counts.values()):
        print(
            "[ERROR] Both classes must have at least one image. "
            f"Found open={original_counts['open']} closed={original_counts['closed']}",
            file=sys.stderr,
        )
        return 1

    balanced_images = balance_classes(
        labeled_images=labeled_images,
        balance=not args.no_balance,
        rng=rng,
    )
    balanced_counts = {label: len(paths) for label, paths in balanced_images.items()}

    stats = build_output_dataset(
        labeled_images=balanced_images,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        copy_mode=args.copy_mode,
        rng=rng,
    )
    print_stats(
        original_counts=original_counts,
        balanced_counts=balanced_counts,
        stats=stats,
        balance_enabled=not args.no_balance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
