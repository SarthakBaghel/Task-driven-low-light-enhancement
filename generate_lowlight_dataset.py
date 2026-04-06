#!/usr/bin/env python3
"""Generate a randomized low-light version of a binary eye-state dataset."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import random
import sys

from low_light_simulator import (
    LowLightConfig,
    degrade_image,
    read_image,
    validate_config,
)


EXPECTED_CLASSES = ("open", "closed")
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOG_FIELDNAMES = [
    "class_name",
    "original_file",
    "output_file",
    "profile",
    "effects_used",
    "gamma",
    "brightness_factor",
    "contrast_factor",
    "black_level_shift",
    "gaussian_sigma",
    "poisson_strength",
    "motion_blur_kernel",
    "motion_blur_angle",
    "desaturation_factor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a randomized low-light copy of an eye-state dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Dataset root containing open/ and closed/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root where lowlight/open and lowlight/closed will be created.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Optional total number of images to process. Defaults to all images.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.6,
        help="Degradation strength in the range [0, 1]. Default: 0.6.",
    )
    parser.add_argument(
        "--profile",
        choices=("standard", "severe", "extreme"),
        default="standard",
        help=(
            "Low-light difficulty preset. 'standard' preserves the current mild "
            "behavior, while 'severe' and 'extreme' produce much darker images."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling and degradation. Default: 42.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=100,
        help="Print progress every N images. Default: 100.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="degradation_log.csv",
        help="Filename for the CSV log written inside the output directory.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {args.input_dir}")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be greater than 0.")
    if not 0.0 <= args.strength <= 1.0:
        raise ValueError("--strength must be between 0 and 1.")
    if args.report_every <= 0:
        raise ValueError("--report-every must be greater than 0.")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def discover_class_dirs(input_dir: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for child in input_dir.iterdir():
        if not child.is_dir():
            continue
        lowered = child.name.lower()
        if lowered in EXPECTED_CLASSES:
            discovered[lowered] = child

    missing = [class_name for class_name in EXPECTED_CLASSES if class_name not in discovered]
    if missing:
        raise ValueError(
            "Input dataset must contain both class folders. Missing: "
            + ", ".join(missing)
        )
    return discovered


def collect_class_images(class_dirs: dict[str, Path]) -> dict[str, list[Path]]:
    images_by_class: dict[str, list[Path]] = {}
    for class_name, class_dir in class_dirs.items():
        image_paths = sorted(path for path in class_dir.rglob("*") if is_image_file(path))
        if not image_paths:
            raise ValueError(f"No image files found under class folder: {class_dir}")
        images_by_class[class_name] = image_paths
    return images_by_class


def select_images(
    images_by_class: dict[str, list[Path]],
    num_samples: int | None,
    rng: random.Random,
) -> list[tuple[str, Path]]:
    shuffled = {name: list(paths) for name, paths in images_by_class.items()}
    for paths in shuffled.values():
        rng.shuffle(paths)

    available = sum(len(paths) for paths in shuffled.values())
    if num_samples is None or num_samples >= available:
        selected: list[tuple[str, Path]] = []
        for class_name in EXPECTED_CLASSES:
            selected.extend((class_name, path) for path in shuffled[class_name])
        rng.shuffle(selected)
        return selected

    selected: list[tuple[str, Path]] = []
    class_queues = {name: list(paths) for name, paths in shuffled.items()}

    while len(selected) < num_samples and any(class_queues.values()):
        for class_name in EXPECTED_CLASSES:
            if len(selected) >= num_samples:
                break
            if class_queues[class_name]:
                selected.append((class_name, class_queues[class_name].pop()))

    rng.shuffle(selected)
    return selected


def interpolate(min_value: float, max_value: float, strength: float) -> float:
    return min_value + (max_value - min_value) * strength


def sample_config(
    strength: float,
    rng: random.Random,
    profile: str = "standard",
) -> tuple[LowLightConfig, list[str]]:
    profile_ranges = {
        "standard": {
            "gamma": (1.15 + 0.15 * strength, 1.45 + 1.75 * strength),
            "brightness": (
                interpolate(0.82, 0.45, strength),
                interpolate(0.96, 0.72, strength),
            ),
            "contrast": (
                interpolate(0.88, 0.52, strength),
                interpolate(0.98, 0.82, strength),
            ),
            "black_level": (0.0, interpolate(0.0, 0.08, strength)),
            "gaussian_prob": interpolate(0.35, 0.9, strength),
            "gaussian_range": (
                interpolate(1.0, 4.0, strength),
                interpolate(4.0, 18.0, strength),
            ),
            "poisson_prob": interpolate(0.4, 0.95, strength),
            "poisson_range": (
                interpolate(0.08, 0.18, strength),
                interpolate(0.2, 0.75, strength),
            ),
            "motion_prob": interpolate(0.1, 0.55, strength),
            "desat_prob": interpolate(0.2, 0.8, strength),
            "desat_range": (
                interpolate(0.03, 0.08, strength),
                interpolate(0.15, 0.6, strength),
            ),
        },
        "severe": {
            # Severe is aimed at joint training: hard for the detector, but still
            # recoverable enough that illumination-focused enhancement can help.
            "gamma": (1.35 + 0.20 * strength, 1.9 + 0.95 * strength),
            "brightness": (
                interpolate(0.76, 0.42, strength),
                interpolate(0.90, 0.68, strength),
            ),
            "contrast": (
                interpolate(0.82, 0.52, strength),
                interpolate(0.94, 0.74, strength),
            ),
            "black_level": (
                interpolate(0.01, 0.02, strength),
                interpolate(0.05, 0.08, strength),
            ),
            "gaussian_prob": interpolate(0.30, 0.70, strength),
            "gaussian_range": (
                interpolate(1.0, 2.0, strength),
                interpolate(4.0, 9.0, strength),
            ),
            "poisson_prob": interpolate(0.25, 0.65, strength),
            "poisson_range": (
                interpolate(0.06, 0.12, strength),
                interpolate(0.18, 0.32, strength),
            ),
            "motion_prob": interpolate(0.10, 0.35, strength),
            "desat_prob": interpolate(0.20, 0.55, strength),
            "desat_range": (
                interpolate(0.03, 0.08, strength),
                interpolate(0.12, 0.28, strength),
            ),
        },
        "extreme": {
            "gamma": (2.2 + 0.4 * strength, 3.2 + 2.5 * strength),
            "brightness": (
                interpolate(0.55, 0.12, strength),
                interpolate(0.72, 0.32, strength),
            ),
            "contrast": (
                interpolate(0.60, 0.18, strength),
                interpolate(0.78, 0.42, strength),
            ),
            "black_level": (
                interpolate(0.08, 0.14, strength),
                interpolate(0.18, 0.32, strength),
            ),
            "gaussian_prob": interpolate(0.8, 1.0, strength),
            "gaussian_range": (
                interpolate(6.0, 12.0, strength),
                interpolate(16.0, 30.0, strength),
            ),
            "poisson_prob": interpolate(0.85, 1.0, strength),
            "poisson_range": (
                interpolate(0.28, 0.40, strength),
                interpolate(0.65, 1.0, strength),
            ),
            "motion_prob": interpolate(0.35, 0.8, strength),
            "desat_prob": interpolate(0.7, 1.0, strength),
            "desat_range": (
                interpolate(0.18, 0.30, strength),
                interpolate(0.50, 0.90, strength),
            ),
        },
    }
    ranges = profile_ranges[profile]

    gamma = rng.uniform(*ranges["gamma"])
    brightness_factor = rng.uniform(*ranges["brightness"])
    contrast_factor = rng.uniform(*ranges["contrast"])
    black_level_shift = rng.uniform(*ranges["black_level"])

    effects_used = ["gamma", "brightness", "contrast"]
    if black_level_shift > 0.0:
        effects_used.append("black_level")

    use_gaussian = rng.random() < ranges["gaussian_prob"]
    gaussian_sigma = rng.uniform(*ranges["gaussian_range"]) if use_gaussian else 0.0
    if use_gaussian:
        effects_used.append("gaussian_noise")

    use_poisson = rng.random() < ranges["poisson_prob"]
    poisson_strength = rng.uniform(*ranges["poisson_range"]) if use_poisson else 0.0
    if use_poisson:
        effects_used.append("poisson_noise")

    use_motion_blur = rng.random() < ranges["motion_prob"]
    if use_motion_blur:
        allowed_kernels = [3, 5, 7, 9, 11]
        max_index = max(1, min(len(allowed_kernels), 1 + int(round(strength * 4))))
        motion_blur_kernel = rng.choice(allowed_kernels[:max_index])
        motion_blur_angle = rng.uniform(0.0, 180.0)
        effects_used.append("motion_blur")
    else:
        motion_blur_kernel = 0
        motion_blur_angle = 0.0

    use_desaturation = rng.random() < ranges["desat_prob"]
    desaturation_factor = rng.uniform(*ranges["desat_range"]) if use_desaturation else 0.0
    if use_desaturation:
        effects_used.append("desaturation")

    config = LowLightConfig(
        gamma=gamma,
        brightness_factor=brightness_factor,
        contrast_factor=contrast_factor,
        black_level_shift=black_level_shift,
        gaussian_sigma=gaussian_sigma,
        poisson_strength=poisson_strength,
        motion_blur_kernel=int(motion_blur_kernel),
        motion_blur_angle=motion_blur_angle,
        desaturation_factor=desaturation_factor,
    )
    validate_config(config)
    return config, effects_used


def make_traceable_filename(image_path: Path, class_dir: Path) -> str:
    relative_path = image_path.relative_to(class_dir)
    stem_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
    safe_stem = "__".join(part.replace(" ", "_") for part in stem_parts)
    return f"{safe_stem}__lowlight{relative_path.suffix.lower()}"


def ensure_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name in EXPECTED_CLASSES:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)


def write_log_header(csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDNAMES)
        writer.writeheader()


def append_log_row(writer: csv.DictWriter, row: dict[str, object]) -> None:
    writer.writerow(row)


def generate_lowlight_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    num_samples: int | None,
    strength: float,
    profile: str,
    seed: int,
    report_every: int,
    csv_name: str,
) -> Path:
    class_dirs = discover_class_dirs(input_dir)
    images_by_class = collect_class_images(class_dirs)

    python_rng = random.Random(seed)
    numpy_rng = np_random_from_seed(seed)
    selected = select_images(images_by_class, num_samples, python_rng)

    ensure_output_dirs(output_dir)
    csv_path = output_dir / csv_name
    write_log_header(csv_path)

    processed = 0
    skipped = 0

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDNAMES)

        for index, (class_name, image_path) in enumerate(selected, start=1):
            try:
                image = read_image(image_path)
            except ValueError as exc:
                skipped += 1
                print(f"[WARN] {exc}", file=sys.stderr)
                continue

            config, effects_used = sample_config(strength, python_rng, profile=profile)
            degraded = degrade_image(image, config, rng=numpy_rng)

            output_filename = make_traceable_filename(image_path, class_dirs[class_name])
            output_path = output_dir / class_name / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

            success = cv2_imwrite(output_path, degraded)
            if not success:
                skipped += 1
                print(f"[WARN] Failed to write image: {output_path}", file=sys.stderr)
                continue

            append_log_row(
                writer,
                {
                    "class_name": class_name,
                    "original_file": str(image_path.relative_to(input_dir)),
                    "output_file": str(output_path.relative_to(output_dir)),
                    "profile": profile,
                    "effects_used": "|".join(effects_used),
                    **asdict(config),
                },
            )
            processed += 1

            if index == 1 or index % report_every == 0 or index == len(selected):
                print(
                    f"[{index}/{len(selected)}] processed={processed} skipped={skipped} "
                    f"last_output={output_path.relative_to(output_dir)}"
                )

    print(
        f"Finished low-light dataset generation. "
        f"processed={processed}, skipped={skipped}, log={csv_path}"
    )
    return csv_path


def cv2_imwrite(path: Path, image) -> bool:
    import cv2

    return bool(cv2.imwrite(str(path), image))


def np_random_from_seed(seed: int):
    import numpy as np

    return np.random.default_rng(seed)


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        generate_lowlight_dataset(
            input_dir=args.input_dir.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            num_samples=args.num_samples,
            strength=args.strength,
            profile=args.profile,
            seed=args.seed,
            report_every=args.report_every,
            csv_name=args.csv_name,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
