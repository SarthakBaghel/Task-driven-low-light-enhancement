#!/usr/bin/env python3
"""Reusable low-light simulation utilities for eye-state images."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class LowLightConfig:
    """Configuration for simulating realistic low-light image degradation."""

    gamma: float = 2.2
    brightness_factor: float = 0.75
    contrast_factor: float = 0.85
    black_level_shift: float = 0.0
    gaussian_sigma: float = 8.0
    poisson_strength: float = 0.35
    motion_blur_kernel: int = 0
    motion_blur_angle: float = 0.0
    desaturation_factor: float = 0.15


def validate_config(config: LowLightConfig) -> None:
    """Validate low-light simulation parameters."""
    if config.gamma <= 1.0:
        raise ValueError("gamma must be greater than 1.0 to darken the image.")
    if not 0.0 < config.brightness_factor <= 1.0:
        raise ValueError("brightness_factor must be in the range (0, 1].")
    if not 0.0 < config.contrast_factor <= 1.0:
        raise ValueError("contrast_factor must be in the range (0, 1].")
    if not 0.0 <= config.black_level_shift < 1.0:
        raise ValueError("black_level_shift must be in the range [0, 1).")
    if config.gaussian_sigma < 0.0:
        raise ValueError("gaussian_sigma must be 0 or greater.")
    if not 0.0 <= config.poisson_strength <= 1.0:
        raise ValueError("poisson_strength must be in the range [0, 1].")
    if not 0.0 <= config.desaturation_factor <= 1.0:
        raise ValueError("desaturation_factor must be in the range [0, 1].")
    if config.motion_blur_kernel < 0:
        raise ValueError("motion_blur_kernel must be 0 or greater.")


def is_image_file(path: Path) -> bool:
    """Return True when the path points to a supported image file."""
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def iter_image_paths(input_path: Path) -> list[Path]:
    """Collect image files from a single image path or a directory tree."""
    if input_path.is_file():
        if not is_image_file(input_path):
            raise ValueError(f"Unsupported image file: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    image_paths = sorted(path for path in input_path.rglob("*") if is_image_file(path))
    if not image_paths:
        raise ValueError(f"No image files found under: {input_path}")
    return image_paths


def read_image(path: Path) -> np.ndarray:
    """Read an image in BGR format."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def to_float_image(image: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR image to float32 in the range [0, 1]."""
    return image.astype(np.float32) / 255.0


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    """Convert float image in the range [0, 1] back to uint8 BGR."""
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)


def apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """Darken mid-tones to mimic underexposed night captures from cheap sensors."""
    return np.power(np.clip(image, 0.0, 1.0), gamma)


def reduce_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Lower overall illumination to simulate reduced exposure in dim scenes."""
    return np.clip(image * factor, 0.0, 1.0)


def reduce_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Compress intensity differences because low light often flattens local contrast."""
    return np.clip((image - 0.5) * factor + 0.5, 0.0, 1.0)


def crush_shadows(image: np.ndarray, shift: float) -> np.ndarray:
    """Subtract a black-level offset to push dark regions closer to near-black."""
    if shift <= 0.0:
        return image
    return np.clip(image - shift, 0.0, 1.0)


def desaturate_colors(image: np.ndarray, factor: float) -> np.ndarray:
    """Night scenes often lose color richness because sensors receive weak chroma signals."""
    if factor <= 0.0:
        return image

    gray = cv2.cvtColor(to_uint8_image(image), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = np.repeat(gray[:, :, None], 3, axis=2)
    return np.clip((1.0 - factor) * image + factor * gray, 0.0, 1.0)


def build_motion_blur_kernel(kernel_size: int, angle_degrees: float) -> np.ndarray:
    """Create a directional blur kernel to mimic camera or subject motion at longer exposure."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0

    center = (kernel_size / 2.0 - 0.5, kernel_size / 2.0 - 0.5)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
    kernel_sum = rotated.sum()
    if kernel_sum <= 0.0:
        return kernel
    return rotated / kernel_sum


def apply_motion_blur(image: np.ndarray, kernel_size: int, angle_degrees: float) -> np.ndarray:
    """Blur along one direction to simulate movement during a longer low-light exposure."""
    if kernel_size <= 1:
        return image

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = build_motion_blur_kernel(kernel_size, angle_degrees)
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate read noise from a sensor operating in low-light conditions."""
    if sigma <= 0.0:
        return image

    noise = rng.normal(loc=0.0, scale=sigma / 255.0, size=image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)


def add_poisson_noise(
    image: np.ndarray,
    strength: float,
    rng: np.random.Generator,
    peak_photons: float = 30.0,
) -> np.ndarray:
    """Simulate photon shot noise, which becomes more visible when very little light is available."""
    if strength <= 0.0:
        return image

    clipped = np.clip(image, 0.0, 1.0)
    noisy = rng.poisson(clipped * peak_photons).astype(np.float32) / peak_photons
    return np.clip((1.0 - strength) * clipped + strength * noisy, 0.0, 1.0)


def degrade_image(
    image: np.ndarray,
    config: LowLightConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply the configured sequence of low-light degradations to one image."""
    validate_config(config)
    generator = rng or np.random.default_rng()

    working = to_float_image(image)
    working = apply_gamma_correction(working, config.gamma)
    working = reduce_brightness(working, config.brightness_factor)
    working = reduce_contrast(working, config.contrast_factor)
    working = crush_shadows(working, config.black_level_shift)
    working = desaturate_colors(working, config.desaturation_factor)
    working = apply_motion_blur(
        working,
        kernel_size=config.motion_blur_kernel,
        angle_degrees=config.motion_blur_angle,
    )
    working = add_gaussian_noise(working, config.gaussian_sigma, generator)
    working = add_poisson_noise(working, config.poisson_strength, generator)
    return to_uint8_image(working)


def build_side_by_side_preview(
    original: np.ndarray,
    degraded: np.ndarray,
    *,
    label_height: int = 36,
) -> np.ndarray:
    """Return a labeled side-by-side preview image."""
    if original.shape[:2] != degraded.shape[:2]:
        degraded = cv2.resize(
            degraded,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    combined = cv2.hconcat([original, degraded])
    canvas = cv2.copyMakeBorder(
        combined,
        top=label_height,
        bottom=0,
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )
    cv2.putText(
        canvas,
        "Original",
        (16, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Low-Light Simulation",
        (original.shape[1] + 16, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def preview_degradation(
    image_path: str | Path,
    config: LowLightConfig,
    *,
    output_path: str | Path | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Load one image, degrade it, and optionally save a side-by-side preview."""
    image_path = Path(image_path).expanduser().resolve()
    original = read_image(image_path)
    rng = np.random.default_rng(seed)
    degraded = degrade_image(original, config, rng=rng)
    preview = build_side_by_side_preview(original, degraded)

    if output_path is not None:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), preview)

    return preview


def resolve_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    """Map an input image to its output location while preserving folder structure."""
    relative_path = input_path.relative_to(input_root)
    destination = output_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def process_images(
    input_path: str | Path,
    output_dir: str | Path,
    config: LowLightConfig,
    *,
    seed: int | None = None,
) -> list[Path]:
    """Degrade one image or a folder of images and save the results to a new location."""
    validate_config(config)

    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_image_paths(input_path)
    input_root = input_path.parent if input_path.is_file() else input_path
    rng = np.random.default_rng(seed)
    saved_paths: list[Path] = []

    for index, image_path in enumerate(image_paths, start=1):
        try:
            original = read_image(image_path)
        except ValueError as exc:
            print(f"[WARN] {exc}", file=sys.stderr)
            continue

        degraded = degrade_image(original, config, rng=rng)
        output_path = resolve_output_path(image_path, input_root, output_dir)
        cv2.imwrite(str(output_path), degraded)
        saved_paths.append(output_path)
        print(f"[{index}/{len(image_paths)}] Saved {output_path}")

    return saved_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Simulate realistic low-light degradations for eye images."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input image or folder.")
    parser.add_argument("--output", type=Path, required=True, help="Output folder.")
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.2,
        help="Gamma correction value greater than 1. Default: 2.2.",
    )
    parser.add_argument(
        "--brightness-factor",
        type=float,
        default=0.75,
        help="Multiplicative brightness reduction factor in (0, 1]. Default: 0.75.",
    )
    parser.add_argument(
        "--contrast-factor",
        type=float,
        default=0.85,
        help="Multiplicative contrast reduction factor in (0, 1]. Default: 0.85.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=8.0,
        help="Gaussian noise sigma in pixel units. Default: 8.0.",
    )
    parser.add_argument(
        "--poisson-strength",
        type=float,
        default=0.35,
        help="Blend factor for Poisson noise in [0, 1]. Default: 0.35.",
    )
    parser.add_argument(
        "--motion-blur-kernel",
        type=int,
        default=0,
        help="Motion blur kernel size. Use 0 or 1 to disable. Default: 0.",
    )
    parser.add_argument(
        "--motion-blur-angle",
        type=float,
        default=0.0,
        help="Angle for motion blur in degrees. Default: 0.0.",
    )
    parser.add_argument(
        "--desaturation-factor",
        type=float,
        default=0.15,
        help="Blend factor toward grayscale in [0, 1]. Default: 0.15.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible noise.",
    )
    parser.add_argument(
        "--preview-image",
        type=Path,
        help="Optional single image path for generating a side-by-side preview.",
    )
    parser.add_argument(
        "--preview-output",
        type=Path,
        help="Optional output path for the saved preview image.",
    )
    return parser


def main() -> int:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    config = LowLightConfig(
        gamma=args.gamma,
        brightness_factor=args.brightness_factor,
        contrast_factor=args.contrast_factor,
        gaussian_sigma=args.gaussian_sigma,
        poisson_strength=args.poisson_strength,
        motion_blur_kernel=args.motion_blur_kernel,
        motion_blur_angle=args.motion_blur_angle,
        desaturation_factor=args.desaturation_factor,
    )

    try:
        saved_paths = process_images(
            input_path=args.input,
            output_dir=args.output,
            config=config,
            seed=args.seed,
        )
        print(f"Saved {len(saved_paths)} degraded image(s) to {args.output.resolve()}")

        if args.preview_image is not None:
            preview = preview_degradation(
                image_path=args.preview_image,
                config=config,
                output_path=args.preview_output,
                seed=args.seed,
            )
            if args.preview_output is None:
                print(
                    "Preview generated in memory. Provide --preview-output to save it "
                    "as an image file."
                )
            else:
                print(f"Saved preview image to {args.preview_output.resolve()}")
            _ = preview
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
