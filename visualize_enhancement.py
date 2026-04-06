#!/usr/bin/env python3
"""Visualize Zero-DCE enhancement results with side-by-side comparison."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

_MPL_DIR = Path(tempfile.gettempdir()) / "codex_matplotlib"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from inference_enhancer import enhance_image_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a side-by-side visualization of Zero-DCE enhancement.",
    )
    parser.add_argument("image_path", type=str, help="Path to the input low-light image.")
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/enhancement_comparison.png",
        help="Path to save the side-by-side comparison figure.",
    )
    parser.add_argument(
        "--enhanced-output-path",
        type=str,
        default="outputs/enhanced_image.png",
        help="Optional path to also save the enhanced image itself.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Optional trained model checkpoint. If omitted, random weights are used.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto, cpu, cuda, mps, or an explicit device like cuda:0.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=32,
        help="Hidden channel width used when no checkpoint config is available.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=8,
        help="Number of iterative enhancement curves to apply.",
    )
    return parser.parse_args()


def save_comparison_figure(
    input_image,
    enhanced_image,
    output_path: str | Path,
) -> Path:
    """Save a report-friendly input-vs-enhanced comparison figure."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    axes[0].imshow(input_image)
    axes[0].set_title("Input Low-Light Image", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(enhanced_image)
    axes[1].set_title("Zero-DCE Enhanced Image", fontweight="bold")
    axes[1].axis("off")

    figure.suptitle("Low-Light Enhancement Comparison", fontsize=14, fontweight="bold")

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    result = enhance_image_file(
        args.image_path,
        output_path=args.enhanced_output_path,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
        hidden_channels=args.hidden_channels,
        num_iterations=args.num_iterations,
    )
    comparison_path = save_comparison_figure(
        result["input_image"],
        result["enhanced_image"],
        args.output_path,
    )

    print(f"Input image: {Path(args.image_path).expanduser().resolve()}")
    print(f"Enhanced image: {result['saved_path']}")
    print(f"Comparison figure: {comparison_path}")
    print(f"Curve map tensor shape: {tuple(result['curve_maps'].shape)}")
    if args.checkpoint is None:
        print("No checkpoint was provided. Visualization used randomly initialized weights.")


if __name__ == "__main__":
    main()
