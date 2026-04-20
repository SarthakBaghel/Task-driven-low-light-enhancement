#!/usr/bin/env python3
"""Simple demo for comparing clean and mixed eye-state detectors."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import textwrap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm.auto import tqdm

from evaluate_enhancer_frozen_detector import (
    build_detector_from_checkpoint,
    load_checkpoint,
    resolve_device,
)
from utils.classifier_metrics import predictions_from_closed_probability, resolve_positive_label
from utils.classifier_transforms import normalize_tensor_for_detector


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple clean-detector vs mixed-detector demo on eye images."
    )
    parser.add_argument("--clean-checkpoint", required=True, help="Path to clean detector checkpoint.")
    parser.add_argument("--mixed-checkpoint", required=True, help="Path to mixed detector checkpoint.")
    parser.add_argument("--image-dir", required=True, help="Folder containing demo images.")
    parser.add_argument("--output-dir", default="demo_outputs", help="Where demo outputs are saved.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, or mps.")
    parser.add_argument("--max-images", type=int, default=24, help="Maximum images to show in contact sheet.")
    return parser.parse_args()


def discover_images(image_dir: Path) -> list[Path]:
    paths = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]
    if not paths:
        raise ValueError(f"No demo images found under: {image_dir}")
    return paths


def build_raw_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def class_name_from_index(class_to_idx: dict[str, int], label: int) -> str:
    inverse = {index: name for name, index in class_to_idx.items()}
    return inverse[int(label)]


def infer_label_from_parent(path: Path, class_to_idx: dict[str, int]) -> tuple[int | None, str]:
    parent_name = path.parent.name
    if parent_name in class_to_idx:
        return int(class_to_idx[parent_name]), parent_name
    return None, "unknown"


def get_closed_probability(logits: torch.Tensor, positive_label: int) -> float:
    probabilities = torch.softmax(logits, dim=1)[0]
    return float(probabilities[positive_label].item())


def predict_from_closed_probability(
    *,
    closed_probability: float,
    threshold: float,
    positive_label: int,
    class_to_idx: dict[str, int],
) -> tuple[int, str]:
    prediction = int(
        predictions_from_closed_probability(
            np.array([closed_probability], dtype=np.float32),
            threshold=threshold,
            positive_label=positive_label,
        )[0]
    )
    return prediction, class_name_from_index(class_to_idx, prediction)


def save_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_contact_sheet(rows: list[dict[str, object]], output_path: Path, max_images: int) -> None:
    selected = rows[:max_images]
    if not selected:
        return

    columns = 3
    rows_count = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows_count, columns, figsize=(4.8 * columns, 4.3 * rows_count))
    axes = np.asarray(axes).reshape(rows_count, columns)

    for axis in axes.reshape(-1):
        axis.axis("off")

    for index, row in enumerate(selected):
        axis = axes[index // columns, index % columns]
        image = load_rgb(Path(str(row["image_path"])))
        axis.imshow(np.asarray(image) / 255.0)
        axis.axis("off")

        title = (
            f"GT: {row['label_name']}\n"
            f"Clean: {row['clean_prediction_name']} ({100.0 * float(row['clean_closed_probability']):.1f}% closed)\n"
            f"Mixed: {row['mixed_prediction_name']} ({100.0 * float(row['mixed_closed_probability']):.1f}% closed)"
        )
        if row.get("recovered_by_mixed") != "":
            title += f"\nRecovered: {row['recovered_by_mixed']}"
        axis.set_title(title, fontsize=9)

    figure.suptitle("Eye-State Detector Demo: Clean vs Mixed Detector", fontsize=14)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    clean_checkpoint = load_checkpoint(args.clean_checkpoint)
    mixed_checkpoint = load_checkpoint(args.mixed_checkpoint)

    class_to_idx = dict(clean_checkpoint["class_to_idx"])
    if dict(mixed_checkpoint["class_to_idx"]) != class_to_idx:
        raise ValueError("Clean and mixed checkpoints use different class mappings.")

    clean_image_size = int(clean_checkpoint["model_config"]["image_size"])
    mixed_image_size = int(mixed_checkpoint["model_config"]["image_size"])
    if clean_image_size != mixed_image_size:
        raise ValueError("Clean and mixed checkpoints must use the same image size for this demo.")

    positive_label = resolve_positive_label(class_to_idx)
    clean_threshold = float(clean_checkpoint.get("best_threshold", 0.5))
    mixed_threshold = float(mixed_checkpoint.get("best_threshold", 0.5))

    clean_model = build_detector_from_checkpoint(
        clean_checkpoint,
        device=device,
        freeze_detector_weights=False,
        print_frozen_layers=False,
    )
    mixed_model = build_detector_from_checkpoint(
        mixed_checkpoint,
        device=device,
        freeze_detector_weights=False,
        print_frozen_layers=False,
    )

    image_paths = discover_images(Path(args.image_dir).expanduser().resolve())
    transform = build_raw_transform(clean_image_size)

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Running demo predictions", dynamic_ncols=True):
            image = load_rgb(image_path)
            tensor = transform(image).unsqueeze(0).to(device)
            detector_input = normalize_tensor_for_detector(tensor)

            clean_closed_probability = get_closed_probability(clean_model(detector_input), positive_label)
            mixed_closed_probability = get_closed_probability(mixed_model(detector_input), positive_label)

            clean_prediction, clean_prediction_name = predict_from_closed_probability(
                closed_probability=clean_closed_probability,
                threshold=clean_threshold,
                positive_label=positive_label,
                class_to_idx=class_to_idx,
            )
            mixed_prediction, mixed_prediction_name = predict_from_closed_probability(
                closed_probability=mixed_closed_probability,
                threshold=mixed_threshold,
                positive_label=positive_label,
                class_to_idx=class_to_idx,
            )

            label, label_name = infer_label_from_parent(image_path, class_to_idx)
            clean_correct = "" if label is None else str(clean_prediction == label)
            mixed_correct = "" if label is None else str(mixed_prediction == label)
            recovered_by_mixed = ""
            if label is not None:
                recovered_by_mixed = str(clean_prediction != label and mixed_prediction == label)

            rows.append(
                {
                    "image_path": str(image_path),
                    "label_name": label_name,
                    "clean_prediction_name": clean_prediction_name,
                    "clean_closed_probability": clean_closed_probability,
                    "clean_threshold": clean_threshold,
                    "clean_correct": clean_correct,
                    "mixed_prediction_name": mixed_prediction_name,
                    "mixed_closed_probability": mixed_closed_probability,
                    "mixed_threshold": mixed_threshold,
                    "mixed_correct": mixed_correct,
                    "recovered_by_mixed": recovered_by_mixed,
                }
            )

    predictions_csv = output_dir / "predictions.csv"
    contact_sheet = output_dir / "demo_contact_sheet.png"
    summary_path = output_dir / "demo_summary.txt"

    save_csv(rows, predictions_csv)
    render_contact_sheet(rows, contact_sheet, max_images=args.max_images)

    labeled_rows = [row for row in rows if row["label_name"] != "unknown"]
    recovered_count = sum(row["recovered_by_mixed"] == "True" for row in labeled_rows)
    clean_correct_count = sum(row["clean_correct"] == "True" for row in labeled_rows)
    mixed_correct_count = sum(row["mixed_correct"] == "True" for row in labeled_rows)

    summary = [
        "Eye-State Detector Demo Summary",
        f"Device: {device}",
        f"Images evaluated: {len(rows)}",
        f"Labeled images: {len(labeled_rows)}",
        f"Clean detector correct: {clean_correct_count}/{len(labeled_rows)}" if labeled_rows else "Clean detector correct: labels unavailable",
        f"Mixed detector correct: {mixed_correct_count}/{len(labeled_rows)}" if labeled_rows else "Mixed detector correct: labels unavailable",
        f"Recovered by mixed detector: {recovered_count}" if labeled_rows else "Recovered by mixed detector: labels unavailable",
        f"Predictions CSV: {predictions_csv}",
        f"Contact sheet: {contact_sheet}",
    ]
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
