#!/usr/bin/env python3
"""Simple clean-detector vs mixed-detector demo for the college submission."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm.auto import tqdm

from models.detector import build_detector
from utils.classifier_metrics import predictions_from_closed_probability, resolve_positive_label
from utils.classifier_transforms import normalize_tensor_for_detector


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare clean and mixed detectors on sample images.")
    parser.add_argument("--clean-checkpoint", required=True, help="Path to clean detector checkpoint.")
    parser.add_argument("--mixed-checkpoint", required=True, help="Path to mixed detector checkpoint.")
    parser.add_argument("--image-dir", required=True, help="Folder containing demo images.")
    parser.add_argument("--output-dir", default="demo_outputs", help="Directory for demo outputs.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, or mps.")
    parser.add_argument("--max-images", type=int, default=24, help="Maximum images to show in the contact sheet.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def load_checkpoint(path: str | Path) -> dict:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def build_detector_from_checkpoint(checkpoint: dict, *, device: torch.device) -> torch.nn.Module:
    model_config = checkpoint["model_config"]
    model = build_detector(
        backbone=model_config["backbone"],
        num_classes=model_config["num_classes"],
        image_size=model_config["image_size"],
        mobilenet_trainable_blocks=model_config.get("mobilenet_trainable_blocks", 3),
        resnet_trainable_layers=model_config.get("resnet_trainable_layers", 1),
        use_pretrained=model_config.get("use_pretrained", True),
        allow_pretrained_fallback=True,
        use_dual_input=bool(model_config.get("use_dual_input", False)),
        dual_input_shared_backbone=bool(model_config.get("dual_input_shared_backbone", True)),
        print_summary=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def discover_images(image_dir: Path) -> list[Path]:
    image_paths = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]
    if not image_paths:
        raise ValueError(f"No images found under: {image_dir}")
    return image_paths


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
    row_count = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(row_count, columns, figsize=(4.8 * columns, 4.3 * row_count))
    axes = np.asarray(axes).reshape(row_count, columns)

    for axis in axes.reshape(-1):
        axis.axis("off")

    for index, row in enumerate(selected):
        axis = axes[index // columns, index % columns]
        image = load_rgb(Path(str(row["image_path"])))
        axis.imshow(np.asarray(image) / 255.0)
        axis.axis("off")
        axis.set_title(
            (
                f"GT: {row['label_name']}\n"
                f"Clean: {row['clean_prediction_name']} ({100.0 * float(row['clean_closed_probability']):.1f}% closed)\n"
                f"Mixed: {row['mixed_prediction_name']} ({100.0 * float(row['mixed_closed_probability']):.1f}% closed)"
            ),
            fontsize=9,
        )

    figure.suptitle("Eye-State Demo: Clean Detector vs Mixed Detector", fontsize=14)
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
        raise ValueError("Checkpoint class mappings do not match.")

    image_size = int(clean_checkpoint["model_config"]["image_size"])
    if image_size != int(mixed_checkpoint["model_config"]["image_size"]):
        raise ValueError("Clean and mixed checkpoints must use the same image size.")

    clean_threshold = float(clean_checkpoint.get("best_threshold", 0.5))
    mixed_threshold = float(mixed_checkpoint.get("best_threshold", 0.5))
    positive_label = resolve_positive_label(class_to_idx)

    clean_model = build_detector_from_checkpoint(clean_checkpoint, device=device)
    mixed_model = build_detector_from_checkpoint(mixed_checkpoint, device=device)

    transform = build_raw_transform(image_size)
    image_paths = discover_images(Path(args.image_dir).expanduser().resolve())

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
            recovered_by_mixed = "" if label is None else str(clean_prediction != label and mixed_prediction == label)

            rows.append(
                {
                    "image_path": str(image_path),
                    "label_name": label_name,
                    "clean_prediction_name": clean_prediction_name,
                    "clean_closed_probability": clean_closed_probability,
                    "clean_correct": clean_correct,
                    "mixed_prediction_name": mixed_prediction_name,
                    "mixed_closed_probability": mixed_closed_probability,
                    "mixed_correct": mixed_correct,
                    "recovered_by_mixed": recovered_by_mixed,
                }
            )

    predictions_csv = output_dir / "predictions.csv"
    contact_sheet = output_dir / "demo_contact_sheet.png"
    summary_txt = output_dir / "demo_summary.txt"

    save_csv(rows, predictions_csv)
    render_contact_sheet(rows, contact_sheet, max_images=args.max_images)

    labeled_rows = [row for row in rows if row["label_name"] != "unknown"]
    clean_correct_count = sum(row["clean_correct"] == "True" for row in labeled_rows)
    mixed_correct_count = sum(row["mixed_correct"] == "True" for row in labeled_rows)
    recovered_count = sum(row["recovered_by_mixed"] == "True" for row in labeled_rows)

    summary_lines = [
        "Demo Summary",
        f"Device: {device}",
        f"Images evaluated: {len(rows)}",
        f"Labeled images: {len(labeled_rows)}",
        f"Clean detector correct: {clean_correct_count}/{len(labeled_rows)}" if labeled_rows else "Clean detector correct: labels unavailable",
        f"Mixed detector correct: {mixed_correct_count}/{len(labeled_rows)}" if labeled_rows else "Mixed detector correct: labels unavailable",
        f"Recovered by mixed detector: {recovered_count}" if labeled_rows else "Recovered by mixed detector: labels unavailable",
        f"Predictions CSV: {predictions_csv}",
        f"Contact sheet: {contact_sheet}",
    ]
    summary_txt.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
