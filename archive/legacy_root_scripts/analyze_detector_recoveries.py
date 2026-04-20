#!/usr/bin/env python3
"""Find low-light detector mistakes recovered by an improved detector."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import textwrap

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
        description=(
            "Extract report-friendly examples where a baseline detector fails on "
            "low-light images but an improved detector gets the same image right."
        )
    )
    parser.add_argument("baseline_checkpoint", type=str, help="Baseline detector checkpoint.")
    parser.add_argument("improved_checkpoint", type=str, help="Improved detector checkpoint.")
    parser.add_argument("lowlight_root", type=str, help="Low-light dataset root.")
    parser.add_argument(
        "--clean-root",
        type=str,
        help="Optional clean dataset root used for clean reference images.",
    )
    parser.add_argument(
        "--subset-manifest",
        type=str,
        help=(
            "Optional subset_manifest.csv from evaluate_transfer_detector.py so the analysis "
            "uses the exact same sampled subset as a previous benchmark."
        ),
    )
    parser.add_argument(
        "--focus-label",
        type=str,
        default="closed",
        choices=("all", "open", "closed"),
        help="Which recovered label group to keep for top-k rendering.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of top recovered cases to render as figures.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use, for example auto, cuda, cpu, or mps.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/detector_recoveries",
        help="Directory where CSVs and figures will be written.",
    )
    return parser.parse_args()


def set_matplotlib_cache(output_dir: Path) -> None:
    cache_dir = output_dir / ".mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def build_raw_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def load_pil_image(path: Path | None) -> Image.Image | None:
    if path is None or not path.is_file():
        return None
    with Image.open(path) as image:
        return image.convert("RGB")


def tensor_to_numpy_image(image_tensor: torch.Tensor) -> np.ndarray:
    tensor = image_tensor.detach().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    return np.clip(tensor.permute(1, 2, 0).numpy(), 0.0, 1.0)


def mean_brightness(image: np.ndarray) -> float:
    return float(image.mean())


def get_closed_probability(logits: torch.Tensor, positive_label: int) -> float:
    probabilities = torch.softmax(logits, dim=1)[0]
    return float(probabilities[positive_label].item())


def class_name_from_index(class_to_idx: dict[str, int], label: int) -> str:
    inverse = {index: name for name, index in class_to_idx.items()}
    return inverse[int(label)]


def prediction_name_from_closed_probability(
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


def true_class_probability(closed_probability: float, label: int, positive_label: int) -> float:
    if int(label) == int(positive_label):
        return float(closed_probability)
    return float(1.0 - closed_probability)


def slugify_filename(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
    cleaned = cleaned.strip("_")
    return cleaned or "sample"


def discover_samples_from_root(
    lowlight_root: Path,
    *,
    class_to_idx: dict[str, int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_name, class_index in sorted(class_to_idx.items(), key=lambda item: item[1]):
        class_dir = lowlight_root / class_name
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                rows.append(
                    {
                        "class_name": class_name,
                        "label": int(class_index),
                        "relative_path": image_path.relative_to(class_dir).as_posix(),
                    }
                )
    return rows


def load_manifest_rows(
    *,
    lowlight_root: Path,
    class_to_idx: dict[str, int],
    subset_manifest: Path | None,
) -> list[dict[str, object]]:
    if subset_manifest is None:
        return discover_samples_from_root(lowlight_root, class_to_idx=class_to_idx)

    rows: list[dict[str, object]] = []
    with subset_manifest.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            class_name = str(row["class_name"])
            relative_path = str(row["relative_path"])
            label = int(row.get("label", class_to_idx[class_name]))
            image_path = lowlight_root / class_name / relative_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing low-light sample from manifest: {image_path}")
            rows.append(
                {
                    "class_name": class_name,
                    "label": label,
                    "relative_path": relative_path,
                }
            )
    return rows


def maybe_clean_reference_path(
    *,
    clean_root: Path | None,
    class_name: str,
    relative_path: str,
) -> Path | None:
    if clean_root is None:
        return None
    candidate = clean_root / class_name / relative_path
    if candidate.is_file():
        return candidate
    return None


def render_case_figure(
    *,
    record: dict[str, object],
    output_path: Path,
    clean_image: Image.Image | None,
    lowlight_image: Image.Image,
) -> None:
    figure = plt.figure(figsize=(13.5, 4.8))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25])
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]

    image_triplet = [
        ("Clean Reference", np.asarray(clean_image) / 255.0 if clean_image is not None else None),
        ("Low-light Input", np.asarray(lowlight_image) / 255.0),
    ]
    for axis, (title, image_array) in zip(axes[:2], image_triplet):
        axis.axis("off")
        axis.set_title(title, fontsize=11)
        if image_array is None:
            axis.text(0.5, 0.5, "No clean\nreference", ha="center", va="center", fontsize=11)
        else:
            axis.imshow(image_array)

    axes[2].axis("off")
    lines = [
        f"GT label: {record['label_name']}",
        f"File: {Path(str(record['lowlight_path'])).name}",
        "",
        f"Baseline: {record['baseline_prediction_name']}",
        (
            f"  closed={100.0 * float(record['baseline_closed_probability']):.1f}% | "
            f"true={100.0 * float(record['baseline_true_probability']):.1f}%"
        ),
        f"Improved: {record['improved_prediction_name']}",
        (
            f"  closed={100.0 * float(record['improved_closed_probability']):.1f}% | "
            f"true={100.0 * float(record['improved_true_probability']):.1f}%"
        ),
        "",
        (
            f"Thresholds: baseline={float(record['baseline_threshold']):.2f}, "
            f"improved={float(record['improved_threshold']):.2f}"
        ),
        f"Brightness: {100.0 * float(record['lowlight_brightness']):.1f}%",
        f"Recovery score: {float(record['recovery_score']):+.4f}",
    ]
    text = "\n".join(
        textwrap.fill(line, width=46, break_long_words=False, break_on_hyphens=False)
        if line
        else ""
        for line in lines
    )
    axes[2].text(
        0.0,
        1.0,
        text,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    figure.suptitle("Recovered Low-light Detection Case", fontsize=13)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def render_contact_sheet(
    *,
    records: list[dict[str, object]],
    output_path: Path,
) -> None:
    if not records:
        return
    rows = len(records)
    figure, axes = plt.subplots(rows, 2, figsize=(8.8, 3.6 * rows))
    axes = np.atleast_2d(axes)
    for row_index, record in enumerate(records):
        clean_image = load_pil_image(Path(str(record["clean_path"]))) if record.get("clean_path") else None
        lowlight_image = load_pil_image(Path(str(record["lowlight_path"])))

        axes[row_index, 0].axis("off")
        axes[row_index, 1].axis("off")

        if clean_image is None:
            axes[row_index, 0].text(0.5, 0.5, "No clean\nreference", ha="center", va="center", fontsize=10)
        else:
            axes[row_index, 0].imshow(np.asarray(clean_image) / 255.0)
        axes[row_index, 0].set_title(f"Clean | GT={record['label_name']}", fontsize=10)

        if lowlight_image is not None:
            axes[row_index, 1].imshow(np.asarray(lowlight_image) / 255.0)
        axes[row_index, 1].set_title(
            (
                f"Low-light | Base={record['baseline_prediction_name']} | "
                f"Improved={record['improved_prediction_name']}"
            ),
            fontsize=10,
        )
    figure.suptitle(output_path.stem.replace("_", " ").title(), fontsize=13)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def save_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_matplotlib_cache(output_dir)

    device = resolve_device(args.device)
    lowlight_root = Path(args.lowlight_root).expanduser().resolve()
    clean_root = Path(args.clean_root).expanduser().resolve() if args.clean_root else None
    subset_manifest = Path(args.subset_manifest).expanduser().resolve() if args.subset_manifest else None

    baseline_checkpoint = load_checkpoint(args.baseline_checkpoint)
    improved_checkpoint = load_checkpoint(args.improved_checkpoint)

    class_to_idx = dict(baseline_checkpoint["class_to_idx"])
    if dict(improved_checkpoint["class_to_idx"]) != class_to_idx:
        raise ValueError("Baseline and improved checkpoints use different class mappings.")

    baseline_image_size = int(baseline_checkpoint["model_config"]["image_size"])
    improved_image_size = int(improved_checkpoint["model_config"]["image_size"])
    if baseline_image_size != improved_image_size:
        raise ValueError(
            "This recovery script expects both checkpoints to use the same image size. "
            f"Received {baseline_image_size} and {improved_image_size}."
        )

    positive_label = resolve_positive_label(class_to_idx)
    baseline_threshold = float(baseline_checkpoint.get("best_threshold", 0.5))
    improved_threshold = float(improved_checkpoint.get("best_threshold", 0.5))

    baseline_model = build_detector_from_checkpoint(
        baseline_checkpoint,
        device=device,
        freeze_detector_weights=False,
        print_frozen_layers=False,
    )
    improved_model = build_detector_from_checkpoint(
        improved_checkpoint,
        device=device,
        freeze_detector_weights=False,
        print_frozen_layers=False,
    )

    raw_transform = build_raw_transform(baseline_image_size)
    manifest_rows = load_manifest_rows(
        lowlight_root=lowlight_root,
        class_to_idx=class_to_idx,
        subset_manifest=subset_manifest,
    )

    recovered_rows: list[dict[str, object]] = []
    progress = tqdm(manifest_rows, desc="Analyzing detector recoveries", dynamic_ncols=True)
    with torch.no_grad():
        for row in progress:
            class_name = str(row["class_name"])
            label = int(row["label"])
            relative_path = str(row["relative_path"])

            lowlight_path = lowlight_root / class_name / relative_path
            lowlight_image = load_pil_image(lowlight_path)
            if lowlight_image is None:
                continue

            lowlight_tensor = raw_transform(lowlight_image).unsqueeze(0).to(device)
            detector_input = normalize_tensor_for_detector(lowlight_tensor)

            baseline_logits = baseline_model(detector_input)
            improved_logits = improved_model(detector_input)

            baseline_closed_probability = get_closed_probability(baseline_logits, positive_label)
            improved_closed_probability = get_closed_probability(improved_logits, positive_label)

            baseline_prediction, baseline_prediction_name = prediction_name_from_closed_probability(
                closed_probability=baseline_closed_probability,
                threshold=baseline_threshold,
                positive_label=positive_label,
                class_to_idx=class_to_idx,
            )
            improved_prediction, improved_prediction_name = prediction_name_from_closed_probability(
                closed_probability=improved_closed_probability,
                threshold=improved_threshold,
                positive_label=positive_label,
                class_to_idx=class_to_idx,
            )

            if baseline_prediction == label or improved_prediction != label:
                continue

            label_name = class_name_from_index(class_to_idx, label)
            if args.focus_label != "all" and label_name != args.focus_label:
                continue

            clean_path = maybe_clean_reference_path(
                clean_root=clean_root,
                class_name=class_name,
                relative_path=relative_path,
            )
            lowlight_numpy = tensor_to_numpy_image(lowlight_tensor)

            baseline_true_probability = true_class_probability(
                baseline_closed_probability,
                label,
                positive_label,
            )
            improved_true_probability = true_class_probability(
                improved_closed_probability,
                label,
                positive_label,
            )

            recovered_rows.append(
                {
                    "label_name": label_name,
                    "label_index": label,
                    "lowlight_path": str(lowlight_path.resolve()),
                    "clean_path": str(clean_path.resolve()) if clean_path is not None else "",
                    "relative_path": relative_path,
                    "baseline_prediction_name": baseline_prediction_name,
                    "baseline_closed_probability": baseline_closed_probability,
                    "baseline_true_probability": baseline_true_probability,
                    "baseline_threshold": baseline_threshold,
                    "improved_prediction_name": improved_prediction_name,
                    "improved_closed_probability": improved_closed_probability,
                    "improved_true_probability": improved_true_probability,
                    "improved_threshold": improved_threshold,
                    "lowlight_brightness": mean_brightness(lowlight_numpy),
                    "recovery_score": improved_true_probability - baseline_true_probability,
                }
            )

    recovered_rows.sort(key=lambda item: float(item["recovery_score"]), reverse=True)
    top_rows = recovered_rows[: max(args.top_k, 0)]

    save_csv(recovered_rows, output_dir / "all_recovered_cases.csv")
    save_csv(top_rows, output_dir / "top_recovered_cases.csv")

    cases_dir = output_dir / "top_case_figures"
    for index, record in enumerate(top_rows, start=1):
        lowlight_path = Path(str(record["lowlight_path"]))
        clean_image = load_pil_image(Path(str(record["clean_path"]))) if record.get("clean_path") else None
        lowlight_image = load_pil_image(lowlight_path)
        if lowlight_image is None:
            continue
        output_name = (
            f"{index:02d}__{record['label_name']}__{slugify_filename(lowlight_path.stem)}.png"
        )
        render_case_figure(
            record=record,
            output_path=cases_dir / output_name,
            clean_image=clean_image,
            lowlight_image=lowlight_image,
        )

    if top_rows:
        render_contact_sheet(records=top_rows, output_path=output_dir / "top_recovered_contact_sheet.png")

    summary_lines = [
        "Detector Recovery Analysis",
        f"Device: {device}",
        f"Low-light root: {lowlight_root}",
        f"Clean root: {clean_root if clean_root is not None else 'None'}",
        f"Subset manifest: {subset_manifest if subset_manifest is not None else 'None'}",
        f"Focus label: {args.focus_label}",
        f"Baseline threshold: {baseline_threshold:.2f}",
        f"Improved threshold: {improved_threshold:.2f}",
        f"Recovered cases found: {len(recovered_rows)}",
        f"Rendered top cases: {len(top_rows)}",
        f"All recovered CSV: {output_dir / 'all_recovered_cases.csv'}",
        f"Top recovered CSV: {output_dir / 'top_recovered_cases.csv'}",
        f"Top contact sheet: {output_dir / 'top_recovered_contact_sheet.png'}",
    ]
    (output_dir / "recovery_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
