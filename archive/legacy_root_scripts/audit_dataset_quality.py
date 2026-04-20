#!/usr/bin/env python3
"""Audit label consistency and image quality for eye-state datasets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
import statistics
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_PATTERN = re.compile(r"frame_(\d+)", re.IGNORECASE)


@dataclass
class ImageRecord:
    path: Path
    label: str
    split: str
    source_id: str
    frame_index: int | None


@dataclass
class ImageMetrics:
    brightness: float
    contrast: float
    sharpness: float
    entropy: float
    detected_eyes: int


@dataclass
class SuspiciousSequenceEvent:
    source_id: str
    label: str
    start_frame: int
    end_frame: int
    run_length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit dataset quality for eye-state classification."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "Dataset root. Supports flat open/closed folders, split train/val folders, "
            "or the original labeled frame layout."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/dataset_audit"),
        help="Directory where the audit report and sample sheets will be written.",
    )
    parser.add_argument(
        "--isolated-run-length",
        type=int,
        default=2,
        help="Flag label runs of this length or shorter as sequence-noise candidates.",
    )
    parser.add_argument(
        "--max-flagged-samples",
        type=int,
        default=24,
        help="Maximum number of suspicious image samples to include in the contact sheets.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def infer_split_and_label(root: Path, path: Path) -> tuple[str, str]:
    relative = path.relative_to(root)
    parts = relative.parts

    if len(parts) >= 3 and parts[0] in {"train", "val", "test"}:
        return parts[0], parts[1].lower()
    if len(parts) >= 2 and parts[0].lower() in {"open", "closed"}:
        return "root", parts[0].lower()
    raise ValueError(f"Could not infer split/label from path: {path}")


def infer_source_id(path: Path, label: str, split: str) -> str:
    """Recover a sequence/source id from either original or prepared filenames."""
    parts = path.parts
    if split == "root":
        label_index = parts.index(label)
        if label_index + 1 < len(parts) - 1:
            return parts[label_index + 1]
        return "unknown"

    stem = path.stem
    match = re.match(r"^(open|closed)__([^_]+)__frame_", stem, re.IGNORECASE)
    if match:
        return match.group(2)
    return "unknown"


def infer_frame_index(path: Path) -> int | None:
    match = FRAME_PATTERN.search(path.name)
    if match:
        return int(match.group(1))
    return None


def collect_image_records(root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for path in sorted(root.rglob("*")):
        if not is_image_file(path):
            continue
        label_split = None
        try:
            split, label = infer_split_and_label(root, path)
            label_split = (split, label)
        except ValueError:
            continue
        split, label = label_split
        records.append(
            ImageRecord(
                path=path,
                label=label,
                split=split,
                source_id=infer_source_id(path, label, split),
                frame_index=infer_frame_index(path),
            )
        )
    if not records:
        raise ValueError(f"No labeled images found under: {root}")
    return records


def load_eye_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Failed to load eye cascade: {cascade_path}")
    return detector


def compute_entropy(gray_image: np.ndarray) -> float:
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).ravel()
    histogram = histogram / max(float(histogram.sum()), 1.0)
    return float(-(histogram * np.log2(histogram + 1e-12)).sum())


def detect_visible_eyes(gray_image: np.ndarray, eye_detector: cv2.CascadeClassifier) -> int:
    equalized = cv2.equalizeHist(gray_image)
    height, width = equalized.shape
    min_eye = max(8, int(round(min(height, width) * 0.08)))
    eyes = eye_detector.detectMultiScale(
        equalized,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(min_eye, min_eye),
    )
    return int(len(eyes))


def compute_image_metrics(
    path: Path,
    eye_detector: cv2.CascadeClassifier,
) -> ImageMetrics:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32) / 255.0
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return ImageMetrics(
        brightness=float(gray_float.mean()),
        contrast=float(gray_float.std()),
        sharpness=sharpness,
        entropy=compute_entropy(gray),
        detected_eyes=detect_visible_eyes(gray, eye_detector),
    )


def percentile_threshold(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), percentile))


def group_records_by_sequence(records: Iterable[ImageRecord]) -> dict[str, list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = {}
    for record in records:
        if record.frame_index is None:
            continue
        key = f"{record.split}:{record.source_id}"
        grouped.setdefault(key, []).append(record)
    for grouped_records in grouped.values():
        grouped_records.sort(key=lambda item: item.frame_index or -1)
    return grouped


def detect_isolated_label_runs(
    records: Iterable[ImageRecord],
    *,
    max_run_length: int,
) -> list[SuspiciousSequenceEvent]:
    events: list[SuspiciousSequenceEvent] = []
    for key, sequence_records in group_records_by_sequence(records).items():
        if not sequence_records:
            continue
        run_label = sequence_records[0].label
        run_start = sequence_records[0].frame_index or 0
        run_end = run_start
        run_length = 1

        for current in sequence_records[1:]:
            frame_index = current.frame_index or run_end
            if current.label == run_label and frame_index == run_end + 1:
                run_end = frame_index
                run_length += 1
                continue

            if run_length <= max_run_length:
                events.append(
                    SuspiciousSequenceEvent(
                        source_id=key,
                        label=run_label,
                        start_frame=run_start,
                        end_frame=run_end,
                        run_length=run_length,
                    )
                )

            run_label = current.label
            run_start = frame_index
            run_end = frame_index
            run_length = 1

        if run_length <= max_run_length:
            events.append(
                SuspiciousSequenceEvent(
                    source_id=key,
                    label=run_label,
                    start_frame=run_start,
                    end_frame=run_end,
                    run_length=run_length,
                )
            )
    return events


def make_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    *,
    title_prefix: str,
    thumb_size: tuple[int, int] = (180, 180),
    columns: int = 4,
) -> None:
    if not image_paths:
        return

    rows = math.ceil(len(image_paths) / columns)
    width = columns * thumb_size[0]
    height = rows * (thumb_size[1] + 26)
    canvas = Image.new("RGB", (width, height), "#f0f0f0")

    for index, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB").resize(thumb_size)
        column = index % columns
        row = index // columns
        x = column * thumb_size[0]
        y = row * (thumb_size[1] + 26)
        canvas.paste(image, (x, y + 26))

        draw = ImageDraw.Draw(canvas)
        text = f"{title_prefix}: {image_path.parent.name}/{image_path.name}"
        draw.text((x + 4, y + 6), text[:36], fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_csv(
    rows: list[dict[str, object]],
    output_path: Path,
    *,
    fieldnames: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_image_records(dataset_root)
    eye_detector = load_eye_detector()

    metric_rows: list[dict[str, object]] = []
    brightness_values: list[float] = []
    contrast_values: list[float] = []
    sharpness_values: list[float] = []

    for record in records:
        metrics = compute_image_metrics(record.path, eye_detector)
        brightness_values.append(metrics.brightness)
        contrast_values.append(metrics.contrast)
        sharpness_values.append(metrics.sharpness)
        metric_rows.append(
            {
                "path": str(record.path.relative_to(dataset_root)),
                "split": record.split,
                "label": record.label,
                "source_id": record.source_id,
                "frame_index": record.frame_index,
                "brightness": metrics.brightness,
                "contrast": metrics.contrast,
                "sharpness": metrics.sharpness,
                "entropy": metrics.entropy,
                "detected_eyes": metrics.detected_eyes,
            }
        )

    brightness_floor = percentile_threshold(brightness_values, 5)
    contrast_floor = percentile_threshold(contrast_values, 5)
    sharpness_floor = percentile_threshold(sharpness_values, 10)

    suspected_label_mismatch = [
        row
        for row in metric_rows
        if (row["label"] == "open" and int(row["detected_eyes"]) == 0)
        or (row["label"] == "closed" and int(row["detected_eyes"]) >= 2)
    ]
    low_quality = [
        row
        for row in metric_rows
        if float(row["brightness"]) <= brightness_floor
        or float(row["contrast"]) <= contrast_floor
        or float(row["sharpness"]) <= sharpness_floor
    ]

    sequence_events = detect_isolated_label_runs(
        records,
        max_run_length=args.isolated_run_length,
    )
    event_lookup = {
        (event.source_id, event.label, event.start_frame, event.end_frame)
        for event in sequence_events
    }
    sequence_suspects = [
        row
        for row in metric_rows
        if row["frame_index"] is not None
        and any(
            row["source_id"] == event.source_id.split(":")[-1]
            and row["label"] == event.label
            and event.start_frame <= int(row["frame_index"]) <= event.end_frame
            for event in sequence_events
        )
    ]

    write_csv(
        metric_rows,
        output_dir / "image_metrics.csv",
        fieldnames=[
            "path",
            "split",
            "label",
            "source_id",
            "frame_index",
            "brightness",
            "contrast",
            "sharpness",
            "entropy",
            "detected_eyes",
        ],
    )
    write_csv(
        suspected_label_mismatch,
        output_dir / "suspected_label_mismatches.csv",
        fieldnames=list(suspected_label_mismatch[0].keys()) if suspected_label_mismatch else list(metric_rows[0].keys()),
    )
    write_csv(
        low_quality,
        output_dir / "low_quality_samples.csv",
        fieldnames=list(low_quality[0].keys()) if low_quality else list(metric_rows[0].keys()),
    )
    write_csv(
        [
            {
                "source_id": event.source_id,
                "label": event.label,
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
                "run_length": event.run_length,
            }
            for event in sequence_events
        ],
        output_dir / "sequence_noise_candidates.csv",
        fieldnames=["source_id", "label", "start_frame", "end_frame", "run_length"],
    )

    make_contact_sheet(
        [dataset_root / row["path"] for row in suspected_label_mismatch[: args.max_flagged_samples]],
        output_dir / "suspected_label_mismatches.png",
        title_prefix="label",
    )
    make_contact_sheet(
        [dataset_root / row["path"] for row in low_quality[: args.max_flagged_samples]],
        output_dir / "low_quality_samples.png",
        title_prefix="quality",
    )
    make_contact_sheet(
        [dataset_root / row["path"] for row in sequence_suspects[: args.max_flagged_samples]],
        output_dir / "sequence_noise_candidates.png",
        title_prefix="sequence",
    )

    label_counts: dict[str, int] = {}
    for row in metric_rows:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

    lines = [
        "Eye-State Dataset Quality Audit",
        "=" * 36,
        f"Dataset root: {dataset_root}",
        f"Total images: {len(metric_rows)}",
        f"Label counts: {label_counts}",
        "",
        "Quality thresholds used for flagging",
        "-" * 33,
        f"Brightness <= {brightness_floor:.4f} (5th percentile)",
        f"Contrast   <= {contrast_floor:.4f} (5th percentile)",
        f"Sharpness  <= {sharpness_floor:.4f} (10th percentile)",
        "",
        "Potential data issues",
        "-" * 22,
        (
            f"Suspected label mismatches from eye-detector sanity check: "
            f"{len(suspected_label_mismatch)}"
        ),
        f"Low-quality / blurry / low-information samples: {len(low_quality)}",
        (
            f"Sequence-noise candidates (label runs <= {args.isolated_run_length} frames): "
            f"{len(sequence_events)}"
        ),
        "",
        "Interpretation hints",
        "-" * 20,
        "- Many short isolated 'closed' runs usually indicate noisy frame-level labels.",
        "- Open samples with zero detected eyes suggest bad crops, blur, occlusion, or wrong labels.",
        "- Closed samples with 2+ detected eyes are also suspicious and worth manual review.",
        "- If crops look like full upper-face regions instead of tight eye crops, the classifier signal is diluted.",
        "",
        "Artifacts written",
        "-" * 16,
        f"- {output_dir / 'image_metrics.csv'}",
        f"- {output_dir / 'suspected_label_mismatches.csv'}",
        f"- {output_dir / 'low_quality_samples.csv'}",
        f"- {output_dir / 'sequence_noise_candidates.csv'}",
        f"- {output_dir / 'suspected_label_mismatches.png'}",
        f"- {output_dir / 'low_quality_samples.png'}",
        f"- {output_dir / 'sequence_noise_candidates.png'}",
    ]
    (output_dir / "dataset_quality_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
