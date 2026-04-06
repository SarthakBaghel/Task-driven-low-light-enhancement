#!/usr/bin/env python3
"""Find low-light failures recovered by enhancement and save report-ready comparisons."""

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

from evaluate_enhancer_frozen_detector import load_checkpoint, resolve_device
from inference_enhancer import load_model as load_enhancer_model
from models.detector import build_detector
from utils.classifier_metrics import (
    DEFAULT_MAX_CLOSED_PREDICTION_RATE,
    DEFAULT_MIN_CLOSED_PREDICTION_RATE,
    evaluate_threshold_candidates,
    predictions_from_closed_probability,
    resolve_positive_label,
    select_best_threshold_metrics,
)
from utils.classifier_transforms import normalize_tensor_for_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract side-by-side examples where enhancement recovers low-light "
            "detection mistakes."
        )
    )
    parser.add_argument("detector_checkpoint", type=str, help="Clean-trained detector checkpoint.")
    parser.add_argument("enhancer_checkpoint", type=str, help="Trained enhancer checkpoint.")
    parser.add_argument("lowlight_root", type=str, help="Low-light validation dataset root.")
    parser.add_argument(
        "--enhanced-detector-checkpoint",
        type=str,
        help="Optional detector checkpoint fine-tuned on enhancer outputs.",
    )
    parser.add_argument(
        "--clean-root",
        type=str,
        help="Optional clean dataset root for reference images.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--threshold-candidates",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6],
        help="Thresholds evaluated for each detector branch.",
    )
    parser.add_argument(
        "--min-closed-prediction-rate",
        type=float,
        default=DEFAULT_MIN_CLOSED_PREDICTION_RATE,
        help="Guardrail lower bound when selecting thresholds.",
    )
    parser.add_argument(
        "--max-closed-prediction-rate",
        type=float,
        default=DEFAULT_MAX_CLOSED_PREDICTION_RATE,
        help="Guardrail upper bound when selecting thresholds.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of top recovered examples to render per category.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/recovered_detection_cases",
        help="Directory for CSVs and comparison figures.",
    )
    return parser.parse_args()


def set_matplotlib_cache(output_dir: Path) -> None:
    cache_dir = output_dir / ".mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def discover_samples(dataset_root: Path, class_to_idx: dict[str, int]) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for class_name, class_index in sorted(class_to_idx.items(), key=lambda item: item[1]):
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                samples.append((image_path, class_index))
    return samples


def load_detector_from_checkpoint(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    model_config = checkpoint["model_config"]
    model = build_detector(
        backbone=model_config["backbone"],
        num_classes=model_config["num_classes"],
        image_size=model_config["image_size"],
        mobilenet_trainable_blocks=model_config.get("mobilenet_trainable_blocks", 3),
        resnet_trainable_layers=model_config.get("resnet_trainable_layers", 1),
        use_pretrained=model_config.get("use_pretrained", True),
        allow_pretrained_fallback=True,
        print_summary=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_raw_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


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


def maybe_clean_reference_path(
    lowlight_path: Path,
    *,
    lowlight_root: Path,
    clean_root: Path | None,
) -> Path | None:
    if clean_root is None:
        return None
    relative_path = lowlight_path.relative_to(lowlight_root)
    clean_name = relative_path.name.replace("__lowlight", "")
    candidate = clean_root / relative_path.parent / clean_name
    if candidate.is_file():
        return candidate
    return None


def load_pil_image(path: Path | None) -> Image.Image | None:
    if path is None:
        return None
    with Image.open(path) as image:
        return image.convert("RGB")


def tensor_to_numpy_image(image_tensor: torch.Tensor) -> np.ndarray:
    tensor = image_tensor.detach().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    array = tensor.permute(1, 2, 0).numpy()
    return np.clip(array, 0.0, 1.0)


def true_class_probability(closed_probability: float, label: int, positive_label: int) -> float:
    if label == positive_label:
        return closed_probability
    return 1.0 - closed_probability


def mean_brightness(image: np.ndarray) -> float:
    return float(image.mean())


def render_case_figure(
    *,
    record: dict[str, object],
    output_path: Path,
    clean_image: Image.Image | None,
    lowlight_image: Image.Image,
    enhanced_image: np.ndarray,
    include_finetuned: bool,
) -> None:
    figure = plt.figure(figsize=(14, 4.8))
    grid = figure.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.2])
    axes = [figure.add_subplot(grid[0, index]) for index in range(4)]

    image_triplet = [
        ("Clean Reference", np.asarray(clean_image) / 255.0 if clean_image is not None else None),
        ("Low-light Input", np.asarray(lowlight_image) / 255.0),
        ("Enhanced Output", enhanced_image),
    ]
    for axis, (title, image_array) in zip(axes[:3], image_triplet):
        axis.axis("off")
        axis.set_title(title, fontsize=11)
        if image_array is None:
            axis.text(0.5, 0.5, "No clean\nreference", ha="center", va="center", fontsize=11)
        else:
            axis.imshow(image_array)

    axes[3].axis("off")
    lines = [
        f"GT label: {record['label_name']}",
        f"File: {Path(str(record['lowlight_path'])).name}",
        "",
        f"Raw low-light: {record['raw_prediction_name']} "
        f"(closed={100.0 * float(record['raw_closed_probability']):.1f}%, "
        f"true={100.0 * float(record['raw_true_probability']):.1f}%)",
        f"Enh+Orig: {record['enhanced_prediction_name']} "
        f"(closed={100.0 * float(record['enhanced_closed_probability']):.1f}%, "
        f"true={100.0 * float(record['enhanced_true_probability']):.1f}%)",
    ]
    if include_finetuned:
        lines.append(
            f"Enh+FT: {record['finetuned_prediction_name']} "
            f"(closed={100.0 * float(record['finetuned_closed_probability']):.1f}%, "
            f"true={100.0 * float(record['finetuned_true_probability']):.1f}%)"
        )
    lines.extend(
        [
            "",
            f"Thresholds: raw={float(record['raw_threshold']):.2f}, "
            f"enh={float(record['enhanced_threshold']):.2f}"
            + (
                f", enh+ft={float(record['finetuned_threshold']):.2f}"
                if include_finetuned and record.get("finetuned_threshold") is not None
                else ""
            ),
            f"Brightness: low-light={100.0 * float(record['lowlight_brightness']):.1f}%, "
            f"enhanced={100.0 * float(record['enhanced_brightness']):.1f}%",
            f"Recovery score: {float(record['recovery_score']):+.4f}",
        ]
    )
    text = "\n".join(
        textwrap.fill(line, width=44, break_long_words=False, break_on_hyphens=False)
        if line
        else ""
        for line in lines
    )
    axes[3].text(
        0.0,
        1.0,
        text,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    figure.suptitle(record["case_title"], fontsize=13)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def render_contact_sheet(
    *,
    records: list[dict[str, object]],
    output_path: Path,
    lowlight_images: dict[str, Image.Image],
    enhanced_images: dict[str, np.ndarray],
    columns: int = 2,
) -> None:
    if not records:
        return
    rows = len(records)
    figure, axes = plt.subplots(rows, columns, figsize=(8.5, 3.6 * rows))
    axes = np.atleast_2d(axes)
    for row_index, record in enumerate(records):
        key = str(record["lowlight_path"])
        axes[row_index, 0].imshow(np.asarray(lowlight_images[key]) / 255.0)
        axes[row_index, 0].axis("off")
        axes[row_index, 0].set_title(
            f"Low-light\nGT={record['label_name']} | Raw={record['raw_prediction_name']}",
            fontsize=10,
        )
        axes[row_index, 1].imshow(enhanced_images[key])
        axes[row_index, 1].axis("off")
        second_title = record["enhanced_prediction_name"]
        if record.get("finetuned_prediction_name"):
            second_title += f" | FT={record['finetuned_prediction_name']}"
        axes[row_index, 1].set_title(
            f"Enhanced\n{second_title}",
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
    set_matplotlib_cache(output_dir)

    device = resolve_device(args.device)
    detector_checkpoint = load_checkpoint(args.detector_checkpoint)
    enhancer_checkpoint = Path(args.enhancer_checkpoint).expanduser().resolve()
    lowlight_root = Path(args.lowlight_root).expanduser().resolve()
    clean_root = Path(args.clean_root).expanduser().resolve() if args.clean_root else None

    class_to_idx = detector_checkpoint["class_to_idx"]
    positive_label = resolve_positive_label(class_to_idx)
    image_size = int(detector_checkpoint["model_config"]["image_size"])

    original_detector = load_detector_from_checkpoint(detector_checkpoint, device=device)
    enhancer = load_enhancer_model(
        checkpoint_path=enhancer_checkpoint,
        device=device,
    )
    enhanced_detector = None
    if args.enhanced_detector_checkpoint:
        enhanced_detector = load_detector_from_checkpoint(
            load_checkpoint(args.enhanced_detector_checkpoint),
            device=device,
        )

    raw_transform = build_raw_transform(image_size)
    samples = discover_samples(lowlight_root, class_to_idx)
    if not samples:
        raise ValueError(f"No images found under {lowlight_root}")

    raw_closed_probabilities: list[float] = []
    enhanced_closed_probabilities: list[float] = []
    finetuned_closed_probabilities: list[float] = []
    labels: list[int] = []
    sample_records: list[dict[str, object]] = []
    lowlight_images: dict[str, Image.Image] = {}
    enhanced_images: dict[str, np.ndarray] = {}

    progress = tqdm(samples, desc="Analyzing recovery cases", dynamic_ncols=True)
    with torch.no_grad():
        for image_path, label in progress:
            lowlight_image = load_pil_image(image_path)
            if lowlight_image is None:
                continue
            lowlight_tensor = raw_transform(lowlight_image).unsqueeze(0).to(device)
            normalized_lowlight = normalize_tensor_for_detector(lowlight_tensor)

            raw_logits = original_detector(normalized_lowlight)
            raw_closed_probability = get_closed_probability(raw_logits, positive_label)

            enhanced_tensor, _curve_maps = enhancer(lowlight_tensor)
            enhanced_numpy = tensor_to_numpy_image(enhanced_tensor)
            normalized_enhanced = normalize_tensor_for_detector(enhanced_tensor)

            enhanced_logits = original_detector(normalized_enhanced)
            enhanced_closed_probability = get_closed_probability(enhanced_logits, positive_label)

            finetuned_closed_probability = None
            if enhanced_detector is not None:
                finetuned_logits = enhanced_detector(normalized_enhanced)
                finetuned_closed_probability = get_closed_probability(finetuned_logits, positive_label)

            labels.append(int(label))
            raw_closed_probabilities.append(raw_closed_probability)
            enhanced_closed_probabilities.append(enhanced_closed_probability)
            if finetuned_closed_probability is not None:
                finetuned_closed_probabilities.append(finetuned_closed_probability)

            key = str(image_path.resolve())
            lowlight_images[key] = lowlight_image
            enhanced_images[key] = enhanced_numpy
            sample_records.append(
                {
                    "lowlight_path": str(image_path.resolve()),
                    "clean_path": str(maybe_clean_reference_path(image_path, lowlight_root=lowlight_root, clean_root=clean_root) or ""),
                    "label_index": int(label),
                    "label_name": class_name_from_index(class_to_idx, label),
                    "raw_closed_probability": raw_closed_probability,
                    "enhanced_closed_probability": enhanced_closed_probability,
                    "finetuned_closed_probability": finetuned_closed_probability,
                    "lowlight_brightness": mean_brightness(np.asarray(lowlight_image, dtype=np.float32) / 255.0),
                    "enhanced_brightness": mean_brightness(enhanced_numpy),
                }
            )

    thresholds = [float(value) for value in args.threshold_candidates]
    labels_array = np.asarray(labels, dtype=np.int64)
    raw_threshold_metrics = evaluate_threshold_candidates(
        targets=labels_array,
        closed_probabilities=np.asarray(raw_closed_probabilities, dtype=np.float32),
        positive_label=positive_label,
        thresholds=thresholds,
    )
    raw_best = select_best_threshold_metrics(
        raw_threshold_metrics,
        objective="f1",
        min_positive_rate=float(args.min_closed_prediction_rate),
        max_positive_rate=float(args.max_closed_prediction_rate),
    )
    enhanced_threshold_metrics = evaluate_threshold_candidates(
        targets=labels_array,
        closed_probabilities=np.asarray(enhanced_closed_probabilities, dtype=np.float32),
        positive_label=positive_label,
        thresholds=thresholds,
    )
    enhanced_best = select_best_threshold_metrics(
        enhanced_threshold_metrics,
        objective="f1",
        min_positive_rate=float(args.min_closed_prediction_rate),
        max_positive_rate=float(args.max_closed_prediction_rate),
    )
    finetuned_best = None
    if finetuned_closed_probabilities:
        finetuned_threshold_metrics = evaluate_threshold_candidates(
            targets=labels_array,
            closed_probabilities=np.asarray(finetuned_closed_probabilities, dtype=np.float32),
            positive_label=positive_label,
            thresholds=thresholds,
        )
        finetuned_best = select_best_threshold_metrics(
            finetuned_threshold_metrics,
            objective="f1",
            min_positive_rate=float(args.min_closed_prediction_rate),
            max_positive_rate=float(args.max_closed_prediction_rate),
        )

    all_rows: list[dict[str, object]] = []
    recovered_by_original: list[dict[str, object]] = []
    recovered_by_finetuned: list[dict[str, object]] = []
    recovered_by_both: list[dict[str, object]] = []

    raw_threshold = float(raw_best["threshold"])
    enhanced_threshold = float(enhanced_best["threshold"])
    finetuned_threshold = float(finetuned_best["threshold"]) if finetuned_best is not None else None

    for record in sample_records:
        label = int(record["label_index"])
        raw_prediction, raw_name = prediction_name_from_closed_probability(
            closed_probability=float(record["raw_closed_probability"]),
            threshold=raw_threshold,
            positive_label=positive_label,
            class_to_idx=class_to_idx,
        )
        enhanced_prediction, enhanced_name = prediction_name_from_closed_probability(
            closed_probability=float(record["enhanced_closed_probability"]),
            threshold=enhanced_threshold,
            positive_label=positive_label,
            class_to_idx=class_to_idx,
        )
        raw_correct = int(raw_prediction == label)
        enhanced_correct = int(enhanced_prediction == label)

        finetuned_prediction = None
        finetuned_name = None
        finetuned_correct = None
        if record["finetuned_closed_probability"] is not None and finetuned_threshold is not None:
            finetuned_prediction, finetuned_name = prediction_name_from_closed_probability(
                closed_probability=float(record["finetuned_closed_probability"]),
                threshold=finetuned_threshold,
                positive_label=positive_label,
                class_to_idx=class_to_idx,
            )
            finetuned_correct = int(finetuned_prediction == label)

        record = dict(record)
        record.update(
            {
                "raw_threshold": raw_threshold,
                "enhanced_threshold": enhanced_threshold,
                "finetuned_threshold": finetuned_threshold,
                "raw_prediction_index": raw_prediction,
                "raw_prediction_name": raw_name,
                "raw_correct": raw_correct,
                "raw_true_probability": true_class_probability(
                    float(record["raw_closed_probability"]),
                    label,
                    positive_label,
                ),
                "enhanced_prediction_index": enhanced_prediction,
                "enhanced_prediction_name": enhanced_name,
                "enhanced_correct": enhanced_correct,
                "enhanced_true_probability": true_class_probability(
                    float(record["enhanced_closed_probability"]),
                    label,
                    positive_label,
                ),
                "finetuned_prediction_index": finetuned_prediction,
                "finetuned_prediction_name": finetuned_name,
                "finetuned_correct": finetuned_correct,
                "finetuned_true_probability": (
                    true_class_probability(
                        float(record["finetuned_closed_probability"]),
                        label,
                        positive_label,
                    )
                    if record["finetuned_closed_probability"] is not None
                    else None
                ),
            }
        )
        all_rows.append(record)

        if not raw_correct and enhanced_correct:
            score = float(record["enhanced_true_probability"]) - float(record["raw_true_probability"])
            record["recovery_score"] = score
            record["case_title"] = "Recovered By Enhancement + Original Detector"
            recovered_by_original.append(record)
        if not raw_correct and finetuned_correct:
            score = float(record["finetuned_true_probability"]) - float(record["raw_true_probability"])
            finetuned_record = dict(record)
            finetuned_record["recovery_score"] = score
            finetuned_record["case_title"] = "Recovered By Enhancement + Fine-tuned Detector"
            recovered_by_finetuned.append(finetuned_record)
        if not raw_correct and enhanced_correct and finetuned_correct:
            score = max(
                float(record["enhanced_true_probability"]) - float(record["raw_true_probability"]),
                float(record["finetuned_true_probability"]) - float(record["raw_true_probability"]),
            )
            both_record = dict(record)
            both_record["recovery_score"] = score
            both_record["case_title"] = "Recovered By Both Enhancement Paths"
            recovered_by_both.append(both_record)

    recovered_by_original.sort(key=lambda row: float(row["recovery_score"]), reverse=True)
    recovered_by_finetuned.sort(key=lambda row: float(row["recovery_score"]), reverse=True)
    recovered_by_both.sort(key=lambda row: float(row["recovery_score"]), reverse=True)

    save_csv(all_rows, output_dir / "all_case_predictions.csv")
    save_csv(recovered_by_original, output_dir / "recovered_by_enhancer_original.csv")
    save_csv(recovered_by_finetuned, output_dir / "recovered_by_enhancer_finetuned.csv")
    save_csv(recovered_by_both, output_dir / "recovered_by_both.csv")

    categories = [
        ("recovered_by_enhancer_original", recovered_by_original, False),
        ("recovered_by_enhancer_finetuned", recovered_by_finetuned, True),
        ("recovered_by_both", recovered_by_both, True),
    ]
    summary_lines = [
        "Recovered Detection Analysis",
        "===========================",
        "",
        f"Low-light root: {lowlight_root}",
        f"Clean root: {clean_root if clean_root is not None else 'N/A'}",
        f"Raw threshold: {raw_threshold:.2f}",
        f"Enhanced threshold: {enhanced_threshold:.2f}",
        (
            f"Enhanced fine-tuned threshold: {finetuned_threshold:.2f}"
            if finetuned_threshold is not None
            else "Enhanced fine-tuned threshold: N/A"
        ),
        "",
        f"Total low-light samples analyzed: {len(all_rows)}",
        f"Recovered by enhancer + original detector: {len(recovered_by_original)}",
        f"Recovered by enhancer + fine-tuned detector: {len(recovered_by_finetuned)}",
        f"Recovered by both enhancement paths: {len(recovered_by_both)}",
    ]

    for folder_name, records, include_finetuned in categories:
        top_records = records[: max(args.top_k, 0)]
        detail_dir = output_dir / folder_name / "detail_cases"
        for index, record in enumerate(top_records, start=1):
            clean_path = Path(str(record["clean_path"])) if str(record["clean_path"]) else None
            clean_image = load_pil_image(clean_path) if clean_path and clean_path.is_file() else None
            lowlight_image = lowlight_images[str(record["lowlight_path"])]
            enhanced_image = enhanced_images[str(record["lowlight_path"])]
            file_stem = Path(str(record["lowlight_path"])).stem
            render_case_figure(
                record=record,
                output_path=detail_dir / f"{index:02d}_{file_stem}.png",
                clean_image=clean_image,
                lowlight_image=lowlight_image,
                enhanced_image=enhanced_image,
                include_finetuned=include_finetuned,
            )
        render_contact_sheet(
            records=top_records,
            output_path=output_dir / folder_name / f"{folder_name}_contact_sheet.png",
            lowlight_images=lowlight_images,
            enhanced_images=enhanced_images,
        )

    (output_dir / "recovery_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"Saved recovery analysis to: {output_dir}")


if __name__ == "__main__":
    main()
