#!/usr/bin/env python3
"""Verify folder structure and image integrity for an eye-state dataset."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
import tempfile

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "verify_dataset_mpl_cache"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASSES = ("open", "closed")


@dataclass
class ClassFolderReport:
    split_name: str
    class_name: str
    folder_path: Path
    valid_images: int = 0
    invalid_images: list[Path] = field(default_factory=list)
    non_image_files: list[Path] = field(default_factory=list)
    sample_files: list[Path] = field(default_factory=list)


@dataclass
class VerificationReport:
    dataset_root: Path
    layout_name: str
    class_counts: dict[str, int]
    folder_reports: list[ClassFolderReport]
    naming_issues: list[str]
    severe_imbalance: bool
    imbalance_ratio: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify eye-state dataset structure and image integrity."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory where dataset_report.txt and class_counts.png will be saved. "
            "Defaults to the dataset root."
        ),
    )
    parser.add_argument(
        "--imbalance-threshold",
        type=float,
        default=2.0,
        help=(
            "Flag severe imbalance when max(class_count) / min(class_count) is greater "
            "than or equal to this value. Default: 2.0."
        ),
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the bar chart in addition to saving it.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.dataset_root}")
    if not args.dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {args.dataset_root}")
    if args.imbalance_threshold <= 1.0:
        raise ValueError("--imbalance-threshold must be greater than 1.0.")


def contains_image_files(directory: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        for path in directory.rglob("*")
    )


def detect_layout(dataset_root: Path) -> tuple[str, dict[str, list[Path]]]:
    expected_dirs = [
        dataset_root / class_name
        for class_name in EXPECTED_CLASSES
        if (dataset_root / class_name).is_dir()
    ]
    if expected_dirs:
        class_dirs = [
            child
            for child in sorted(dataset_root.iterdir())
            if child.is_dir() and contains_image_files(child)
        ]
        return "flat", {"root": class_dirs}

    split_map: dict[str, list[Path]] = {}
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue

        class_dirs = [
            grandchild
            for grandchild in sorted(child.iterdir())
            if grandchild.is_dir() and contains_image_files(grandchild)
        ]
        if class_dirs:
            split_map[child.name] = class_dirs

    if split_map:
        return "split", split_map

    class_dirs = [
        child
        for child in sorted(dataset_root.iterdir())
        if child.is_dir() and contains_image_files(child)
    ]
    if class_dirs:
        return "flat", {"root": class_dirs}

    raise ValueError(
        f"Could not detect a valid dataset layout under: {dataset_root}"
    )


def verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def inspect_class_folder(
    class_dir: Path,
    *,
    split_name: str,
    dataset_root: Path,
) -> ClassFolderReport:
    report = ClassFolderReport(
        split_name=split_name,
        class_name=class_dir.name,
        folder_path=class_dir,
    )

    for path in sorted(class_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            report.non_image_files.append(path.relative_to(dataset_root))
            continue

        if verify_image(path):
            report.valid_images += 1
            if len(report.sample_files) < 10:
                report.sample_files.append(path.relative_to(dataset_root))
        else:
            report.invalid_images.append(path.relative_to(dataset_root))

    return report


def detect_naming_issues(
    layout: dict[str, list[Path]],
    layout_name: str,
) -> list[str]:
    issues: list[str] = []
    expected_lower = set(EXPECTED_CLASSES)

    if layout_name == "flat":
        class_names = [path.name for path in layout["root"]]
        lowered = {name.lower() for name in class_names}

        missing = [name for name in EXPECTED_CLASSES if name not in lowered]
        unexpected = [name for name in class_names if name.lower() not in expected_lower]
        case_mismatch = [
            name for name in class_names if name.lower() in expected_lower and name not in EXPECTED_CLASSES
        ]

        if missing:
            issues.append(f"Missing expected class folder(s): {', '.join(missing)}")
        if unexpected:
            issues.append(f"Unexpected class folder(s): {', '.join(unexpected)}")
        if case_mismatch:
            issues.append(
                "Class folder case mismatch detected: "
                + ", ".join(case_mismatch)
                + " (expected lowercase folder names)."
            )
        return issues

    split_names = sorted(layout)
    expected_set = set(EXPECTED_CLASSES)
    union_classes = {path.name.lower() for class_dirs in layout.values() for path in class_dirs}

    for split_name in split_names:
        class_names = [path.name for path in layout[split_name]]
        lowered = {name.lower() for name in class_names}

        missing = [name for name in EXPECTED_CLASSES if name not in lowered]
        unexpected = [name for name in class_names if name.lower() not in expected_set]
        case_mismatch = [
            name for name in class_names if name.lower() in expected_set and name not in EXPECTED_CLASSES
        ]

        if missing:
            issues.append(
                f"Split '{split_name}' is missing expected class folder(s): {', '.join(missing)}"
            )
        if unexpected:
            issues.append(
                f"Split '{split_name}' has unexpected class folder(s): {', '.join(unexpected)}"
            )
        if case_mismatch:
            issues.append(
                f"Split '{split_name}' has class folder case mismatch: "
                + ", ".join(case_mismatch)
                + " (expected lowercase folder names)."
            )

    missing_global = [name for name in EXPECTED_CLASSES if name not in union_classes]
    if missing_global:
        issues.append(
            "Dataset is missing expected class folder(s) across all splits: "
            + ", ".join(missing_global)
        )

    return issues


def aggregate_class_counts(folder_reports: list[ClassFolderReport]) -> dict[str, int]:
    counts = defaultdict(int)
    for report in folder_reports:
        counts[report.class_name.lower()] += report.valid_images
    return dict(sorted(counts.items()))


def compute_imbalance(
    class_counts: dict[str, int],
    threshold: float,
) -> tuple[bool, float | None]:
    positive_counts = [count for count in class_counts.values() if count > 0]
    if len(positive_counts) < 2:
        return False, None

    ratio = max(positive_counts) / min(positive_counts)
    return ratio >= threshold, ratio


def build_report_lines(report: VerificationReport) -> list[str]:
    lines = [
        "Eye-State Dataset Verification Report",
        "=" * 40,
        f"Dataset root: {report.dataset_root}",
        f"Detected layout: {report.layout_name}",
        "",
        "Image Counts Per Class",
        "-" * 24,
    ]

    total_images = sum(report.class_counts.values())
    for class_name, count in report.class_counts.items():
        lines.append(f"{class_name}: {count}")
    lines.append(f"total_samples: {total_images}")
    lines.append("")

    lines.append("Folder Naming Issues")
    lines.append("-" * 20)
    if report.naming_issues:
        lines.extend(report.naming_issues)
    else:
        lines.append("No folder naming issues detected.")
    lines.append("")

    lines.append("Class Imbalance Check")
    lines.append("-" * 21)
    if report.imbalance_ratio is None:
        lines.append("Not enough non-empty classes to compute imbalance ratio.")
    else:
        lines.append(f"imbalance_ratio: {report.imbalance_ratio:.3f}")
        lines.append(f"severe_imbalance: {report.severe_imbalance}")
    lines.append("")

    lines.append("Per-Folder Verification")
    lines.append("-" * 23)
    for folder_report in report.folder_reports:
        lines.append(
            f"[{folder_report.split_name}] {folder_report.class_name} -> "
            f"valid={folder_report.valid_images}, "
            f"invalid={len(folder_report.invalid_images)}, "
            f"non_image={len(folder_report.non_image_files)}"
        )
        if folder_report.invalid_images:
            lines.append("Invalid images:")
            lines.extend(str(path) for path in folder_report.invalid_images[:10])
        if folder_report.non_image_files:
            lines.append("Non-image files:")
            lines.extend(str(path) for path in folder_report.non_image_files[:10])
        lines.append("")

    lines.append("First 10 File Paths Per Class")
    lines.append("-" * 27)
    samples_by_class: dict[str, list[Path]] = defaultdict(list)
    for folder_report in report.folder_reports:
        bucket = samples_by_class[folder_report.class_name.lower()]
        for path in folder_report.sample_files:
            if len(bucket) >= 10:
                break
            bucket.append(path)

    for class_name in sorted(samples_by_class):
        lines.append(f"{class_name}:")
        for path in samples_by_class[class_name][:10]:
            lines.append(str(path))
        if not samples_by_class[class_name]:
            lines.append("(no valid files found)")
        lines.append("")

    return lines


def save_report(report_lines: list[str], report_path: Path) -> None:
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def plot_class_counts(
    class_counts: dict[str, int],
    plot_path: Path,
    *,
    show_plot: bool,
) -> None:
    class_names = list(class_counts.keys())
    counts = [class_counts[name] for name in class_names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(class_names, counts, color=["#4C78A8", "#F58518"][: len(class_names)])
    plt.title("Image Counts Per Class")
    plt.xlabel("Class")
    plt.ylabel("Valid Image Count")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close()


def verify_dataset(
    dataset_root: Path,
    imbalance_threshold: float,
) -> VerificationReport:
    layout_name, layout = detect_layout(dataset_root)
    naming_issues = detect_naming_issues(layout, layout_name)

    folder_reports: list[ClassFolderReport] = []
    for split_name, class_dirs in layout.items():
        for class_dir in class_dirs:
            folder_reports.append(
                inspect_class_folder(
                    class_dir,
                    split_name=split_name,
                    dataset_root=dataset_root,
                )
            )

    class_counts = aggregate_class_counts(folder_reports)
    severe_imbalance, imbalance_ratio = compute_imbalance(
        class_counts=class_counts,
        threshold=imbalance_threshold,
    )

    return VerificationReport(
        dataset_root=dataset_root,
        layout_name=layout_name,
        class_counts=class_counts,
        folder_reports=folder_reports,
        naming_issues=naming_issues,
        severe_imbalance=severe_imbalance,
        imbalance_ratio=imbalance_ratio,
    )


def print_console_summary(report: VerificationReport) -> None:
    print("=== Dataset Verification ===")
    print(f"Dataset root: {report.dataset_root}")
    print(f"Detected layout: {report.layout_name}")
    print(f"Counts per class: {report.class_counts}")
    total_images = sum(report.class_counts.values())
    print(f"Total valid images: {total_images}")
    if report.imbalance_ratio is not None:
        print(f"Imbalance ratio: {report.imbalance_ratio:.3f}")
        print(f"Severe imbalance: {report.severe_imbalance}")
    else:
        print("Imbalance ratio: not available")

    if report.naming_issues:
        print("Naming issues detected:")
        for issue in report.naming_issues:
            print(f"- {issue}")
    else:
        print("Naming issues detected: none")

    print("")
    samples_by_class: dict[str, list[Path]] = defaultdict(list)
    for folder_report in report.folder_reports:
        bucket = samples_by_class[folder_report.class_name.lower()]
        for path in folder_report.sample_files:
            if len(bucket) >= 10:
                break
            bucket.append(path)

    for class_name in sorted(samples_by_class):
        print(f"First files for class '{class_name}':")
        for path in samples_by_class[class_name][:10]:
            print(f"  {path}")
        if not samples_by_class[class_name]:
            print("  (no valid files found)")


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = (args.output_dir or dataset_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        report = verify_dataset(
            dataset_root=dataset_root,
            imbalance_threshold=args.imbalance_threshold,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    report_lines = build_report_lines(report)
    report_path = output_dir / "dataset_report.txt"
    plot_path = output_dir / "class_counts.png"

    save_report(report_lines, report_path)
    plot_class_counts(
        class_counts=report.class_counts,
        plot_path=plot_path,
        show_plot=args.show_plot,
    )
    print_console_summary(report)
    print(f"\nSaved report to: {report_path}")
    print(f"Saved class count plot to: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
