#!/usr/bin/env python3
"""Create a cleaned copy of a labeled eye-state dataset using audit CSV outputs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import shutil

from audit_dataset_quality import ImageRecord, collect_image_records


@dataclass(frozen=True)
class RemovalEntry:
    relative_path: str
    label: str
    split: str
    source_id: str
    frame_index: int | None
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a cleaned dataset copy by removing audit-flagged images."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root labeled-image directory to clean.",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        required=True,
        help="Audit output directory containing CSV files from audit_dataset_quality.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the cleaned dataset copy.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "move"),
        default="copy",
        help="Whether to copy or move retained images into the cleaned dataset.",
    )
    parser.add_argument(
        "--keep-low-quality",
        action="store_true",
        help="Keep files flagged in low_quality_samples.csv.",
    )
    parser.add_argument(
        "--keep-sequence-noise",
        action="store_true",
        help="Keep files flagged in sequence_noise_candidates.csv.",
    )
    parser.add_argument(
        "--keep-suspected-mismatches",
        action="store_true",
        help="Keep files flagged in suspected_label_mismatches.csv.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the output directory before writing cleaned files.",
    )
    return parser.parse_args()


def read_relative_paths(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            Path(str(row["path"])).as_posix()
            for row in reader
            if row.get("path")
        }


def read_sequence_events(csv_path: Path) -> dict[str, list[tuple[int, int]]]:
    if not csv_path.exists():
        return {}
    events: dict[str, list[tuple[int, int]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_id = str(row.get("source_id", "")).strip()
            if not source_id:
                continue
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
            events.setdefault(source_id, []).append((start_frame, end_frame))
    return events


def record_key(record: ImageRecord) -> str:
    return f"{record.split}:{record.source_id}"


def is_sequence_noise(record: ImageRecord, events: dict[str, list[tuple[int, int]]]) -> bool:
    if record.frame_index is None:
        return False
    ranges = events.get(record_key(record), [])
    return any(start <= record.frame_index <= end for start, end in ranges)


def copy_or_move_file(source: Path, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(source, destination)
    else:
        shutil.move(source, destination)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_output_dir(output_dir: Path, clear_output: bool) -> None:
    if output_dir.exists() and clear_output:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    audit_dir = args.audit_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not audit_dir.is_dir():
        raise FileNotFoundError(f"Audit directory does not exist: {audit_dir}")

    ensure_output_dir(output_dir, clear_output=args.clear_output)

    low_quality_paths = read_relative_paths(audit_dir / "low_quality_samples.csv")
    mismatch_paths = read_relative_paths(audit_dir / "suspected_label_mismatches.csv")
    sequence_events = read_sequence_events(audit_dir / "sequence_noise_candidates.csv")

    records = collect_image_records(input_dir)
    removed_entries: list[RemovalEntry] = []
    kept_rows: list[dict[str, object]] = []

    kept_count = 0
    removed_count = 0

    for record in records:
        relative_path = record.path.relative_to(input_dir).as_posix()
        reasons: list[str] = []

        if not args.keep_low_quality and relative_path in low_quality_paths:
            reasons.append("low_quality")
        if not args.keep_suspected_mismatches and relative_path in mismatch_paths:
            reasons.append("suspected_label_mismatch")
        if not args.keep_sequence_noise and is_sequence_noise(record, sequence_events):
            reasons.append("sequence_noise")

        if reasons:
            removed_count += 1
            removed_entries.append(
                RemovalEntry(
                    relative_path=relative_path,
                    label=record.label,
                    split=record.split,
                    source_id=record.source_id,
                    frame_index=record.frame_index,
                    reason=";".join(sorted(set(reasons))),
                )
            )
            continue

        destination = output_dir / relative_path
        copy_or_move_file(record.path, destination, args.copy_mode)
        kept_count += 1
        kept_rows.append(
            {
                "relative_path": relative_path,
                "label": record.label,
                "split": record.split,
                "source_id": record.source_id,
                "frame_index": record.frame_index,
            }
        )

    removed_rows = [
        {
            "relative_path": entry.relative_path,
            "label": entry.label,
            "split": entry.split,
            "source_id": entry.source_id,
            "frame_index": entry.frame_index,
            "reason": entry.reason,
        }
        for entry in removed_entries
    ]

    write_csv(
        output_dir / "cleaning_kept_files.csv",
        ["relative_path", "label", "split", "source_id", "frame_index"],
        kept_rows,
    )
    write_csv(
        output_dir / "cleaning_removed_files.csv",
        ["relative_path", "label", "split", "source_id", "frame_index", "reason"],
        removed_rows,
    )

    label_counts: dict[str, int] = {}
    for row in kept_rows:
        label = str(row["label"])
        label_counts[label] = label_counts.get(label, 0) + 1

    report_lines = [
        "Cleaned Labeled Dataset Report",
        "============================",
        "",
        f"Input directory: {input_dir}",
        f"Audit directory: {audit_dir}",
        f"Output directory: {output_dir}",
        f"Copy mode: {args.copy_mode}",
        "",
        f"Total input images: {len(records)}",
        f"Retained images: {kept_count}",
        f"Removed images: {removed_count}",
        "",
        "Retained label counts:",
    ]
    for label in sorted(label_counts):
        report_lines.append(f"- {label}: {label_counts[label]}")
    report_lines.extend(
        [
            "",
            "Removal sources:",
            f"- low_quality_samples.csv: {0 if args.keep_low_quality else len(low_quality_paths)} candidate paths",
            f"- sequence_noise_candidates.csv: {0 if args.keep_sequence_noise else sum(len(value) for value in sequence_events.values())} flagged runs",
            f"- suspected_label_mismatches.csv: {0 if args.keep_suspected_mismatches else len(mismatch_paths)} candidate paths",
            "",
            f"Kept manifest: {output_dir / 'cleaning_kept_files.csv'}",
            f"Removed manifest: {output_dir / 'cleaning_removed_files.csv'}",
        ]
    )
    (output_dir / "cleaning_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Saved cleaned dataset to: {output_dir}")
    print(f"Retained images: {kept_count}")
    print(f"Removed images: {removed_count}")
    if label_counts:
        print(
            "Retained label counts: "
            + ", ".join(f"{label}={label_counts[label]}" for label in sorted(label_counts))
        )


if __name__ == "__main__":
    main()
