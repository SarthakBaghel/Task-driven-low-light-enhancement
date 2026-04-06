#!/usr/bin/env python3
"""Extract frames from videos while preserving the input folder structure."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
import time
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".m4v"}


@dataclass
class VideoStats:
    processed: int = 0
    saved: int = 0
    skipped_sampling: int = 0
    skipped_similarity: int = 0
    skipped_detection: int = 0
    reused_face: int = 0


@dataclass
class HaarDetectors:
    face: cv2.CascadeClassifier
    eye: cv2.CascadeClassifier | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from videos in a folder tree."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory containing videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory where extracted frames will be saved.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--every-n-frames",
        type=int,
        help="Save 1 frame for every N frames read.",
    )
    group.add_argument(
        "--interval-seconds",
        type=float,
        help="Save 1 frame every N seconds of video.",
    )

    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality from 0 to 100. Default: 95.",
    )
    parser.add_argument(
        "--difference-method",
        choices=("pixel", "ssim"),
        default="pixel",
        help="Method used to compare candidate frames. Default: pixel.",
    )
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=None,
        help=(
            "Minimum change score needed to save a sampled frame. "
            "If omitted, a method-specific default is used."
        ),
    )
    parser.add_argument(
        "--diff-max-side",
        type=int,
        default=160,
        help=(
            "Resize the grayscale frame used for difference checking so that its "
            "largest side is at most this value. Default: 160."
        ),
    )
    parser.add_argument(
        "--roi-mode",
        choices=("face", "upper-face", "eyes"),
        default="upper-face",
        help=(
            "Crop mode applied after face detection. "
            "'upper-face' is the best default for blinking detection."
        ),
    )
    parser.add_argument(
        "--detect-max-side",
        type=int,
        default=640,
        help=(
            "Resize frames before Haar detection so the largest side is at most "
            "this value. Default: 640."
        ),
    )
    parser.add_argument(
        "--face-scale-factor",
        type=float,
        default=1.1,
        help="Scale factor used by the Haar face detector. Default: 1.1.",
    )
    parser.add_argument(
        "--face-min-neighbors",
        type=int,
        default=5,
        help="Minimum neighbors for the Haar face detector. Default: 5.",
    )
    parser.add_argument(
        "--face-min-size",
        type=int,
        default=80,
        help="Minimum detected face size in pixels on the original frame. Default: 80.",
    )
    parser.add_argument(
        "--eye-scale-factor",
        type=float,
        default=1.1,
        help="Scale factor used by the Haar eye detector. Default: 1.1.",
    )
    parser.add_argument(
        "--eye-min-neighbors",
        type=int,
        default=4,
        help="Minimum neighbors for the Haar eye detector. Default: 4.",
    )
    parser.add_argument(
        "--no-reuse-last-face",
        action="store_false",
        dest="reuse_last_face",
        help="Disable fallback to the last detected face when a frame misses detection.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=500,
        help="Print progress every N processed frames. Default: 500.",
    )
    parser.set_defaults(reuse_last_face=True)
    return parser.parse_args()


def iter_video_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def build_output_dir(input_dir: Path, output_dir: Path, video_path: Path) -> Path:
    relative_path = video_path.relative_to(input_dir)
    return output_dir / relative_path.with_suffix("")


def should_save_frame(
    frame_index: int,
    fps: float,
    every_n_frames: int | None,
    interval_seconds: float | None,
) -> bool:
    if every_n_frames is not None:
        return frame_index % every_n_frames == 0

    interval_frames = max(1, int(round(interval_seconds * fps))) if fps > 0 else 1
    return frame_index % interval_frames == 0


def get_change_threshold(
    difference_method: str,
    user_threshold: float | None,
) -> float:
    if user_threshold is not None:
        return user_threshold

    defaults = {
        "pixel": 0.015,
        "ssim": 0.08,
    }
    return defaults[difference_method]


def prepare_diff_frame(frame: np.ndarray, diff_max_side: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if diff_max_side > 0:
        height, width = gray.shape
        largest_side = max(height, width)
        if largest_side > diff_max_side:
            scale = diff_max_side / largest_side
            new_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)

    return gray


def compute_ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    image_a = image_a.astype(np.float32)
    image_b = image_b.astype(np.float32)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(image_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(image_b, (11, 11), 1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(image_a * image_a, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(image_b * image_b, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(image_a * image_b, (11, 11), 1.5) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)

    ssim_map = numerator / (denominator + 1e-12)
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def compute_change_score(
    previous_saved_gray: np.ndarray | None,
    current_gray: np.ndarray,
    difference_method: str,
) -> float:
    if previous_saved_gray is None:
        return 1.0

    if difference_method == "pixel":
        return float(cv2.absdiff(previous_saved_gray, current_gray).mean() / 255.0)

    return 1.0 - compute_ssim(previous_saved_gray, current_gray)


def load_haar_detectors(roi_mode: str) -> HaarDetectors:
    haar_root = Path(cv2.data.haarcascades)
    face_path = haar_root / "haarcascade_frontalface_default.xml"
    eye_path = haar_root / "haarcascade_eye_tree_eyeglasses.xml"

    face = cv2.CascadeClassifier(str(face_path))
    if face.empty():
        raise RuntimeError(f"Failed to load Haar face cascade: {face_path}")

    eye: cv2.CascadeClassifier | None = None
    if roi_mode == "eyes":
        eye = cv2.CascadeClassifier(str(eye_path))
        if eye.empty():
            raise RuntimeError(f"Failed to load Haar eye cascade: {eye_path}")

    return HaarDetectors(face=face, eye=eye)


def prepare_detection_frame(
    frame: np.ndarray,
    detect_max_side: int,
) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    height, width = gray.shape
    largest_side = max(height, width)
    scale = 1.0

    if largest_side > detect_max_side:
        scale = detect_max_side / largest_side
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)

    return gray, scale


def clamp_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    frame_height, frame_width = frame_shape[:2]

    x = max(0, x)
    y = max(0, y)
    w = min(w, frame_width - x)
    h = min(h, frame_height - y)

    if w <= 1 or h <= 1:
        return None
    return x, y, w, h


def detect_face_bbox(
    frame: np.ndarray,
    detectors: HaarDetectors,
    detect_max_side: int,
    face_scale_factor: float,
    face_min_neighbors: int,
    face_min_size: int,
) -> tuple[int, int, int, int] | None:
    detection_gray, scale = prepare_detection_frame(frame, detect_max_side)
    scaled_min_size = max(24, int(round(face_min_size * scale)))

    faces = detectors.face.detectMultiScale(
        detection_gray,
        scaleFactor=face_scale_factor,
        minNeighbors=face_min_neighbors,
        minSize=(scaled_min_size, scaled_min_size),
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda bbox: bbox[2] * bbox[3])
    if scale != 1.0:
        inv_scale = 1.0 / scale
        x = int(round(x * inv_scale))
        y = int(round(y * inv_scale))
        w = int(round(w * inv_scale))
        h = int(round(h * inv_scale))

    return clamp_bbox(x, y, w, h, frame.shape)


def crop_from_face_bbox(
    frame: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    roi_mode: str,
    detectors: HaarDetectors,
    eye_scale_factor: float,
    eye_min_neighbors: int,
) -> np.ndarray | None:
    x, y, w, h = face_bbox
    face_crop = frame[y : y + h, x : x + w]
    if face_crop.size == 0:
        return None

    if roi_mode == "face":
        return face_crop

    upper_height = max(1, int(round(h * 0.55)))
    upper_crop = frame[y : y + upper_height, x : x + w]
    if upper_crop.size == 0:
        return None

    if roi_mode == "upper-face" or detectors.eye is None:
        return upper_crop

    upper_gray = cv2.cvtColor(upper_crop, cv2.COLOR_BGR2GRAY)
    upper_gray = cv2.equalizeHist(upper_gray)
    min_eye = max(12, int(round(min(w, upper_height) * 0.12)))
    eyes = detectors.eye.detectMultiScale(
        upper_gray,
        scaleFactor=eye_scale_factor,
        minNeighbors=eye_min_neighbors,
        minSize=(min_eye, min_eye),
    )

    if len(eyes) == 0:
        return upper_crop

    ex = min(int(eye[0]) for eye in eyes)
    ey = min(int(eye[1]) for eye in eyes)
    ex2 = max(int(eye[0] + eye[2]) for eye in eyes)
    ey2 = max(int(eye[1] + eye[3]) for eye in eyes)

    pad_x = int(round((ex2 - ex) * 0.15))
    pad_y = int(round((ey2 - ey) * 0.35))

    eye_bbox = clamp_bbox(
        ex - pad_x,
        ey - pad_y,
        (ex2 - ex) + (2 * pad_x),
        (ey2 - ey) + (2 * pad_y),
        upper_crop.shape,
    )
    if eye_bbox is None:
        return upper_crop

    crop_x, crop_y, crop_w, crop_h = eye_bbox
    return upper_crop[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]


def percent(processed_frames: int, total_frames: int) -> str:
    if total_frames <= 0:
        return "unknown"
    return f"{(processed_frames / total_frames) * 100:5.1f}%"


def format_eta(start_time: float, processed_frames: int, total_frames: int) -> str:
    if total_frames <= 0 or processed_frames <= 0:
        return "ETA unknown"

    elapsed = time.time() - start_time
    rate = processed_frames / elapsed if elapsed > 0 else 0.0
    if rate <= 0:
        return "ETA unknown"

    remaining_seconds = max(0.0, (total_frames - processed_frames) / rate)
    return f"ETA {remaining_seconds:,.1f}s"


def extract_video(
    video_path: Path,
    input_dir: Path,
    output_dir: Path,
    detectors: HaarDetectors,
    every_n_frames: int | None,
    interval_seconds: float | None,
    jpg_quality: int,
    difference_method: str,
    change_threshold: float,
    diff_max_side: int,
    roi_mode: str,
    detect_max_side: int,
    face_scale_factor: float,
    face_min_neighbors: int,
    face_min_size: int,
    eye_scale_factor: float,
    eye_min_neighbors: int,
    reuse_last_face: bool,
    report_every: int,
) -> VideoStats:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return VideoStats()

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_dir = build_output_dir(input_dir, output_dir, video_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    stats = VideoStats()
    start_time = time.time()
    jpg_options = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
    previous_saved_gray: np.ndarray | None = None
    last_face_bbox: tuple[int, int, int, int] | None = None

    print(
        f"[START] {video_path.relative_to(input_dir)} | "
        f"fps={fps:.2f} | total_frames={total_frames if total_frames > 0 else 'unknown'} | "
        f"method={difference_method} | threshold={change_threshold:.4f} | "
        f"roi={roi_mode}"
    )

    while True:
        success, frame = capture.read()
        if not success:
            break

        stats.processed += 1
        sampled = should_save_frame(
            stats.processed,
            fps,
            every_n_frames,
            interval_seconds,
        )

        if not sampled:
            stats.skipped_sampling += 1
        else:
            face_bbox = detect_face_bbox(
                frame=frame,
                detectors=detectors,
                detect_max_side=detect_max_side,
                face_scale_factor=face_scale_factor,
                face_min_neighbors=face_min_neighbors,
                face_min_size=face_min_size,
            )
            if face_bbox is None and reuse_last_face and last_face_bbox is not None:
                face_bbox = last_face_bbox
                stats.reused_face += 1

            if face_bbox is None:
                stats.skipped_detection += 1
                continue

            cropped_frame = crop_from_face_bbox(
                frame=frame,
                face_bbox=face_bbox,
                roi_mode=roi_mode,
                detectors=detectors,
                eye_scale_factor=eye_scale_factor,
                eye_min_neighbors=eye_min_neighbors,
            )
            if cropped_frame is None or cropped_frame.size == 0:
                stats.skipped_detection += 1
                continue

            last_face_bbox = face_bbox
            current_gray = prepare_diff_frame(cropped_frame, diff_max_side)
            change_score = compute_change_score(
                previous_saved_gray,
                current_gray,
                difference_method,
            )

            if change_score < change_threshold:
                stats.skipped_similarity += 1
            else:
                stats.saved += 1
                frame_name = f"frame_{stats.saved:04d}.jpg"
                frame_path = target_dir / frame_name
                cv2.imwrite(str(frame_path), cropped_frame, jpg_options)
                previous_saved_gray = current_gray

        if report_every > 0 and stats.processed % report_every == 0:
            progress = percent(stats.processed, total_frames)
            eta = format_eta(start_time, stats.processed, total_frames)
            print(
                f"  [PROGRESS] {video_path.name}: "
                f"{stats.processed} frames processed, {stats.saved} saved, "
                f"{stats.skipped_sampling + stats.skipped_similarity + stats.skipped_detection} skipped "
                f"({stats.skipped_sampling} interval, {stats.skipped_similarity} similar, "
                f"{stats.skipped_detection} detection), "
                f"{progress}, {eta}"
            )

    capture.release()

    elapsed = time.time() - start_time
    print(
        f"[DONE] {video_path.relative_to(input_dir)} | "
        f"processed={stats.processed}, saved={stats.saved}, "
        f"skipped={stats.skipped_sampling + stats.skipped_similarity + stats.skipped_detection} "
        f"({stats.skipped_sampling} interval, {stats.skipped_similarity} similar, "
        f"{stats.skipped_detection} detection), "
        f"reused_face={stats.reused_face}, "
        f"elapsed={elapsed:.1f}s"
    )
    return stats


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if args.every_n_frames is not None and args.every_n_frames <= 0:
        raise ValueError("--every-n-frames must be greater than 0.")
    if args.interval_seconds is not None and args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be greater than 0.")
    if args.change_threshold is not None and not 0 <= args.change_threshold <= 1:
        raise ValueError("--change-threshold must be between 0 and 1.")
    if args.diff_max_side <= 0:
        raise ValueError("--diff-max-side must be greater than 0.")
    if args.detect_max_side <= 0:
        raise ValueError("--detect-max-side must be greater than 0.")
    if args.face_scale_factor <= 1.0:
        raise ValueError("--face-scale-factor must be greater than 1.0.")
    if args.eye_scale_factor <= 1.0:
        raise ValueError("--eye-scale-factor must be greater than 1.0.")
    if args.face_min_neighbors < 0:
        raise ValueError("--face-min-neighbors must be 0 or greater.")
    if args.eye_min_neighbors < 0:
        raise ValueError("--eye-min-neighbors must be 0 or greater.")
    if args.face_min_size <= 0:
        raise ValueError("--face-min-size must be greater than 0.")
    if not 0 <= args.jpg_quality <= 100:
        raise ValueError("--jpg-quality must be between 0 and 100.")
    if args.report_every < 0:
        raise ValueError("--report-every must be 0 or greater.")


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    videos = iter_video_files(args.input_dir)
    if not videos:
        print(f"[ERROR] No video files found under {args.input_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(videos)} video(s) in {args.input_dir}")

    change_threshold = get_change_threshold(
        args.difference_method,
        args.change_threshold,
    )
    try:
        detectors = load_haar_detectors(args.roi_mode)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    total_processed = 0
    total_saved = 0
    total_skipped_sampling = 0
    total_skipped_similarity = 0
    total_skipped_detection = 0
    total_reused_face = 0
    batch_start = time.time()

    for index, video_path in enumerate(videos, start=1):
        print(f"\n=== Video {index}/{len(videos)} ===")
        stats = extract_video(
            video_path=video_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            detectors=detectors,
            every_n_frames=args.every_n_frames,
            interval_seconds=args.interval_seconds,
            jpg_quality=args.jpg_quality,
            difference_method=args.difference_method,
            change_threshold=change_threshold,
            diff_max_side=args.diff_max_side,
            roi_mode=args.roi_mode,
            detect_max_side=args.detect_max_side,
            face_scale_factor=args.face_scale_factor,
            face_min_neighbors=args.face_min_neighbors,
            face_min_size=args.face_min_size,
            eye_scale_factor=args.eye_scale_factor,
            eye_min_neighbors=args.eye_min_neighbors,
            reuse_last_face=args.reuse_last_face,
            report_every=args.report_every,
        )
        total_processed += stats.processed
        total_saved += stats.saved
        total_skipped_sampling += stats.skipped_sampling
        total_skipped_similarity += stats.skipped_similarity
        total_skipped_detection += stats.skipped_detection
        total_reused_face += stats.reused_face

    total_elapsed = time.time() - batch_start
    print("\n=== Summary ===")
    print(f"Videos processed: {len(videos)}")
    print(f"Frames read: {total_processed}")
    print(f"Frames saved: {total_saved}")
    print(
        "Frames skipped: "
        f"{total_skipped_sampling + total_skipped_similarity + total_skipped_detection} "
        f"({total_skipped_sampling} interval, {total_skipped_similarity} similar, "
        f"{total_skipped_detection} detection)"
    )
    print(f"Frames using reused face box: {total_reused_face}")
    print(f"Total time: {total_elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
