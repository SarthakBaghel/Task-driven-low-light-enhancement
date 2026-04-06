#!/usr/bin/env python3
"""Auto-label eye state using MediaPipe landmarks with a Haar fallback."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import cv2
import numpy as np


_CACHE_DIR = Path(tempfile.gettempdir()) / "label_eye_state_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# MediaPipe Face Mesh indices commonly used for EAR.
LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)


@dataclass
class LabelStats:
    processed: int = 0
    open_count: int = 0
    closed_count: int = 0
    skipped_no_face: int = 0


@dataclass
class HaarDetectors:
    face: cv2.CascadeClassifier
    eye: cv2.CascadeClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label eye state as open/closed using MediaPipe landmarks when "
            "available, with an OpenCV Haar fallback."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory containing extracted frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory where open/ and closed/ folders will be created.",
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.21,
        help="EAR threshold below which an image is labeled closed. Default: 0.21.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=640,
        help="Resize image for landmark detection so the largest side is at most this value.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum detection confidence. Default: 0.5.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum tracking confidence. Default: 0.5.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "mediapipe-solutions", "mediapipe-tasks", "haar"),
        default="auto",
        help=(
            "Landmark/detection backend. 'auto' tries MediaPipe solutions first, "
            "then MediaPipe Tasks when a model is provided, then Haar fallback."
        ),
    )
    parser.add_argument(
        "--face-landmarker-model",
        type=Path,
        help=(
            "Optional path to a MediaPipe Face Landmarker .task model used by the "
            "'mediapipe-tasks' backend."
        ),
    )
    parser.add_argument(
        "--haar-face-scale-factor",
        type=float,
        default=1.1,
        help="Scale factor for Haar face detection. Default: 1.1.",
    )
    parser.add_argument(
        "--haar-face-min-neighbors",
        type=int,
        default=5,
        help="Minimum neighbors for Haar face detection. Default: 5.",
    )
    parser.add_argument(
        "--haar-face-min-size",
        type=int,
        default=80,
        help="Minimum Haar face size in pixels. Default: 80.",
    )
    parser.add_argument(
        "--haar-eye-scale-factor",
        type=float,
        default=1.1,
        help="Scale factor for Haar eye detection. Default: 1.1.",
    )
    parser.add_argument(
        "--haar-eye-min-neighbors",
        type=int,
        default=4,
        help="Minimum neighbors for Haar eye detection. Default: 4.",
    )
    parser.add_argument(
        "--haar-min-open-eyes",
        type=int,
        default=2,
        help="Number of detected eyes required to label an image as open. Default: 2.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=500,
        help="Print progress every N images. Default: 500.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if not 0.0 < args.ear_threshold < 1.0:
        raise ValueError("--ear-threshold must be between 0 and 1.")
    if args.max_side <= 0:
        raise ValueError("--max-side must be greater than 0.")
    if not 0.0 <= args.min_detection_confidence <= 1.0:
        raise ValueError("--min-detection-confidence must be between 0 and 1.")
    if not 0.0 <= args.min_tracking_confidence <= 1.0:
        raise ValueError("--min-tracking-confidence must be between 0 and 1.")
    if args.face_landmarker_model is not None and not args.face_landmarker_model.exists():
        raise FileNotFoundError(
            f"Face landmarker model does not exist: {args.face_landmarker_model}"
        )
    if args.backend == "mediapipe-tasks" and args.face_landmarker_model is None:
        raise ValueError(
            "--face-landmarker-model is required when --backend=mediapipe-tasks."
        )
    if args.haar_face_scale_factor <= 1.0:
        raise ValueError("--haar-face-scale-factor must be greater than 1.0.")
    if args.haar_face_min_neighbors < 0:
        raise ValueError("--haar-face-min-neighbors must be 0 or greater.")
    if args.haar_face_min_size <= 0:
        raise ValueError("--haar-face-min-size must be greater than 0.")
    if args.haar_eye_scale_factor <= 1.0:
        raise ValueError("--haar-eye-scale-factor must be greater than 1.0.")
    if args.haar_eye_min_neighbors < 0:
        raise ValueError("--haar-eye-min-neighbors must be 0 or greater.")
    if args.haar_min_open_eyes <= 0:
        raise ValueError("--haar-min-open-eyes must be greater than 0.")
    if args.report_every < 0:
        raise ValueError("--report-every must be 0 or greater.")


def iter_image_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class BaseEyeStateLabeler:
    """Common interface for eye-state labelers."""

    backend_name = "unknown"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Release backend resources when needed."""

    def predict(self, image: np.ndarray) -> tuple[str | None, float | None]:
        """Return the predicted label and an optional score."""
        raise NotImplementedError


class MediaPipeSolutionsLabeler(BaseEyeStateLabeler):
    """Eye-state labeler that uses the classic MediaPipe Face Mesh API."""

    backend_name = "mediapipe-solutions"

    def __init__(self, args: argparse.Namespace, mp_module: Any) -> None:
        self.ear_threshold = args.ear_threshold
        self.face_mesh = mp_module.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )

    def close(self) -> None:
        self.face_mesh.close()

    def predict(self, image: np.ndarray) -> tuple[str | None, float | None]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_image)
        landmarks = select_mediapipe_landmarks(result.multi_face_landmarks, "landmark")
        if landmarks is None:
            return None, None

        average_ear = compute_average_ear(landmarks, image.shape)
        label = "closed" if average_ear < self.ear_threshold else "open"
        return label, average_ear


class MediaPipeTasksLabeler(BaseEyeStateLabeler):
    """Eye-state labeler that uses the newer MediaPipe Tasks Face Landmarker API."""

    backend_name = "mediapipe-tasks"

    def __init__(self, args: argparse.Namespace, mp_module: Any) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self.ear_threshold = args.ear_threshold
        self.mp_module = mp_module
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(args.face_landmarker_model.expanduser().resolve())
            ),
            num_faces=1,
            min_face_detection_confidence=args.min_detection_confidence,
            min_face_presence_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self.face_landmarker.close()

    def predict(self, image: np.ndarray) -> tuple[str | None, float | None]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self.mp_module.Image(
            image_format=self.mp_module.ImageFormat.SRGB,
            data=rgb_image,
        )
        result = self.face_landmarker.detect(mp_image)
        landmarks = select_mediapipe_landmarks(result.face_landmarks, None)
        if landmarks is None:
            return None, None

        average_ear = compute_average_ear(landmarks, image.shape)
        label = "closed" if average_ear < self.ear_threshold else "open"
        return label, average_ear


class HaarEyeStateLabeler(BaseEyeStateLabeler):
    """Fallback eye-state labeler that uses OpenCV Haar cascades."""

    backend_name = "haar"

    def __init__(self, args: argparse.Namespace) -> None:
        self.min_open_eyes = args.haar_min_open_eyes
        self.face_scale_factor = args.haar_face_scale_factor
        self.face_min_neighbors = args.haar_face_min_neighbors
        self.face_min_size = args.haar_face_min_size
        self.eye_scale_factor = args.haar_eye_scale_factor
        self.eye_min_neighbors = args.haar_eye_min_neighbors
        self.detectors = load_haar_detectors()

    def predict(self, image: np.ndarray) -> tuple[str | None, float | None]:
        face_bbox = detect_face_bbox(
            image=image,
            detectors=self.detectors,
            scale_factor=self.face_scale_factor,
            min_neighbors=self.face_min_neighbors,
            min_size=self.face_min_size,
        )
        if face_bbox is None:
            region = image
            region_width = image.shape[1]
            region_height = image.shape[0]
        else:
            x, y, w, h = face_bbox
            upper_height = max(1, int(round(h * 0.55)))
            region = image[y : y + upper_height, x : x + w]
            if region.size == 0:
                return None, None
            region_width = w
            region_height = upper_height

        upper_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        upper_gray = cv2.equalizeHist(upper_gray)
        min_eye = max(6, int(round(min(region_width, region_height) * 0.08)))
        eyes = self.detectors.eye.detectMultiScale(
            upper_gray,
            scaleFactor=self.eye_scale_factor,
            minNeighbors=self.eye_min_neighbors,
            minSize=(min_eye, min_eye),
        )

        open_score = float(len(eyes))
        label = "open" if len(eyes) >= self.min_open_eyes else "closed"
        return label, open_score


def try_import_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is not installed. Install it with 'pip install mediapipe'."
        ) from exc
    return mp


def build_labeler(args: argparse.Namespace) -> BaseEyeStateLabeler:
    """Build the requested eye-state labeling backend."""
    if args.backend == "haar":
        return HaarEyeStateLabeler(args)

    mp_module = try_import_mediapipe()
    has_solutions = hasattr(mp_module, "solutions")

    if args.backend == "mediapipe-solutions":
        if not has_solutions:
            raise RuntimeError(
                "The installed MediaPipe package does not expose 'mediapipe.solutions'. "
                "Use --backend=haar, or provide --face-landmarker-model with "
                "--backend=mediapipe-tasks."
            )
        return MediaPipeSolutionsLabeler(args, mp_module)

    if args.backend == "mediapipe-tasks":
        return MediaPipeTasksLabeler(args, mp_module)

    if has_solutions:
        return MediaPipeSolutionsLabeler(args, mp_module)
    if args.face_landmarker_model is not None:
        return MediaPipeTasksLabeler(args, mp_module)
    return HaarEyeStateLabeler(args)


def resize_for_detection(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_side:
        return image

    scale = max_side / largest_side
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def select_mediapipe_landmarks(
    faces: Any,
    landmark_attr_name: str | None,
) -> list | None:
    """Select the largest detected face and return its landmark list."""
    if not faces:
        return None

    if landmark_attr_name is None:
        all_landmarks = list(faces)
        if len(all_landmarks) == 1:
            return list(all_landmarks[0])
    else:
        all_landmarks = [getattr(face, landmark_attr_name) for face in faces]
        if len(all_landmarks) == 1:
            return list(all_landmarks[0])

    def face_area(landmarks: list) -> float:
        xs = [point.x for point in landmarks]
        ys = [point.y for point in landmarks]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    best_landmarks = max(all_landmarks, key=face_area)
    return list(best_landmarks)


def landmark_to_point(landmarks: list, index: int, shape: tuple[int, int, int]) -> np.ndarray:
    height, width = shape[:2]
    point = landmarks[index]
    return np.array([point.x * width, point.y * height], dtype=np.float32)


def compute_eye_ear(
    landmarks: list,
    indices: tuple[int, int, int, int, int, int],
    shape: tuple[int, int, int],
) -> float:
    p1 = landmark_to_point(landmarks, indices[0], shape)
    p2 = landmark_to_point(landmarks, indices[1], shape)
    p3 = landmark_to_point(landmarks, indices[2], shape)
    p4 = landmark_to_point(landmarks, indices[3], shape)
    p5 = landmark_to_point(landmarks, indices[4], shape)
    p6 = landmark_to_point(landmarks, indices[5], shape)

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal <= 1e-6:
        return 0.0
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def compute_average_ear(landmarks: list, shape: tuple[int, int, int]) -> float:
    left_ear = compute_eye_ear(landmarks, LEFT_EYE_INDICES, shape)
    right_ear = compute_eye_ear(landmarks, RIGHT_EYE_INDICES, shape)
    return (left_ear + right_ear) / 2.0


def save_labeled_image(
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    label: str,
) -> None:
    relative_path = image_path.relative_to(input_dir)
    destination = output_dir / label / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, destination)


def load_haar_detectors() -> HaarDetectors:
    """Load OpenCV Haar cascades used by the fallback labeler."""
    haar_root = Path(cv2.data.haarcascades)
    face_path = haar_root / "haarcascade_frontalface_default.xml"
    eye_path = haar_root / "haarcascade_eye_tree_eyeglasses.xml"

    face = cv2.CascadeClassifier(str(face_path))
    if face.empty():
        raise RuntimeError(f"Failed to load Haar face cascade: {face_path}")

    eye = cv2.CascadeClassifier(str(eye_path))
    if eye.empty():
        raise RuntimeError(f"Failed to load Haar eye cascade: {eye_path}")

    return HaarDetectors(face=face, eye=eye)


def detect_face_bbox(
    image: np.ndarray,
    detectors: HaarDetectors,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
) -> tuple[int, int, int, int] | None:
    """Detect the largest face bounding box in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detectors.face.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda bbox: bbox[2] * bbox[3])
    return int(x), int(y), int(w), int(h)


def format_distribution(stats: LabelStats) -> list[str]:
    labeled_total = stats.open_count + stats.closed_count
    return [
        f"Processed: {stats.processed}",
        f"Open: {stats.open_count}",
        f"Closed: {stats.closed_count}",
        f"Skipped (no face): {stats.skipped_no_face}",
        f"Labeled total: {labeled_total}",
    ]


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    image_paths = iter_image_files(args.input_dir)
    if not image_paths:
        print(f"[ERROR] No image files found under {args.input_dir}", file=sys.stderr)
        return 1

    try:
        labeler = build_labeler(args)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    stats = LabelStats()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Found {len(image_paths)} image(s) in {args.input_dir} | "
        f"EAR threshold={args.ear_threshold:.3f} | "
        f"backend={labeler.backend_name}"
    )

    with labeler:
        for image_path in image_paths:
            stats.processed += 1

            image = cv2.imread(str(image_path))
            if image is None:
                stats.skipped_no_face += 1
                continue

            detection_image = resize_for_detection(image, args.max_side)
            label, _score = labeler.predict(detection_image)
            if label is None:
                stats.skipped_no_face += 1
                continue

            save_labeled_image(
                image_path=image_path,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                label=label,
            )

            if label == "open":
                stats.open_count += 1
            else:
                stats.closed_count += 1

            if args.report_every > 0 and stats.processed % args.report_every == 0:
                print(
                    f"[PROGRESS] {stats.processed}/{len(image_paths)} images | "
                    f"open={stats.open_count} | closed={stats.closed_count} | "
                    f"skipped={stats.skipped_no_face}"
                )

    print("\n=== Label Distribution ===")
    for line in format_distribution(stats):
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
