#!/usr/bin/env python3
"""Helpers for switching smoothly between local and Google Colab training."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Iterable


def is_colab() -> bool:
    """Return True when the current process is running inside Google Colab."""
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_runtime(runtime_name: str = "auto") -> str:
    """Resolve the active runtime to either 'local' or 'colab'."""
    normalized = runtime_name.strip().lower()
    if normalized == "auto":
        return "colab" if is_colab() else "local"
    if normalized not in {"local", "colab"}:
        raise ValueError("runtime must be one of: auto, local, colab")
    return normalized


def mount_google_drive(
    *,
    mount_point: str | Path = "/content/drive",
    force_remount: bool = False,
) -> Path | None:
    """Mount Google Drive when running inside Colab."""
    if not is_colab():
        print("[WARN] Google Drive mount requested outside Colab; skipping mount.")
        return None

    from google.colab import drive  # type: ignore

    mount_path = Path(mount_point).expanduser()
    drive.mount(str(mount_path), force_remount=force_remount)
    return mount_path


def resolve_workspace_root(
    *,
    runtime: str,
    local_workspace_root: str | Path,
    colab_workspace_root: str | Path,
) -> Path:
    """Pick the correct workspace root for the active runtime."""
    selected_root = colab_workspace_root if runtime == "colab" else local_workspace_root
    return Path(selected_root).expanduser().resolve()


def resolve_runtime_path(path_value: str | Path, *, workspace_root: str | Path) -> Path:
    """Resolve a path relative to the chosen workspace root unless already absolute."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(workspace_root).expanduser().resolve() / path).resolve()


def copy_checkpoints_to_directory(
    source_paths: Iterable[str | Path],
    *,
    destination_dir: str | Path,
) -> list[Path]:
    """Copy checkpoints to a secondary directory such as Google Drive."""
    destination_root = Path(destination_dir).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    copied_paths: list[Path] = []
    for source_path in source_paths:
        source = Path(source_path).expanduser().resolve()
        if not source.is_file():
            continue
        destination = destination_root / source.name
        shutil.copy2(source, destination)
        copied_paths.append(destination)
    return copied_paths
