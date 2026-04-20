#!/usr/bin/env python3
"""Run Zero-DCE enhancement on a single low-light image and save the result."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models.zerodce import ZeroDCE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance one low-light image using a Zero-DCE style network.",
    )
    parser.add_argument("image_path", type=str, help="Path to the input low-light RGB image.")
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/enhanced_image.png",
        help="Path to save the enhanced image.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Optional trained model checkpoint. If omitted, random weights are used.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto, cpu, cuda, mps, or an explicit device like cuda:0.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=32,
        help="Hidden channel width used when no checkpoint config is available.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=8,
        help="Number of iterative enhancement curves to apply.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested inference device."""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def load_image_as_tensor(image_path: str | Path, device: torch.device) -> tuple[torch.Tensor, Image.Image]:
    """Load an RGB image and convert it to a BCHW float tensor in [0, 1]."""
    path = Path(image_path).expanduser().resolve()
    image = Image.open(path).convert("RGB")
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, image


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert a single-image tensor in [0, 1] to a PIL image."""
    tensor = image_tensor.detach().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)

    image_array = tensor.permute(1, 2, 0).numpy()
    image_array = np.clip(image_array * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(image_array)


def extract_model_state(checkpoint: object) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Extract a state dict and optional model config from flexible checkpoint formats."""
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"], checkpoint.get("model_config", {})
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"], checkpoint.get("model_config", {})
        if checkpoint and all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint, {}
    raise ValueError("Unsupported checkpoint format for ZeroDCE model loading.")


def load_model(
    *,
    checkpoint_path: str | Path | None,
    device: torch.device,
    hidden_channels: int = 32,
    num_iterations: int = 8,
) -> ZeroDCE:
    """Create the Zero-DCE model and optionally load trained weights."""
    model = ZeroDCE(
        hidden_channels=hidden_channels,
        num_iterations=num_iterations,
    )

    if checkpoint_path is not None:
        resolved_checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        try:
            checkpoint = torch.load(
                resolved_checkpoint_path,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(resolved_checkpoint_path, map_location=device)
        state_dict, model_config = extract_model_state(checkpoint)
        model = ZeroDCE(
            hidden_channels=int(model_config.get("hidden_channels", hidden_channels)),
            num_iterations=int(model_config.get("num_iterations", num_iterations)),
        )
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


def enhance_tensor(
    model: ZeroDCE,
    image_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enhance one input tensor and return the enhanced result and curve maps."""
    with torch.no_grad():
        enhanced_image, curve_maps = model(image_tensor)
    return enhanced_image, curve_maps


def save_image(image: Image.Image, output_path: str | Path) -> Path:
    """Save an image and return the resolved path."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def enhance_image_file(
    image_path: str | Path,
    *,
    output_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device_name: str = "auto",
    hidden_channels: int = 32,
    num_iterations: int = 8,
) -> dict[str, object]:
    """Load one image, enhance it, and optionally save the result."""
    device = resolve_device(device_name)
    model = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
        hidden_channels=hidden_channels,
        num_iterations=num_iterations,
    )
    input_tensor, input_image = load_image_as_tensor(image_path, device)
    enhanced_tensor, curve_maps = enhance_tensor(model, input_tensor)
    enhanced_image = tensor_to_pil(enhanced_tensor)

    saved_path = None
    if output_path is not None:
        saved_path = save_image(enhanced_image, output_path)

    return {
        "device": str(device),
        "model": model,
        "input_tensor": input_tensor,
        "input_image": input_image,
        "enhanced_tensor": enhanced_tensor,
        "enhanced_image": enhanced_image,
        "curve_maps": curve_maps,
        "saved_path": saved_path,
    }


def main() -> None:
    args = parse_args()
    result = enhance_image_file(
        args.image_path,
        output_path=args.output_path,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
        hidden_channels=args.hidden_channels,
        num_iterations=args.num_iterations,
    )

    print(f"Device: {result['device']}")
    print(f"Input image: {Path(args.image_path).expanduser().resolve()}")
    print(f"Enhanced image: {result['saved_path']}")
    print(f"Curve map tensor shape: {tuple(result['curve_maps'].shape)}")
    if args.checkpoint is None:
        print("No checkpoint was provided. Enhancement used randomly initialized weights.")


if __name__ == "__main__":
    main()
