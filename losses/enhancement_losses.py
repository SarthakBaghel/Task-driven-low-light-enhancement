#!/usr/bin/env python3
"""Loss functions commonly used for Zero-DCE style low-light enhancement."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _validate_rgb_image(image: torch.Tensor, name: str) -> None:
    """Ensure the tensor is a BCHW RGB image."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(
            f"{name} must have shape [batch, 3, height, width], "
            f"but received {tuple(image.shape)}."
        )


def _normalize_patch_size(
    patch_size: int | Sequence[int],
    height: int,
    width: int,
) -> tuple[int, int]:
    """Convert an int or tuple patch size into a valid kernel size."""
    if isinstance(patch_size, int):
        patch_h = patch_w = patch_size
    else:
        patch_h, patch_w = patch_size

    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("'patch_size' values must be greater than 0.")

    return min(int(patch_h), height), min(int(patch_w), width)


def _reduce_loss(loss_map: torch.Tensor, reduction: str) -> torch.Tensor:
    """Apply the requested reduction to a loss tensor."""
    if reduction == "mean":
        return loss_map.mean()
    if reduction == "sum":
        return loss_map.sum()
    if reduction == "none":
        return loss_map
    raise ValueError("reduction must be one of: 'mean', 'sum', or 'none'.")


def _flatten_curve_maps(curve_maps: torch.Tensor) -> torch.Tensor:
    """Convert curve maps to [batch, channels, height, width] for TV-style loss."""
    if curve_maps.ndim == 4:
        return curve_maps
    if curve_maps.ndim == 5:
        batch_size, iterations, channels, height, width = curve_maps.shape
        return curve_maps.view(batch_size, iterations * channels, height, width)
    raise ValueError(
        "curve_maps must have shape [batch, channels, height, width] or "
        "[batch, iterations, channels, height, width]."
    )


def exposure_control_loss(
    enhanced_image: torch.Tensor,
    *,
    patch_size: int | Sequence[int] = 16,
    target_mean: float = 0.6,
    reduction: str = "mean",
) -> torch.Tensor:
    """Encourage local regions to reach a desired exposure level.

    Zero-DCE uses local averages rather than pixel-wise brightness so the model
    learns to correct underexposed areas without forcing every pixel to the same
    intensity.
    """
    _validate_rgb_image(enhanced_image, "enhanced_image")

    height, width = enhanced_image.shape[-2:]
    kernel_size = _normalize_patch_size(patch_size, height, width)
    grayscale = enhanced_image.mean(dim=1, keepdim=True)
    pooled_mean = F.avg_pool2d(grayscale, kernel_size=kernel_size, stride=kernel_size)
    loss_map = torch.square(pooled_mean - float(target_mean))
    return _reduce_loss(loss_map, reduction)


def color_constancy_loss(
    enhanced_image: torch.Tensor,
    *,
    use_sqrt: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """Discourage large color imbalance between the RGB channels.

    The idea is borrowed from the gray-world assumption: after enhancement,
    channel averages should remain reasonably balanced instead of drifting toward
    strong color casts.
    """
    _validate_rgb_image(enhanced_image, "enhanced_image")

    mean_rgb = enhanced_image.mean(dim=(2, 3))
    red, green, blue = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]

    loss_map = (
        torch.square(red - green)
        + torch.square(red - blue)
        + torch.square(green - blue)
    )
    if use_sqrt:
        loss_map = torch.sqrt(loss_map + eps)
    return _reduce_loss(loss_map, reduction)


def illumination_smoothness_loss(
    curve_maps: torch.Tensor,
    *,
    power: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Regularize curve maps so illumination changes stay spatially smooth.

    This is the illumination smoothness / total-variation style prior used to
    avoid noisy, blotchy enhancement maps.
    """
    flattened = _flatten_curve_maps(curve_maps)
    if power <= 0:
        raise ValueError("'power' must be greater than 0.")

    diff_h = flattened[:, :, 1:, :] - flattened[:, :, :-1, :]
    diff_w = flattened[:, :, :, 1:] - flattened[:, :, :, :-1]

    loss_h = torch.abs(diff_h).pow(power)
    loss_w = torch.abs(diff_w).pow(power)

    if reduction == "none":
        return loss_h.mean(dim=(1, 2, 3)) + loss_w.mean(dim=(1, 2, 3))
    return _reduce_loss(loss_h, reduction) + _reduce_loss(loss_w, reduction)


def spatial_consistency_loss(
    input_image: torch.Tensor,
    enhanced_image: torch.Tensor,
    *,
    pool_size: int | Sequence[int] = 4,
    reduction: str = "mean",
) -> torch.Tensor:
    """Preserve local intensity relationships between the input and output.

    Instead of matching raw pixels, this compares directional gradients on
    pooled luminance maps so the enhancer keeps coarse spatial structure while
    adjusting brightness.
    """
    _validate_rgb_image(input_image, "input_image")
    _validate_rgb_image(enhanced_image, "enhanced_image")
    if input_image.shape != enhanced_image.shape:
        raise ValueError(
            "input_image and enhanced_image must have the same shape, "
            f"but received {tuple(input_image.shape)} and {tuple(enhanced_image.shape)}."
        )

    height, width = input_image.shape[-2:]
    kernel_size = _normalize_patch_size(pool_size, height, width)

    input_luma = input_image.mean(dim=1, keepdim=True)
    enhanced_luma = enhanced_image.mean(dim=1, keepdim=True)

    input_pooled = F.avg_pool2d(input_luma, kernel_size=kernel_size, stride=kernel_size)
    enhanced_pooled = F.avg_pool2d(
        enhanced_luma,
        kernel_size=kernel_size,
        stride=kernel_size,
    )

    kernels = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],  # left
            [[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]],  # right
            [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],  # up
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],  # down
        ],
        dtype=input_pooled.dtype,
        device=input_pooled.device,
    ).unsqueeze(1)

    input_gradients = F.conv2d(input_pooled, kernels, padding=1)
    enhanced_gradients = F.conv2d(enhanced_pooled, kernels, padding=1)
    loss_map = torch.square(enhanced_gradients - input_gradients)

    if reduction == "none":
        return loss_map.mean(dim=(1, 2, 3))
    return _reduce_loss(loss_map, reduction)


def enhancement_loss(
    enhanced_image: torch.Tensor,
    curve_maps: torch.Tensor,
    *,
    input_image: torch.Tensor | None = None,
    exposure_weight: float = 10.0,
    color_weight: float = 5.0,
    smoothness_weight: float = 200.0,
    spatial_weight: float = 1.0,
    exposure_patch_size: int | Sequence[int] = 16,
    exposure_target_mean: float = 0.6,
    color_use_sqrt: bool = False,
    color_eps: float = 1e-8,
    smoothness_power: float = 2.0,
    spatial_pool_size: int | Sequence[int] = 4,
    use_spatial_consistency: bool = True,
) -> dict[str, torch.Tensor]:
    """Combine Zero-DCE losses into one configurable objective.

    Returns a dictionary so training code can log both the total objective and
    the individual components.
    """
    exp_loss = exposure_control_loss(
        enhanced_image,
        patch_size=exposure_patch_size,
        target_mean=exposure_target_mean,
    )
    col_loss = color_constancy_loss(
        enhanced_image,
        use_sqrt=color_use_sqrt,
        eps=color_eps,
    )
    smooth_loss = illumination_smoothness_loss(
        curve_maps,
        power=smoothness_power,
    )

    if use_spatial_consistency:
        if input_image is None:
            raise ValueError(
                "input_image is required when use_spatial_consistency=True."
            )
        spa_loss = spatial_consistency_loss(
            input_image,
            enhanced_image,
            pool_size=spatial_pool_size,
        )
    else:
        spa_loss = enhanced_image.new_tensor(0.0)

    total = (
        exposure_weight * exp_loss
        + color_weight * col_loss
        + smoothness_weight * smooth_loss
        + spatial_weight * spa_loss
    )

    return {
        "total": total,
        "exposure": exp_loss,
        "color_constancy": col_loss,
        "illumination_smoothness": smooth_loss,
        "spatial_consistency": spa_loss,
    }
