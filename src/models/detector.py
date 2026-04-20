#!/usr/bin/env python3
"""Detector backbones for binary eye-state classification."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    mobilenet_v2,
    resnet18,
)

from models.baseline_cnn import BaselineCNN


DEFAULT_IMAGE_SIZE = 224
SUPPORTED_BACKBONES = ("custom", "mobilenetv2", "resnet18")
DEFAULT_TORCH_CACHE_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "torch_cache"


@dataclass
class DetectorConfig:
    """Configuration used to build an eye-state detector."""

    backbone: str = "custom"
    num_classes: int = 2
    image_size: int = DEFAULT_IMAGE_SIZE
    custom_dropout_rate: float = 0.3
    mobilenet_dropout_rate: float = 0.2
    mobilenet_trainable_blocks: int = 3
    resnet_dropout_rate: float = 0.2
    resnet_trainable_layers: int = 1
    use_pretrained: bool = True
    allow_pretrained_fallback: bool = True
    freeze_detector: bool = False
    print_frozen_layers: bool = True
    use_dual_input: bool = False
    dual_input_shared_backbone: bool = True


class CustomFeatureBackbone(nn.Module):
    """Feature extractor used for custom single-input or dual-input detectors."""

    output_dim: int = 128

    def __init__(self) -> None:
        super().__init__()
        baseline = BaselineCNN(num_classes=2, dropout_rate=0.0)
        self.features = baseline.features
        self.pool = baseline.pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, start_dim=1)


class EyeStateDetector(nn.Module):
    """Wrapper that switches between a custom CNN and MobileNetV2."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or DetectorConfig()
        self.backbone = normalize_backbone_name(self.config.backbone)
        self.use_dual_input = bool(self.config.use_dual_input)
        self.dual_input_shared_backbone = bool(self.config.dual_input_shared_backbone)
        self.frozen_layer_names: list[str] = []

        if self.use_dual_input:
            self._init_dual_input_model()
        else:
            self._init_single_input_model()

        if self.config.freeze_detector:
            self.frozen_layer_names = freeze_model(
                self,
                set_eval=True,
                module_name="detector",
                print_frozen_layers=self.config.print_frozen_layers,
            )

    def _init_single_input_model(self) -> None:
        """Preserve the original single-input detector structure for checkpoint compatibility."""
        if self.backbone == "custom":
            self.model = BaselineCNN(
                num_classes=self.config.num_classes,
                dropout_rate=self.config.custom_dropout_rate,
            )
            self.pretrained_loaded = False
        elif self.backbone == "mobilenetv2":
            self.model, self.pretrained_loaded = build_mobilenetv2_detector(
                num_classes=self.config.num_classes,
                dropout_rate=self.config.mobilenet_dropout_rate,
                trainable_feature_blocks=self.config.mobilenet_trainable_blocks,
                use_pretrained=self.config.use_pretrained,
                allow_pretrained_fallback=self.config.allow_pretrained_fallback,
            )
        elif self.backbone == "resnet18":
            self.model, self.pretrained_loaded = build_resnet18_detector(
                num_classes=self.config.num_classes,
                dropout_rate=self.config.resnet_dropout_rate,
                trainable_residual_layers=self.config.resnet_trainable_layers,
                use_pretrained=self.config.use_pretrained,
                allow_pretrained_fallback=self.config.allow_pretrained_fallback,
            )
        else:
            raise ValueError(
                f"Unsupported backbone '{self.config.backbone}'. "
                f"Expected one of: {', '.join(SUPPORTED_BACKBONES)}"
            )

    def _init_dual_input_model(self) -> None:
        """Build the dual-input detector with concatenated branch features."""
        feature_backbone, pretrained_loaded, feature_dim = build_feature_backbone(
            config=self.config,
        )
        classifier_dropout = resolve_classifier_dropout_rate(self.config)

        if self.dual_input_shared_backbone:
            self.shared_backbone = feature_backbone
        else:
            self.raw_backbone = feature_backbone
            self.enhanced_backbone = copy.deepcopy(feature_backbone)

        self.feature_dim = int(feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(self.feature_dim * 2, self.config.num_classes),
        )
        self.pretrained_loaded = pretrained_loaded

    def _extract_branch_features(
        self,
        raw_image: torch.Tensor,
        enhanced_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw and enhanced feature vectors for the dual-input detector."""
        if self.dual_input_shared_backbone:
            raw_features = self.shared_backbone(raw_image)
            enhanced_features = self.shared_backbone(enhanced_image)
        else:
            raw_features = self.raw_backbone(raw_image)
            enhanced_features = self.enhanced_backbone(enhanced_image)
        return raw_features, enhanced_features

    def forward(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
        enhanced_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_dual_input:
            if enhanced_x is not None:
                raise ValueError("Single-input detector received an unexpected second image tensor.")
            if isinstance(x, (tuple, list)):
                if len(x) != 1:
                    raise ValueError(
                        "Single-input detector expects one BCHW tensor, not a tuple/list of inputs."
                    )
                x = x[0]
            return self.model(x)

        if enhanced_x is None:
            if not isinstance(x, (tuple, list)) or len(x) != 2:
                raise ValueError(
                    "Dual-input detector expects `(raw_image, enhanced_image)` tensors."
                )
            raw_x, enhanced_x = x
        else:
            raw_x = x

        raw_features, enhanced_features = self._extract_branch_features(raw_x, enhanced_x)
        fused_features = torch.cat([raw_features, enhanced_features], dim=1)
        return self.classifier(fused_features)


def resolve_classifier_dropout_rate(config: DetectorConfig) -> float:
    """Return the classifier-head dropout used for the selected backbone."""
    if normalize_backbone_name(config.backbone) == "custom":
        return float(config.custom_dropout_rate)
    if normalize_backbone_name(config.backbone) == "mobilenetv2":
        return float(config.mobilenet_dropout_rate)
    return float(config.resnet_dropout_rate)


def normalize_backbone_name(backbone: str) -> str:
    """Normalize backbone aliases to one canonical string."""
    normalized = backbone.strip().lower().replace("_", "")
    if normalized in {"custom", "cnn", "baseline"}:
        return "custom"
    if normalized in {"mobilenetv2", "mobilenet"}:
        return "mobilenetv2"
    if normalized in {"resnet18", "resnet"}:
        return "resnet18"
    return normalized


def build_mobilenetv2_detector(
    *,
    num_classes: int = 2,
    dropout_rate: float = 0.2,
    trainable_feature_blocks: int = 3,
    use_pretrained: bool = True,
    allow_pretrained_fallback: bool = True,
) -> tuple[nn.Module, bool]:
    """Create a MobileNetV2 classifier and freeze all but the last feature blocks."""
    pretrained_loaded = False
    weights = None

    if use_pretrained:
        configure_torch_cache_dir()
        try:
            weights = MobileNet_V2_Weights.DEFAULT
        except Exception:
            weights = None

    try:
        model = mobilenet_v2(weights=weights)
        pretrained_loaded = weights is not None
    except Exception as exc:
        if use_pretrained and allow_pretrained_fallback:
            print(
                "[WARN] Falling back to randomly initialized MobileNetV2 because "
                f"pretrained weights could not be loaded: {exc}",
                file=sys.stderr,
            )
            model = mobilenet_v2(weights=None)
        else:
            raise

    freeze_mobilenetv2_early_layers(model, trainable_feature_blocks)

    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes),
    )
    return model, pretrained_loaded


def build_resnet18_detector(
    *,
    num_classes: int = 2,
    dropout_rate: float = 0.2,
    trainable_residual_layers: int = 1,
    use_pretrained: bool = True,
    allow_pretrained_fallback: bool = True,
) -> tuple[nn.Module, bool]:
    """Create a ResNet18 classifier and fine-tune only the last residual stages."""
    pretrained_loaded = False
    weights = None

    if use_pretrained:
        configure_torch_cache_dir()
        try:
            weights = ResNet18_Weights.DEFAULT
        except Exception:
            weights = None

    try:
        model = resnet18(weights=weights)
        pretrained_loaded = weights is not None
    except Exception as exc:
        if use_pretrained and allow_pretrained_fallback:
            print(
                "[WARN] Falling back to randomly initialized ResNet18 because "
                f"pretrained weights could not be loaded: {exc}",
                file=sys.stderr,
            )
            model = resnet18(weights=None)
        else:
            raise

    freeze_resnet18_early_layers(model, trainable_residual_layers)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes),
    )
    return model, pretrained_loaded


def build_custom_feature_backbone() -> tuple[nn.Module, bool, int]:
    """Create the custom CNN feature extractor used in dual-input mode."""
    return CustomFeatureBackbone(), False, CustomFeatureBackbone.output_dim


def build_mobilenetv2_feature_backbone(
    *,
    trainable_feature_blocks: int,
    use_pretrained: bool,
    allow_pretrained_fallback: bool,
) -> tuple[nn.Module, bool, int]:
    """Create a MobileNetV2 feature extractor that outputs pooled embeddings."""
    model, pretrained_loaded = build_mobilenetv2_detector(
        num_classes=2,
        dropout_rate=0.0,
        trainable_feature_blocks=trainable_feature_blocks,
        use_pretrained=use_pretrained,
        allow_pretrained_fallback=allow_pretrained_fallback,
    )
    if isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 0:
        in_features = int(model.classifier[-1].in_features)
    else:
        in_features = int(model.last_channel)
    model.classifier = nn.Identity()
    return model, pretrained_loaded, in_features


def build_resnet18_feature_backbone(
    *,
    trainable_residual_layers: int,
    use_pretrained: bool,
    allow_pretrained_fallback: bool,
) -> tuple[nn.Module, bool, int]:
    """Create a ResNet18 feature extractor that returns pooled 512-D embeddings."""
    model, pretrained_loaded = build_resnet18_detector(
        num_classes=2,
        dropout_rate=0.0,
        trainable_residual_layers=trainable_residual_layers,
        use_pretrained=use_pretrained,
        allow_pretrained_fallback=allow_pretrained_fallback,
    )
    if isinstance(model.fc, nn.Sequential) and len(model.fc) > 0:
        in_features = int(model.fc[-1].in_features)
    else:
        in_features = int(model.fc.in_features)
    model.fc = nn.Identity()
    return model, pretrained_loaded, in_features


def build_feature_backbone(
    *,
    config: DetectorConfig,
) -> tuple[nn.Module, bool, int]:
    """Create the shared/separate branch backbone used by the dual-input detector."""
    backbone = normalize_backbone_name(config.backbone)
    if backbone == "custom":
        return build_custom_feature_backbone()
    if backbone == "mobilenetv2":
        return build_mobilenetv2_feature_backbone(
            trainable_feature_blocks=config.mobilenet_trainable_blocks,
            use_pretrained=config.use_pretrained,
            allow_pretrained_fallback=config.allow_pretrained_fallback,
        )
    if backbone == "resnet18":
        return build_resnet18_feature_backbone(
            trainable_residual_layers=config.resnet_trainable_layers,
            use_pretrained=config.use_pretrained,
            allow_pretrained_fallback=config.allow_pretrained_fallback,
        )
    raise ValueError(
        f"Unsupported backbone '{config.backbone}'. "
        f"Expected one of: {', '.join(SUPPORTED_BACKBONES)}"
    )


def configure_torch_cache_dir(cache_dir: Path = DEFAULT_TORCH_CACHE_DIR) -> None:
    """Point Torch's model cache at a writable project-local directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache_dir))
    torch.hub.set_dir(str(cache_dir))


def freeze_mobilenetv2_early_layers(
    model: nn.Module,
    trainable_feature_blocks: int = 3,
) -> None:
    """Freeze early MobileNetV2 features and keep only the last blocks trainable."""
    if not hasattr(model, "features"):
        raise ValueError("Expected a MobileNetV2 model with a features attribute.")

    features = model.features
    total_blocks = len(features)
    if total_blocks == 0:
        raise ValueError("MobileNetV2 features are empty.")

    trainable_feature_blocks = max(0, min(trainable_feature_blocks, total_blocks))

    for parameter in features.parameters():
        parameter.requires_grad = False

    if trainable_feature_blocks > 0:
        for block in features[-trainable_feature_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def freeze_resnet18_early_layers(
    model: nn.Module,
    trainable_residual_layers: int = 1,
) -> None:
    """Freeze early ResNet18 stages and fine-tune the last residual layers."""
    residual_layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    trainable_residual_layers = max(0, min(trainable_residual_layers, len(residual_layers)))

    for parameter in model.parameters():
        parameter.requires_grad = False

    if trainable_residual_layers > 0:
        for layer in residual_layers[-trainable_residual_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def collect_layer_names_from_parameters(model: nn.Module) -> list[str]:
    """Return a sorted list of layer names inferred from named parameters."""
    layer_names = set()
    for parameter_name, _ in model.named_parameters():
        if "." in parameter_name:
            layer_names.add(parameter_name.rsplit(".", 1)[0])
        else:
            layer_names.add(parameter_name)
    return sorted(layer_names)


def freeze_model(
    model: nn.Module,
    *,
    set_eval: bool = True,
    module_name: str = "model",
    print_frozen_layers: bool = True,
) -> list[str]:
    """Freeze a model for inference by disabling grads and optionally switching to eval.

    This helper is used by the "Enhancer + Frozen Detector" experiment so the
    detector stays fixed and only the forward pass is used.
    """
    frozen_parameter_count = 0
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False
        parameter.grad = None
        frozen_parameter_count += 1

    if set_eval:
        model.eval()

    frozen_layer_names = collect_layer_names_from_parameters(model)
    if print_frozen_layers:
        print(
            f"[Freeze] {module_name}: froze {frozen_parameter_count} parameter tensors "
            f"across {len(frozen_layer_names)} layers and set eval={set_eval}."
        )
        for layer_name in frozen_layer_names:
            print(f"[Freeze] {module_name}: {layer_name}")
    return frozen_layer_names


def count_stored_gradients(model: nn.Module) -> int:
    """Return how many parameters currently have a stored gradient tensor."""
    return sum(1 for parameter in model.parameters() if parameter.grad is not None)


def gradients_enabled_for_model(model: nn.Module) -> bool:
    """Return True when any parameter still allows gradient computation."""
    return any(parameter.requires_grad for parameter in model.parameters())


def format_gradient_debug_info(
    model: nn.Module,
    *,
    module_name: str = "model",
) -> str:
    """Return a compact debug line describing freeze and gradient state."""
    total_params, trainable_params = count_parameters(model)
    stored_gradients = count_stored_gradients(model)
    return (
        f"[FreezeDebug] {module_name}: "
        f"training={model.training} | "
        f"trainable_params={trainable_params}/{total_params} | "
        f"requires_grad_enabled={gradients_enabled_for_model(model)} | "
        f"stored_gradients={stored_gradients}"
    )


def print_gradient_debug_info(
    model: nn.Module,
    *,
    module_name: str = "model",
) -> None:
    """Print the current freeze and gradient state."""
    print(format_gradient_debug_info(model, module_name=module_name))


def summarize_model(
    model: nn.Module,
    *,
    input_size: tuple[int, int, int, int] = (1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
) -> str:
    """Build a compact summary string including parameter counts and output size."""
    try:
        reference_parameter = next(model.parameters())
        device = reference_parameter.device
    except StopIteration:
        device = torch.device("cpu")

    was_training = model.training
    model.eval()

    with torch.no_grad():
        dummy_input = torch.zeros(input_size, device=device)
        if getattr(model, "use_dual_input", False):
            output = model(dummy_input, dummy_input.clone())
        else:
            output = model(dummy_input)

    if was_training:
        model.train()

    total_params, trainable_params = count_parameters(model)
    frozen_params = total_params - trainable_params
    backbone_name = getattr(model, "backbone", model.__class__.__name__)
    pretrained_loaded = getattr(model, "pretrained_loaded", False)

    lines = [
        f"Model summary for {model.__class__.__name__}",
        f"Backbone: {backbone_name}",
        (
            f"Input size per branch: {tuple(input_size)}"
            if getattr(model, "use_dual_input", False)
            else f"Input size: {tuple(input_size)}"
        ),
        f"Output size: {tuple(output.shape)}",
        f"Total parameters: {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
        f"Frozen parameters: {frozen_params:,}",
        f"Pretrained weights loaded: {pretrained_loaded}",
    ]

    config = getattr(model, "config", None)
    if isinstance(config, DetectorConfig):
        lines.append(f"Dual input enabled: {config.use_dual_input}")
        if config.use_dual_input:
            lines.append(
                "Dual-input backbone sharing: "
                f"{'shared weights' if config.dual_input_shared_backbone else 'separate weights'}"
            )
    if isinstance(config, DetectorConfig) and backbone_name == "mobilenetv2":
        lines.append(
            "Trainable MobileNetV2 feature blocks: "
            f"{config.mobilenet_trainable_blocks}"
        )
    if isinstance(config, DetectorConfig) and backbone_name == "resnet18":
        lines.append(
            "Trainable ResNet18 residual stages: "
            f"{config.resnet_trainable_layers}"
        )

    return "\n".join(lines)


def print_model_summary(
    model: nn.Module,
    *,
    input_size: tuple[int, int, int, int] = (1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
) -> None:
    """Print a compact model summary."""
    print(summarize_model(model, input_size=input_size))


def build_detector(
    *,
    backbone: str = "custom",
    num_classes: int = 2,
    image_size: int = DEFAULT_IMAGE_SIZE,
    custom_dropout_rate: float = 0.3,
    mobilenet_dropout_rate: float = 0.2,
    mobilenet_trainable_blocks: int = 3,
    resnet_dropout_rate: float = 0.2,
    resnet_trainable_layers: int = 1,
    use_pretrained: bool = True,
    allow_pretrained_fallback: bool = True,
    freeze_detector: bool = False,
    print_frozen_layers: bool = True,
    use_dual_input: bool = False,
    dual_input_shared_backbone: bool = True,
    print_summary: bool = False,
) -> EyeStateDetector:
    """Build a detector from a named backbone."""
    config = DetectorConfig(
        backbone=backbone,
        num_classes=num_classes,
        image_size=image_size,
        custom_dropout_rate=custom_dropout_rate,
        mobilenet_dropout_rate=mobilenet_dropout_rate,
        mobilenet_trainable_blocks=mobilenet_trainable_blocks,
        resnet_dropout_rate=resnet_dropout_rate,
        resnet_trainable_layers=resnet_trainable_layers,
        use_pretrained=use_pretrained,
        allow_pretrained_fallback=allow_pretrained_fallback,
        freeze_detector=freeze_detector,
        print_frozen_layers=print_frozen_layers,
        use_dual_input=use_dual_input,
        dual_input_shared_backbone=dual_input_shared_backbone,
    )
    model = EyeStateDetector(config=config)
    if print_summary:
        print_model_summary(model, input_size=(1, 3, image_size, image_size))
    return model


def run_dummy_forward_pass(
    model: nn.Module,
    *,
    batch_size: int = 2,
    image_size: int = DEFAULT_IMAGE_SIZE,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Run a small dummy forward pass to sanity-check output shape."""
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    model = model.to(target_device)
    dummy_batch = torch.randn(batch_size, 3, image_size, image_size, device=target_device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        if getattr(model, "use_dual_input", False):
            outputs = model(dummy_batch, dummy_batch.clone())
        else:
            outputs = model(dummy_batch)
    if was_training:
        model.train()
    return outputs.cpu()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for quick local detector checks."""
    parser = argparse.ArgumentParser(description="Build and sanity-check an eye-state detector.")
    parser.add_argument(
        "--backbone",
        choices=SUPPORTED_BACKBONES,
        default="custom",
        help="Detector backbone to build.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Dummy batch size for the forward-pass sanity check.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Input image size used for the summary and dummy forward pass.",
    )
    parser.add_argument(
        "--mobilenet-trainable-blocks",
        type=int,
        default=3,
        help="Number of final MobileNetV2 feature blocks to keep trainable.",
    )
    parser.add_argument(
        "--resnet-trainable-layers",
        type=int,
        default=1,
        help="Number of final ResNet18 residual stages to keep trainable.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained weights for the MobileNetV2 path.",
    )
    parser.add_argument(
        "--use-dual-input",
        action="store_true",
        help="Build the detector in dual-input mode (raw + enhanced image).",
    )
    parser.add_argument(
        "--dual-input-separate-backbones",
        action="store_true",
        help="Use separate backbone copies instead of shared weights in dual-input mode.",
    )
    return parser.parse_args()


def main() -> None:
    """Print a model summary and run a dummy forward pass."""
    args = parse_args()
    model = build_detector(
        backbone=args.backbone,
        image_size=args.image_size,
        mobilenet_trainable_blocks=args.mobilenet_trainable_blocks,
        resnet_trainable_layers=args.resnet_trainable_layers,
        use_pretrained=not args.no_pretrained,
        use_dual_input=args.use_dual_input,
        dual_input_shared_backbone=not args.dual_input_separate_backbones,
        print_summary=True,
    )
    outputs = run_dummy_forward_pass(
        model,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    print(f"Dummy forward output shape: {tuple(outputs.shape)}")


if __name__ == "__main__":
    main()
