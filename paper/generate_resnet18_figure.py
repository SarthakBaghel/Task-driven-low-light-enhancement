from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontcache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"
OUTPUT = FIGURES_DIR / "resnet18_architecture_real.png"


def add_box(ax, x, y, w, h, text, facecolor, edgecolor="#2f4b6e", fontsize=10, weight="bold"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.5,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        family="sans-serif",
        color="#16324f",
        wrap=True,
    )
    return patch


def add_arrow(ax, x0, y0, x1, y1):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color="#333333",
        )
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15.5, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.30
    h = 0.42

    boxes = [
        (0.02, 0.09, "Input\n224×224×3", "#d9edf7", 11),
        (0.14, 0.10, "Conv 7×7, 64\nstride 2\n+ BN + ReLU", "#d4f0c8", 10),
        (0.27, 0.08, "MaxPool\n3×3, stride 2", "#ececec", 10),
        (0.39, 0.13, "Layer 1\n2 BasicBlocks\n[64, 64]", "#b9d6f2", 10),
        (0.52, 0.13, "Layer 2\n2 BasicBlocks\n[128, 128]\nstride 2", "#b9d6f2", 10),
        (0.65, 0.13, "Layer 3\n2 BasicBlocks\n[256, 256]\nstride 2", "#b9d6f2", 10),
        (0.78, 0.13, "Layer 4\n2 BasicBlocks\n[512, 512]\nstride 2", "#b9d6f2", 10),
        (0.90, 0.10, "Global\nAvgPool", "#ececec", 10),
    ]

    sizes = [
        ("224×224", 0.065),
        ("112×112", 0.205),
        ("56×56", 0.335),
        ("56×56", 0.465),
        ("28×28", 0.595),
        ("14×14", 0.725),
        ("7×7", 0.855),
    ]

    for label, xpos in sizes:
        ax.text(xpos, 0.18, label, ha="center", va="center", fontsize=9, color="#555555")

    patches = []
    for x, w, text, color, fontsize in boxes:
        patches.append(add_box(ax, x, y, w, h, text, color, fontsize=fontsize))

    fc = add_box(ax, 1.01, y, 0.10, h, "FC\n2 classes\n(open,\nclosed)", "#f8d7da", edgecolor="#ad3f4b", fontsize=10)

    all_patches = patches + [fc]
    for left, right in zip(all_patches[:-1], all_patches[1:]):
        add_arrow(
            ax,
            left.get_x() + left.get_width(),
            y + h / 2,
            right.get_x(),
            y + h / 2,
        )

    ax.text(
        0.50,
        0.93,
        "ResNet18 Detector Used in This Work",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#153b63",
    )
    ax.text(
        0.50,
        0.08,
        "Standard ResNet18 backbone with residual stage pattern [2, 2, 2, 2]; final fully connected layer replaced for binary eye-state classification.",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )

    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
