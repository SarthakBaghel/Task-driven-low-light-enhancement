from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontcache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT.parent / "results"
FIGURES_DIR = ROOT / "figures"

SEED42_CSV = RESULTS_DIR / "final_comparison_seed42.csv"
SEED314_CSV = RESULTS_DIR / "final_comparison_seed314.csv"

TRADEOFF_OUTPUT = FIGURES_DIR / "model_tradeoff_analysis.png"


MODEL_ORDER = [
    "Clean detector",
    "Low-light detector",
    "Mixed detector",
]

MODEL_COLORS = {
    "Clean detector": "#4C78A8",
    "Low-light detector": "#F58518",
    "Mixed detector": "#54A24B",
}

DOMAIN_LABELS = {
    "Clean F1": "Clean",
    "Low-light F1": "Low-light",
}


def load_results() -> pd.DataFrame:
    frames = []
    for seed, csv_path in [(42, SEED42_CSV), (314, SEED314_CSV)]:
        frame = pd.read_csv(csv_path)
        frame["Subset Seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "Clean F1",
        "Low-light F1",
        "Clean Closed Recall",
        "Low-light Closed Recall",
        "Avg F1",
    ]
    summary = (
        df.groupby("Model", sort=False)[metric_cols]
        .agg(["mean", "std"])
        .reindex(MODEL_ORDER)
    )
    return summary


def build_tradeoff_plot(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    ax = axes[0]
    for model in MODEL_ORDER:
        color = MODEL_COLORS[model]
        model_rows = df[df["Model"] == model]
        ax.scatter(
            model_rows["Clean F1"],
            model_rows["Low-light F1"],
            s=48,
            color=color,
            alpha=0.35,
            edgecolor="none",
        )

        mean_clean = summary.loc[model, ("Clean F1", "mean")]
        std_clean = summary.loc[model, ("Clean F1", "std")]
        mean_lowlight = summary.loc[model, ("Low-light F1", "mean")]
        std_lowlight = summary.loc[model, ("Low-light F1", "std")]

        ax.errorbar(
            mean_clean,
            mean_lowlight,
            xerr=std_clean,
            yerr=std_lowlight,
            fmt="o",
            markersize=9,
            capsize=4,
            lw=1.6,
            color=color,
            ecolor=color,
        )
        ax.text(
            mean_clean + 0.35,
            mean_lowlight + 0.15,
            model.replace(" detector", ""),
            fontsize=9,
            weight="bold",
            color=color,
        )

    ax.set_title("Domain Trade-off in F1 Score", fontsize=12, weight="bold")
    ax.set_xlabel("Clean F1 (%)")
    ax.set_ylabel("Low-light F1 (%)")
    ax.set_xlim(66, 96)
    ax.set_ylim(86, 95.5)
    ax.plot([66, 96], [66, 96], ls="--", lw=1.0, color="#999999", alpha=0.65)
    ax.text(93.8, 93.1, "Ideal\n(top-right)", ha="right", va="top", fontsize=8, color="#666666")

    ax = axes[1]
    x_positions = range(len(MODEL_ORDER))
    width = 0.34

    clean_means = [summary.loc[model, ("Clean Closed Recall", "mean")] for model in MODEL_ORDER]
    clean_stds = [summary.loc[model, ("Clean Closed Recall", "std")] for model in MODEL_ORDER]
    low_means = [summary.loc[model, ("Low-light Closed Recall", "mean")] for model in MODEL_ORDER]
    low_stds = [summary.loc[model, ("Low-light Closed Recall", "std")] for model in MODEL_ORDER]

    bars_clean = ax.bar(
        [x - width / 2 for x in x_positions],
        clean_means,
        width=width,
        yerr=clean_stds,
        capsize=3,
        label="Clean",
        color="#8FB7E3",
        edgecolor="#4C78A8",
    )
    bars_low = ax.bar(
        [x + width / 2 for x in x_positions],
        low_means,
        width=width,
        yerr=low_stds,
        capsize=3,
        label="Low-light",
        color="#A9D99B",
        edgecolor="#54A24B",
    )

    for bar_group in [bars_clean, bars_low]:
        for bar in bar_group:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.35,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Closed-eye Recall by Domain", fontsize=12, weight="bold")
    ax.set_ylabel("Closed-eye Recall (%)")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(["Clean", "Low-light", "Mixed"])
    ax.set_ylim(75, 101.5)
    ax.legend(frameon=True, fontsize=9)

    fig.suptitle(
        "Final Detector Behavior Across Clean and Low-light Domains",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(TRADEOFF_OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    combined = load_results()
    summary = summarize(combined)
    build_tradeoff_plot(combined, summary)
    print(f"Saved {TRADEOFF_OUTPUT}")


if __name__ == "__main__":
    main()
