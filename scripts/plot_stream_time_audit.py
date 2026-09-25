"""Render aggregate research evidence; no model fitting or raw-label access."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/stream_time_audit"


def main():
    scores = pd.read_csv(REPORT / "results.csv")
    sensitivity = pd.read_csv(REPORT / "sensitivity.csv")
    views = ["original", "valid_like", "test_like"]
    labels = ["Original TRAIN", "Moderate synthetic noise", "Harsher synthetic noise"]
    variants = {
        "control": "Control",
        "no_temporal": "Without temporal columns",
        "no_refund": "Without refund columns",
        "cycle_state": "Added explicit cycle state",
    }
    colors = ["#244B68", "#B56642", "#929DA6", "#37958C"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    positions = np.arange(3)
    for offset, ((variant, name), color) in enumerate(zip(variants.items(), colors)):
        values = (
            scores[scores.variant.eq(variant)].set_index("view").loc[views, "macro_f1"]
        )
        bars = ax.bar(
            positions + (offset - 1.5) * 0.2, values, 0.19, label=name, color=color
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, min(1, scores.macro_f1.max() + 0.13))
    ax.set_ylabel("Pooled eight-class Macro-F1")
    ax.set_title(
        "Relative-time investigation: five client folds, 2,000 TRAIN clients",
        loc="left",
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.text(
        0.06,
        0.015,
        "Single-seed compact component. Synthetic views are not official VALID/TEST scores. No production promotion.",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(bottom=0.23, top=0.88, left=0.08, right=0.98)
    fig.savefig(REPORT / "paired_results.png", dpi=160)
    plt.close(fig)

    selected = sensitivity[sensitivity.intervention.str.startswith("permute_")].copy()
    selected["group"] = selected.intervention.str.replace(
        r"^permute_|_\d+$", "", regex=True
    )
    groups = ["temporal", "refund", "identity", "amount", "context"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for ax, view, label in zip(axes, views, labels):
        aggregate = (
            selected[selected.view.eq(view)]
            .groupby("group")
            .f1_loss.agg(["mean", "min", "max"])
            .reindex(groups)
        )
        error = np.vstack(
            [aggregate["mean"] - aggregate["min"], aggregate["max"] - aggregate["mean"]]
        )
        ax.barh(groups, aggregate["mean"], xerr=error, color="#244B68", capsize=3)
        ax.axvline(0, color="#888888", linewidth=0.8)
        ax.set_title(label, fontsize=11)
        ax.set_xlim(
            min(0.0, float(selected.f1_loss.min()) * 1.1),
            float(selected.f1_loss.max()) * 1.1,
        )
        ax.set_xlabel("Macro-F1 loss after permutation")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle(
        "What the held-out component relies on", x=0.08, ha="left", fontweight="bold"
    )
    fig.text(
        0.06,
        0.025,
        "Three fixed permutations. Refund group excludes time-dependent refund fields, which belong to temporal. Not causal shares.",
        fontsize=8.5,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(REPORT / "group_sensitivity.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
