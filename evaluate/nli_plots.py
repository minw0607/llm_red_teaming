"""
evaluate/nli_plots.py — Visualisation for NLI robustness runs (NB05).

Keeps the demo notebook light: one call renders the three-panel summary
(accuracy-by-dataset · ANLI difficulty curve · confusion matrix) and optionally
saves it. The same function backs the figure published in docs/README.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from .nli_metrics import nli_summary, anli_round_curve, confusion_matrix


def plot_nli_summary(results, *, save_path: str | None = None, show: bool = True):
    """Render the NB05 three-panel summary. Returns the matplotlib Figure."""
    summ = nli_summary(results)
    curve = anli_round_curve(results)
    cm = confusion_matrix(results)
    clean = summ[summ["kind"] == "clean"]["accuracy"]
    clean_acc = float(clean.iloc[0]) if len(clean) else None

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    # (1) per-dataset accuracy — clean vs adversarial
    colors = ["#2E7D32" if k == "clean" else "#C62828" for k in summ["kind"]]
    ax[0].barh(summ["source"], summ["accuracy"], color=colors)
    ax[0].set_xlim(0, 1); ax[0].invert_yaxis()
    ax[0].set_title("Accuracy by dataset (green = clean baseline)")
    ax[0].set_xlabel("accuracy")
    for y, v in enumerate(summ["accuracy"]):
        ax[0].text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=9)

    # (2) ANLI difficulty curve, with the clean baseline for reference
    if not curve.empty:
        ax[1].plot(curve["round"], curve["accuracy"], "o-", color="#C62828", lw=2, ms=9)
        if clean_acc is not None:
            ax[1].axhline(clean_acc, ls="--", color="#2E7D32", lw=1.5, label="clean (MNLI)")
            ax[1].legend()
        ax[1].set_ylim(0, 1); ax[1].set_title("ANLI difficulty curve (R1→R3)")
        ax[1].set_ylabel("accuracy"); ax[1].grid(alpha=.3)
        for x, v in zip(curve["round"], curve["accuracy"]):
            ax[1].text(x, v + 0.03, f"{v:.0%}", ha="center", fontsize=9)

    # (3) confusion heatmap (drop all-zero unparsed column)
    cmp = cm.drop(columns=["unparsed"]) if cm["unparsed"].sum() == 0 else cm
    sns.heatmap(cmp, annot=True, fmt="d", cmap="Blues", ax=ax[2], cbar=False)
    ax[2].set_title("Confusion (correct → predicted)")
    ax[2].set_ylabel("correct"); ax[2].set_xlabel("predicted")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig
