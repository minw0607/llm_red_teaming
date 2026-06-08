"""
evaluate/plots.py — Reusable visualisation functions for adversarial NLP evaluation.

Functions
---------
plot_adversarial_summary   : Two-panel figure (bar chart + risk matrix), stacked vertically.
                             Returns a matplotlib Figure; optionally saves to disk.

Customisable parameters
-----------------------
All user-tunable settings (colors, thresholds, figure size, save path) are exposed
as keyword arguments with sensible defaults.  The notebook cells stay as one-liners;
callers only need to pass what they want to change.

Usage
-----
    from evaluate.plots import plot_adversarial_summary

    fig = plot_adversarial_summary(
        summary_df,
        dataset_name=DATASET,
        model_name=target.model,
        n_samples=N_SAMPLES,
        attack_suite=ATTACK_SUITE,
        stealth_mode=STEALTH_MODE,
        stealth_threshold=STEALTH_THRESHOLD,
        save_path='../results/01_risk_matrix.png',
    )
"""

from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Default level → colour mapping ────────────────────────────────────────────

DEFAULT_LEVEL_COLORS: dict[str, str] = {
    "character":  "#E84393",   # hot-pink
    "word":       "#FF7F0E",   # orange
    "sentence":   "#2CA02C",   # green
    "semantic":   "#1F77B4",   # blue
    "structural": "#9467BD",   # purple
    "unknown":    "#7F7F7F",   # grey
}

# Vertical offset patterns for banded label staggering (points).
# Key = number of attacks in a band; value = list of dy offsets (top → bottom).
_STAGGER: dict[int, list[int]] = {
    1: [0],
    2: [22, -22],
    3: [30,  0, -30],
    4: [38, 14, -14, -38],
    5: [46, 22,  0, -22, -46],
}


# ── Public API ─────────────────────────────────────────────────────────────────

def plot_adversarial_summary(
    summary_df: pd.DataFrame,
    dataset_name:      str   = "",
    model_name:        str   = "",
    n_samples:         int | None = None,
    attack_suite:      str   = "",
    stealth_mode:      str   = "composite",
    stealth_threshold: float = 0.80,
    save_path:         str | None = "../results/01_risk_matrix.png",
    figsize:           tuple = (14, 13),
    dpi:               int   = 150,
    level_colors:      dict | None = None,
) -> plt.Figure:
    """
    Render two adversarial evaluation charts stacked vertically:

    * **Top** — Grouped accuracy bar chart: original vs attacked accuracy,
      sorted by accuracy drop descending.  Drop % annotated above each pair.
    * **Bottom** — Risk matrix bubble chart: impact (y) × stealth (x),
      bubble size = ASR.  Labels use colour-coded arrow annotations with
      banded vertical staggering to avoid overlaps.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of ``compute_attack_summary()``.
    dataset_name : str
        Dataset identifier shown in the figure title (e.g. ``"sst2"``).
    model_name : str
        Target model name shown in the figure title.
    n_samples : int | None
        Sample count shown in the figure title.
    attack_suite : str
        Suite name shown in the figure title (e.g. ``"extended"``).
    stealth_mode : str
        ``"composite"`` or ``"semantic_only"`` — used in the risk-matrix
        x-axis label.
    stealth_threshold : float
        Stealth score ≥ this value → shaded danger zone (default 0.80).
    save_path : str | None
        File path for saving the figure.  Creates parent dirs automatically.
        Pass ``None`` to skip saving.
    figsize : tuple
        ``(width, height)`` in inches.  Default ``(14, 13)``.
    dpi : int
        Output resolution for the saved file (default 150).
    level_colors : dict | None
        Override any subset of ``DEFAULT_LEVEL_COLORS``.
        Keys are level strings (e.g. ``"word"``), values are hex colours.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure (already displayed if in a Jupyter environment).
    """
    colors = {**DEFAULT_LEVEL_COLORS, **(level_colors or {})}

    fig, (ax_bar, ax_risk) = plt.subplots(2, 1, figsize=figsize)

    _draw_accuracy_bars(ax_bar, summary_df)
    _draw_risk_matrix(
        ax_risk, summary_df,
        colors=colors,
        stealth_mode=stealth_mode,
        stealth_threshold=stealth_threshold,
    )

    # ── Shared title ──────────────────────────────────────────────────────────
    parts = [p for p in [dataset_name, model_name] if p]
    title = "Adversarial NLP Evaluation"
    if parts:
        title += f" — {' / '.join(parts)}"
    if n_samples is not None or attack_suite:
        meta = []
        if n_samples is not None:
            meta.append(f"n={n_samples}")
        if attack_suite:
            meta.append(f"{attack_suite} suite")
        title += f"  ({', '.join(meta)})"

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✅ Saved → {save_path}")

    return fig


# ── Private drawing helpers ────────────────────────────────────────────────────

def _draw_accuracy_bars(ax: plt.Axes, summary_df: pd.DataFrame) -> None:
    """Grouped bar chart: original vs attacked accuracy, sorted by acc_drop."""
    sorted_df = summary_df.sort_values("acc_drop", ascending=False)
    x     = range(len(sorted_df))
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x], sorted_df["original_acc"],
        width, label="Original", color="#4C72B0", alpha=0.85,
    )
    ax.bar(
        [i + width / 2 for i in x], sorted_df["attacked_acc"],
        width, label="Attacked", color="#DD8452", alpha=0.85,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(sorted_df["attack"], rotation=28, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylim(0, 1.12)
    ax.set_title("Original vs Attacked Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.axhline(1.0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)

    for i, (_, row) in enumerate(sorted_df.iterrows()):
        drop = row["acc_drop"]
        if drop > 0:
            ax.annotate(
                f"−{drop:.0%}",
                xy=(i, row["attacked_acc"] + 0.01),
                ha="center", va="bottom", fontsize=7.5,
                color="crimson", fontweight="bold",
            )


def _draw_risk_matrix(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    colors: dict,
    stealth_mode: str,
    stealth_threshold: float,
) -> None:
    """Risk matrix bubble chart with banded-stagger label placement."""
    if not summary_df["stealth_score"].notna().any():
        ax.text(0.5, 0.5, "Stealth scores not available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="grey")
        return

    plot_df = summary_df.dropna(subset=["stealth_score", "acc_drop"]).copy()

    # ── Bubbles ───────────────────────────────────────────────────────────────
    for _, row in plot_df.iterrows():
        c  = colors.get(row["level"], "#7F7F7F")
        sz = 300 + row["asr"] * 2000
        ax.scatter(
            row["stealth_score"], row["acc_drop"],
            s=sz, color=c, alpha=0.75,
            edgecolors="white", linewidth=1.5, zorder=3,
        )

    # ── Banded stagger label placement ───────────────────────────────────────
    # Group attacks by acc_drop band, sort each band left→right by stealth,
    # then fan labels evenly above/below the band to eliminate overlaps.
    _stealth = plot_df.set_index("attack")["stealth_score"].to_dict()
    bands: dict[float, list[str]] = {}
    for _, row in plot_df.iterrows():
        k = round(row["acc_drop"], 3)
        bands.setdefault(k, []).append(row["attack"])
    for k in bands:
        bands[k].sort(key=lambda a: _stealth[a])

    _offset: dict[str, tuple[int, int]] = {}
    for names in bands.values():
        n       = len(names)
        dy_list = _STAGGER.get(n, [i * 20 - (n - 1) * 10 for i in range(n)])
        for name, dy in zip(names, dy_list):
            _offset[name] = (14, dy)

    for _, row in plot_df.iterrows():
        c      = colors.get(row["level"], "#7F7F7F")
        dx, dy = _offset.get(row["attack"], (14, 0))
        ax.annotate(
            row["attack"],
            xy         = (row["stealth_score"], row["acc_drop"]),
            xytext     = (dx, dy),
            textcoords = "offset points",
            fontsize   = 9,
            fontweight = "semibold",
            ha="left", va="center",
            bbox       = dict(boxstyle="round,pad=0.28", fc="white",
                              ec=c, lw=1.1, alpha=0.93),
            arrowprops = dict(arrowstyle="->", color=c, lw=0.9,
                              connectionstyle="arc3,rad=0.15"),
            zorder=5,
        )

    # ── Danger zone ───────────────────────────────────────────────────────────
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(bottom=-0.005)
    ax.axvspan(stealth_threshold, 1.05, alpha=0.06, color="red", zorder=0)
    ax.axhline(0.05, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axvline(stealth_threshold, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(
        stealth_threshold + 0.006, ax.get_ylim()[1] * 0.97,
        "⚠ Danger zone", color="red", fontsize=8.5, va="top", fontweight="bold",
    )

    patches = [
        mpatches.Patch(color=c, label=lvl.capitalize())
        for lvl, c in colors.items()
        if lvl in plot_df["level"].values
    ]
    ax.legend(handles=patches, title="Level", fontsize=8.5,
              title_fontsize=9, loc="upper left", framealpha=0.9)

    ax.set_xlabel("Stealth Score", fontsize=10)
    ax.set_ylabel("Accuracy Drop (impact)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    stealth_label = "composite stealth" if stealth_mode == "composite" else "semantic similarity"
    ax.set_title(
        f"Risk Matrix: Impact × Stealth\n(x = {stealth_label}, bubble size = ASR)",
        fontsize=11, fontweight="bold",
    )
