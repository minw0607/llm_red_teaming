"""
evaluate/display.py — Notebook-friendly display helpers for adversarial evaluation.

Functions
---------
display_human_review   : Flag high-risk adversarial examples, print a summary,
                         and render styled DataFrames in the notebook.
                         Returns the flagged DataFrame for further analysis.

Usage
-----
    from evaluate.display import display_human_review

    review_df = display_human_review(
        results_df,
        stealth_threshold=STEALTH_THRESHOLD,
        top_n=20,
    )
"""

from __future__ import annotations

import pandas as pd

from .metrics import flag_human_review


def display_human_review(
    results_df: pd.DataFrame,
    stealth_threshold: float = 0.80,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Flag, display, and return the human review queue.

    Calls ``flag_human_review()`` to identify adversarial cases that warrant
    manual inspection, then renders:

    * A console summary of priority counts (HIGH / MEDIUM / LOW).
    * A styled DataFrame of HIGH-priority cases (flipped AND stealthy).
    * A compact styled table of all flagged cases (up to *top_n* rows).

    Parameters
    ----------
    results_df : pd.DataFrame
        Long-form output from ``run_all_attacks()``.
    stealth_threshold : float
        Minimum stealth/semantic similarity for a flipped case to be rated
        HIGH priority (default 0.80).
    top_n : int
        Maximum number of rows shown in the compact all-flagged table
        (default 20).

    Returns
    -------
    pd.DataFrame
        Full flagged DataFrame with a ``review_priority`` column
        (HIGH / MEDIUM / LOW).  Sorted HIGH-first, then by semantic_sim desc.
    """
    try:
        from IPython.display import display as ipy_display
    except ImportError:
        # Fallback: plain print when not in a Jupyter environment
        ipy_display = print  # type: ignore[assignment]

    review_df = flag_human_review(results_df, stealth_threshold=stealth_threshold)

    # ── Priority summary ───────────────────────────────────────────────────────
    priority_counts = review_df["review_priority"].value_counts()
    print(f"Human review queue: {len(review_df)} cases flagged")
    for priority, icon in [("HIGH", "🔴"), ("MEDIUM", "🟠"), ("LOW", "🟡")]:
        count = priority_counts.get(priority, 0)
        print(f"  {icon} {priority:8s} : {count}")

    # Warn if any MEDIUM cases are due to model non-determinism
    if "text_changed" in review_df.columns:
        nondeterminism = review_df[
            (review_df["review_priority"] == "MEDIUM") & (~review_df["text_changed"])
        ]
        if len(nondeterminism):
            print(f"\n  ℹ️  {len(nondeterminism)} MEDIUM case(s) flagged due to model non-determinism")
            print(f"     (text_changed=False but model gave different answers on two calls)")
            print(f"     These reveal decision-boundary instability, not a true attack success.")

    # ── HIGH priority detail table ────────────────────────────────────────────
    high_priority = review_df[review_df["review_priority"] == "HIGH"]
    sim_col = "composite_stealth" if "composite_stealth" in review_df.columns else "semantic_sim"

    if len(high_priority):
        print(f"\n🔴 HIGH priority — flipped AND stealth ≥ {stealth_threshold}:")
        display_cols = [
            "attack", "original_text", "attacked_text",
            "label", "original_pred", "attacked_pred", sim_col,
        ]
        display_cols = [c for c in display_cols if c in high_priority.columns]
        fmt = {sim_col: "{:.3f}"}
        ipy_display(
            high_priority[display_cols]
            .style
            .set_properties(**{"text-align": "left"})
            .format(fmt, na_rep="—")
            .set_caption("🔴 HIGH Priority: Successful & Stealthy Adversarial Examples")
        )
    else:
        print(
            "\n✅ No HIGH-priority cases — model was robust to stealthy attacks "
            "at this sample size."
        )

    # ── Compact all-flagged table ──────────────────────────────────────────────
    if len(review_df):
        print(f"\nAll flagged cases (top {min(top_n, len(review_df))}):")
        compact_cols = [
            "attack", "review_priority", "label",
            "original_pred", "attacked_pred",
            sim_col, "original_text", "attacked_text",
        ]
        compact_cols = [c for c in compact_cols if c in review_df.columns]
        fmt_compact = {sim_col: "{:.3f}"}
        ipy_display(
            review_df[compact_cols]
            .head(top_n)
            .style
            .format(fmt_compact, na_rep="—")
            .background_gradient(subset=[sim_col], cmap="YlOrRd", axis=0)
            .set_caption(f"Human Review Queue (top {min(top_n, len(review_df))})")
        )

    return review_df
