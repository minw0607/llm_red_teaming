"""
evaluate/metrics.py — Standardised red-teaming evaluation metrics.

Functions
---------
accuracy_drop        : Original accuracy minus attacked accuracy (NLP attacks)
attack_success_rate  : Fraction of responses judged as "violation" (jailbreak)
verdict_summary      : Verdict counts and percentages from jailbreak runs
adversarial_report   : Full summary table for an NLP adversarial experiment
jailbreak_report     : Full summary table for a jailbreak experiment
"""

from __future__ import annotations

from collections import Counter

import pandas as pd


# ── NLP adversarial metrics ───────────────────────────────────────────────────

def accuracy_drop(
    original_correct: int,
    attacked_correct: int,
    n_samples: int,
) -> dict[str, float]:
    """
    Compute accuracy on original vs attacked inputs and the resulting drop.

    Parameters
    ----------
    original_correct : int
        Number of correctly classified original samples.
    attacked_correct : int
        Number of correctly classified attacked samples.
    n_samples : int
        Total number of samples evaluated.

    Returns
    -------
    dict with keys: ``original_acc``, ``attacked_acc``, ``acc_drop``
    """
    orig_acc = original_correct / n_samples if n_samples else 0.0
    atk_acc  = attacked_correct / n_samples if n_samples else 0.0
    return {
        "original_acc": round(orig_acc, 4),
        "attacked_acc": round(atk_acc, 4),
        "acc_drop":     round(orig_acc - atk_acc, 4),
    }


def adversarial_report(
    results_by_attack: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """
    Build a comparison DataFrame across multiple attack methods.

    Parameters
    ----------
    results_by_attack : dict[str, dict[str, float]]
        Mapping of attack_name → metrics dict (from ``accuracy_drop``).

    Returns
    -------
    pd.DataFrame
        Columns: attack, original_acc, attacked_acc, acc_drop
    """
    rows = []
    for attack_name, metrics in results_by_attack.items():
        rows.append({"attack": attack_name, **metrics})
    df = pd.DataFrame(rows).sort_values("acc_drop", ascending=False)
    return df.reset_index(drop=True)


# ── Jailbreak metrics ─────────────────────────────────────────────────────────

def attack_success_rate(results: list) -> float:
    """
    Fraction of responses judged as "violation".

    Parameters
    ----------
    results : list
        List of result objects / dicts with a ``verdict`` field.

    Returns
    -------
    float  (0.0 – 1.0)
    """
    if not results:
        return 0.0
    violations = sum(
        1 for r in results
        if (r["verdict"] if isinstance(r, dict) else r.verdict) == "violation"
    )
    return round(violations / len(results), 4)


def verdict_summary(results: list) -> pd.DataFrame:
    """
    Tabulate verdict counts and percentages.

    Parameters
    ----------
    results : list
        List of result objects / dicts with a ``verdict`` field.

    Returns
    -------
    pd.DataFrame  with columns: verdict, count, pct
    """
    verdicts = [
        r["verdict"] if isinstance(r, dict) else r.verdict
        for r in results
    ]
    counts = Counter(verdicts)
    total = len(verdicts)
    rows = [
        {"verdict": v, "count": c, "pct": round(c / total * 100, 1)}
        for v, c in counts.most_common()
    ]
    return pd.DataFrame(rows)


def jailbreak_report(results: list, title: str = "Jailbreak Evaluation") -> None:
    """
    Print a formatted summary of a jailbreak run to stdout.

    Parameters
    ----------
    results : list
        List of JBBResult or ArtifactResult objects (or dicts).
    title : str
        Header label for the report.
    """
    n = len(results)
    asr = attack_success_rate(results)
    summary_df = verdict_summary(results)

    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"  Total evaluated  : {n}")
    print(f"  Attack Success Rate (ASR): {asr:.2%}")
    print(f"\n  Verdict Breakdown:")
    for _, row in summary_df.iterrows():
        bar = "█" * int(row["pct"] / 5)
        print(f"    {row['verdict']:12s}  {row['count']:4d}  ({row['pct']:5.1f}%)  {bar}")
    print(f"{'='*55}\n")
