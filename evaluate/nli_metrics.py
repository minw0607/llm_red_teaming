"""
evaluate/nli_metrics.py — Scoring for NLI robustness runs.

Turns a flat list of ``NLIResult`` (across MNLI / ANLI / AdvGLUE) into the
metrics a robustness assessment reports:

    nli_summary          per-source accuracy + unparsed rate
    robustness_gap       clean accuracy − adversarial accuracy (the headline)
    confusion_matrix     3×3 gold-vs-predicted, per source or overall
    anli_round_curve     accuracy by ANLI round (difficulty curve)
    error_cases          wrong predictions, with the ANLI 'reason' annotation
    print_nli_report     console summary

Everything is deterministic — labels are parsed from the model reply, no judge
LLM is involved in scoring.
"""

from __future__ import annotations

import pandas as pd

from attacks.robustness.datasets import NLI_LABELS

_CLEAN_SOURCE = "mnli"


def _d(r) -> dict:
    return r if isinstance(r, dict) else r.__dict__


def _acc(rows: list[dict]) -> float:
    scored = [r for r in rows if r["pred"] != -1 or True]  # unparsed counts as wrong
    return round(sum(1 for r in rows if r["correct"]) / len(rows), 4) if rows else 0.0


# ── Per-source summary ───────────────────────────────────────────────────────────

def nli_summary(results) -> pd.DataFrame:
    """Per-dataset accuracy, count, and unparsed-reply rate."""
    rows = [_d(r) for r in results]
    by_src: dict[str, list[dict]] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)

    out = []
    for src, rs in by_src.items():
        n = len(rs)
        correct = sum(1 for r in rs if r["correct"])
        unparsed = sum(1 for r in rs if r["pred"] == -1)
        out.append({
            "source": src,
            "kind": "clean" if src == _CLEAN_SOURCE else "adversarial",
            "n": n,
            "correct": correct,
            "accuracy": round(correct / n, 4) if n else 0.0,
            "unparsed": unparsed,
            "unparsed_rate": round(unparsed / n, 4) if n else 0.0,
        })
    df = pd.DataFrame(out)
    # clean first, then adversarial sources alphabetically
    df["_o"] = df["source"].apply(lambda s: (0, s) if s == _CLEAN_SOURCE else (1, s))
    return df.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def nli_accuracy(results, source: str | None = None) -> float:
    rows = [_d(r) for r in results if source is None or _d(r)["source"] == source]
    return _acc(rows)


# ── Robustness gap ───────────────────────────────────────────────────────────────

def robustness_gap(results, clean_source: str = _CLEAN_SOURCE) -> pd.DataFrame:
    """
    For each adversarial source, clean accuracy − adversarial accuracy.

    A large positive gap means the model handles ordinary NLI but is brittle on
    adversarially-constructed reasoning — the core robustness finding.
    """
    summ = nli_summary(results).set_index("source")
    if clean_source not in summ.index:
        clean_acc = float("nan")
    else:
        clean_acc = summ.loc[clean_source, "accuracy"]

    out = []
    for src, row in summ.iterrows():
        if src == clean_source:
            continue
        gap = round(clean_acc - row["accuracy"], 4) if clean_acc == clean_acc else None
        out.append({
            "source": src,
            "clean_acc": clean_acc,
            "adv_acc": row["accuracy"],
            "robustness_gap": gap,
            "n": int(row["n"]),
        })
    return pd.DataFrame(out)


# ── Confusion matrix ─────────────────────────────────────────────────────────────

def confusion_matrix(results, source: str | None = None) -> pd.DataFrame:
    """3×3 gold (rows) vs predicted (cols). An 'unparsed' column captures pred=-1."""
    rows = [_d(r) for r in results if source is None or _d(r)["source"] == source]
    labels = [0, 1, 2]
    names = [NLI_LABELS[i] for i in labels]
    mat = pd.DataFrame(0, index=names, columns=names + ["unparsed"])
    for r in rows:
        g = NLI_LABELS[r["gold"]]
        if r["pred"] == -1:
            mat.loc[g, "unparsed"] += 1
        else:
            mat.loc[g, NLI_LABELS[r["pred"]]] += 1
    return mat


# ── ANLI difficulty curve ────────────────────────────────────────────────────────

def anli_round_curve(results) -> pd.DataFrame:
    """Accuracy by ANLI round (R1 → R3 = increasing adversarial difficulty)."""
    summ = nli_summary(results)
    rnds = summ[summ["source"].str.startswith("anli_r")].copy()
    rnds["round"] = rnds["source"].str.replace("anli_r", "R", regex=False)
    return rnds[["round", "n", "accuracy", "unparsed_rate"]].reset_index(drop=True)


# ── Error inspection ─────────────────────────────────────────────────────────────

def error_cases(results, source: str | None = None, n: int = 8,
                include_unparsed: bool = True) -> list[dict]:
    """
    Wrong predictions for human review. Each record carries the premise,
    hypothesis, gold/predicted labels, the model reply, and — for ANLI — the
    human 'reason' annotation explaining why the item is adversarial.
    """
    rows = [_d(r) for r in results if source is None or _d(r)["source"] == source]
    errs = []
    for r in rows:
        if r["correct"]:
            continue
        if r["pred"] == -1 and not include_unparsed:
            continue
        errs.append({
            "source": r["source"],
            "premise": r["premise"],
            "hypothesis": r["hypothesis"],
            "gold": NLI_LABELS[r["gold"]],
            "predicted": NLI_LABELS.get(r["pred"], "UNPARSED"),
            "reason": r.get("reason", ""),
            "response": r.get("response", ""),
        })
    # ANLI-annotated errors first (they're the most informative), then the rest
    errs.sort(key=lambda e: (e["reason"] == "", e["source"]))
    return errs[:n]


def error_direction_summary(results, sources=None) -> pd.DataFrame:
    """
    Gold→predicted breakdown of *incorrect* items, ranked by frequency.

    Surfaces the dominant failure mode dynamically — e.g. ``entailment → neutral``
    (the model under-commits, missing a valid inference) vs. ``neutral →
    contradiction`` (it over-commits). Pass ``sources`` (a set/list) to focus on
    specific datasets, e.g. ``{'anli_r2', 'anli_r3'}``.
    """
    from collections import Counter
    rows = [_d(r) for r in results]
    if sources is not None:
        sources = set(sources)
        rows = [r for r in rows if r["source"] in sources]
    c: Counter = Counter()
    for r in rows:
        if r["correct"]:
            continue
        g = NLI_LABELS.get(r["gold"], "?")
        p = NLI_LABELS.get(r["pred"], "unparsed")
        c[(g, p)] += 1
    out = [{"gold": g, "predicted": p, "count": n} for (g, p), n in c.most_common()]
    df = pd.DataFrame(out)
    if not df.empty:
        df["pct_of_errors"] = (df["count"] / df["count"].sum()).round(3)
    return df


def explain_nli_errors(results, source: str | None = None, n: int = 6) -> None:
    """Pretty-print the most informative misclassifications."""
    errs = error_cases(results, source=source, n=n)
    if not errs:
        print("✅ No misclassifications in scope.")
        return
    print(f"🔎 {len(errs)} misclassification(s) shown "
          f"({'all sources' if source is None else source}):\n")
    for i, e in enumerate(errs, 1):
        print(f"[{i}] {e['source']}  —  gold={e['gold']} · predicted={e['predicted']}")
        print(f"    Premise:    {e['premise'][:160]}")
        print(f"    Hypothesis: {e['hypothesis'][:160]}")
        if e["reason"]:
            print(f"    Why it's hard (ANLI): {e['reason'][:200]}")
        print()


# ── Dynamic narrative helpers (keep the demo notebook light) ─────────────────────

def nli_regulatory_note(results) -> str:
    """One-paragraph regulatory read tied to the observed worst gap + failure mode."""
    gap = robustness_gap(results)
    if gap.empty:
        return "No adversarial datasets scored — regulatory read unavailable."
    w = gap.loc[gap["robustness_gap"].idxmax()]
    sev = ("material" if w["robustness_gap"] >= 0.25
           else "moderate" if w["robustness_gap"] >= 0.10 else "minor")
    adv_dir = error_direction_summary(results, sources={"anli_r2", "anli_r3"})
    fm = f"{adv_dir.iloc[0]['gold']} → {adv_dir.iloc[0]['predicted']}" if not adv_dir.empty else "mixed"
    return (
        f"📋 Regulatory read: worst gap is {w['source']} at {w['robustness_gap']:+.0%} ({sev}); "
        f"dominant failure mode is {fm}.\n"
        f"   → NIST AI 600-1 §2.5 Information Integrity and EU AI Act Art. 15 (accuracy/robustness) "
        f"are the primary obligations;\n"
        f"     document the gap + remediation in the model risk register."
    )


def print_nli_takeaways(results) -> None:
    """Dynamic, plain-text headline summary of a run (used in the executive step)."""
    summ = nli_summary(results)
    gap = robustness_gap(results)
    curve = anli_round_curve(results)
    clean = summ[summ["kind"] == "clean"]["accuracy"]
    clean_acc = float(clean.iloc[0]) if len(clean) else float("nan")
    adv = summ[summ["kind"] == "adversarial"]
    worst = gap.loc[gap["robustness_gap"].idxmax()] if not gap.empty else None
    adv_dir = error_direction_summary(results, sources={"anli_r2", "anli_r3"})

    print("── NLI ROBUSTNESS — KEY TAKEAWAYS ──")
    print(f"  Clean accuracy (MNLI)        : {clean_acc:.1%}")
    print(f"  Adversarial accuracy (mean)  : {adv['accuracy'].mean():.1%}  over {len(adv)} adversarial set(s)")
    if worst is not None:
        print(f"  Worst robustness gap         : {worst['source']}  {worst['robustness_gap']:+.1%} "
              f"(clean {worst['clean_acc']:.0%} → adv {worst['adv_acc']:.0%})")
    if not curve.empty:
        trend = "falls" if curve["accuracy"].iloc[0] > curve["accuracy"].iloc[-1] else "holds/rises"
        print(f"  ANLI curve R1→R3             : {' → '.join(f'{v:.0%}' for v in curve['accuracy'])}  ({trend} with difficulty)")
    if not adv_dir.empty:
        t = adv_dir.iloc[0]
        print(f"  Dominant failure mode        : {t['gold']} → {t['predicted']}  ({t['pct_of_errors']:.0%} of hard-round misses)")
    unparsed = sum(1 for r in results if _d(r)["pred"] == -1)
    print(f"  Unparseable replies          : {unparsed} / {len(results)}  (content-filtered; counted as wrong)")

    big_gap = gap[gap["robustness_gap"] >= 0.25] if not gap.empty else gap
    print()
    if not gap.empty and not big_gap.empty:
        print(f"⚠️  {len(big_gap)} dataset(s) show a robustness gap ≥ 25% — clean accuracy materially overstates")
        print("   reliability on adversarial reasoning. Add adversarial NLI to the eval suite and track the gap")
        print("   as a release gate (NIST Information Integrity · EU AI Act Art. 15). See Step 7 for failure modes.")
    else:
        print("✅ No dataset shows a robustness gap ≥ 25%. Reasoning degrades gracefully under adversarial")
        print("   pressure; continue monitoring the ANLI curve as harder rounds / new benchmarks are added.")


# ── Console report ───────────────────────────────────────────────────────────────

def print_nli_report(results) -> None:
    summ = nli_summary(results)
    gap = robustness_gap(results)
    total = len(results)
    overall_acc = nli_accuracy(results)

    print("=" * 64)
    print("  NLI ROBUSTNESS REPORT")
    print("=" * 64)
    print(f"  Total items: {total}   ·   Overall accuracy: {overall_acc:.2%}\n")

    print("  Per-dataset accuracy")
    print("  " + "-" * 58)
    for r in summ.itertuples(index=False):
        tag = "clean " if r.kind == "clean" else "adv.  "
        print(f"   {tag} {r.source:26s} n={r.n:4d}  acc={r.accuracy:6.2%}"
              f"  unparsed={r.unparsed_rate:5.1%}")

    if not gap.empty:
        print("\n  Robustness gap  (clean acc − adversarial acc; higher = more brittle)")
        print("  " + "-" * 58)
        for r in gap.itertuples(index=False):
            g = r.robustness_gap
            flag = "🔴" if g is not None and g >= 0.25 else "🟠" if g is not None and g >= 0.10 else "🟢"
            print(f"   {flag} {r.source:26s} adv={r.adv_acc:6.2%}  gap={g:+.2%}")

    curve = anli_round_curve(results)
    if not curve.empty:
        print("\n  ANLI difficulty curve")
        print("  " + "-" * 58)
        for r in curve.itertuples(index=False):
            print(f"   {r.round}  n={r.n:4d}  acc={r.accuracy:6.2%}")
    print("=" * 64)
