"""
evaluate/adversarial_eval.py — End-to-end adversarial evaluation runner.

Wraps the full attack → target → score loop into a single callable,
returning a tidy long-form DataFrame for downstream analysis and reporting.

Checkpointing
-------------
Pass ``checkpoint_path`` to save results after each attack completes.
If the kernel dies mid-run, restart the cell — completed attacks are
reloaded from the checkpoint and skipped automatically.

Usage
-----
    from evaluate.adversarial_eval import run_all_attacks
    from attacks.character import TextBugger
    from targets.azure_openai import AzureOpenAITarget

    results_df = run_all_attacks(
        attacks         = {"TextBugger": TextBugger(seed=42)},
        target          = AzureOpenAITarget(),
        eval_df         = dev_df,
        n_samples       = 50,
        checkpoint_path = "../results/01_checkpoint_n50.csv",  # resumable
    )
"""

from __future__ import annotations

import gc
import os
import re
import time

import pandas as pd
from tqdm.auto import tqdm


# ── Constants ──────────────────────────────────────────────────────────────────

ATTACK_LEVELS: dict[str, str] = {
    "TextBugger":        "character",
    "DeepWordBug":       "character",
    "TextFooler":        "word",
    "BERTAttack":        "word",
    "CheckList":         "sentence",
    "StressTest":        "sentence",
    "SemanticAttack":    "semantic",
    "Homoglyph":         "structural",
    "BackTranslation":   "structural",
    "NegationInjection": "structural",
}

_SCHEMA = [
    "attack", "level", "idx", "original_text", "attacked_text",
    "text_changed", "label", "original_pred", "attacked_pred",
    "orig_correct", "atk_correct", "flipped", "semantic_sim",
]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Collapse internal whitespace and strip edges for text_changed comparison."""
    return re.sub(r"\s+", " ", s).strip()


def _load_checkpoint(path: str) -> tuple[set[str], list[dict]]:
    """
    Load an existing checkpoint CSV.

    Returns
    -------
    (completed_attacks, records)
        completed_attacks — set of attack names that are fully saved
        records           — list of dicts ready for pd.DataFrame()
    """
    if not os.path.exists(path):
        return set(), []
    try:
        df = pd.read_csv(path, dtype=str)
        # Restore boolean / numeric columns
        bool_cols = ["text_changed", "orig_correct", "atk_correct", "flipped"]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].map({"True": True, "False": False}).fillna(False)
        num_cols = ["idx", "semantic_sim"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        completed = set(df["attack"].unique())
        print(f"  📂 Checkpoint loaded: {len(completed)} attack(s) complete "
              f"({len(df):,} rows) — skipping: {sorted(completed)}")
        return completed, df.to_dict("records")
    except Exception as e:
        print(f"  ⚠️  Could not read checkpoint ({e}) — starting fresh.")
        return set(), []


def _save_checkpoint(path: str, records: list[dict]) -> None:
    """Write current records to the checkpoint CSV (overwrite)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


# ── Main evaluation loop ───────────────────────────────────────────────────────

def run_all_attacks(
    attacks: dict,
    target,
    eval_df: pd.DataFrame,
    n_samples: int = 50,
    sleep_sec: float = 1.2,
    encoder=None,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """
    Run every attack in *attacks* against the target model on *n_samples*
    rows of *eval_df* and return a long-form per-sample results DataFrame.

    Parameters
    ----------
    attacks : dict[str, Attack]
        Mapping of attack name → attack instance.
    target : OpenAICompatibleTarget
        Model connector with a ``get_sentiment(text) -> str`` method.
    eval_df : pd.DataFrame
        Dataset with columns ``sentence`` and ``label``.
    n_samples : int
        Number of rows to evaluate (default 50).
    sleep_sec : float
        Delay between API calls (default 1.2 s).
    encoder : SentenceTransformer | None
        Optional encoder for semantic similarity. When provided,
        ``semantic_sim`` is populated for rows where text changed.
    checkpoint_path : str | None
        Path to a CSV checkpoint file.  When provided:
        - Results are saved after each attack completes.
        - On restart, completed attacks are reloaded and skipped.
        - Recommended for n ≥ 200 or when using expensive attacks.
        Example: ``"../results/01_checkpoint_n872.csv"``

    Returns
    -------
    pd.DataFrame
        Columns: attack, level, idx, original_text, attacked_text,
        text_changed, label, original_pred, attacked_pred,
        orig_correct, atk_correct, flipped, semantic_sim
    """
    # ── Load checkpoint (if any) ───────────────────────────────────────────────
    completed_attacks: set[str] = set()
    records: list[dict] = []

    if checkpoint_path:
        completed_attacks, records = _load_checkpoint(checkpoint_path)

    sample_df = eval_df.head(n_samples).reset_index(drop=True)
    total_attacks  = len(attacks)
    done_count     = len(completed_attacks)

    for atk_num, (attack_name, atk) in enumerate(attacks.items(), start=1):
        # ── Skip if already checkpointed ──────────────────────────────────────
        if attack_name in completed_attacks:
            continue

        level = ATTACK_LEVELS.get(attack_name, "unknown")
        remaining = total_attacks - done_count - (atk_num - 1 - done_count)
        print(f"\n{'═'*56}")
        print(f"  [{atk_num}/{total_attacks}] {attack_name:<22} level={level}  n={n_samples}")
        if checkpoint_path:
            print(f"  💾 Checkpoint active — progress will be saved after this attack")
        print(f"{'═'*56}")

        attack_records: list[dict] = []

        for idx, row in tqdm(sample_df.iterrows(), total=n_samples, leave=False):
            text  = row["sentence"]
            label = "positive" if row["label"] == 1 else "negative"

            # ── Original prediction ────────────────────────────────────────────
            try:
                orig_pred = target.get_sentiment(text)
            except Exception:
                orig_pred = "error"

            # ── Attack + attacked prediction ───────────────────────────────────
            try:
                attacked = atk.attack(text)
                atk_pred = target.get_sentiment(attacked)
            except Exception:
                attacked = text
                atk_pred = "error"

            # Normalise whitespace before comparing — prevents trailing-space
            # differences (e.g. BERTAttack) from being counted as text changes.
            text_changed = (_norm(attacked) != _norm(text))

            # ── Semantic similarity ────────────────────────────────────────────
            sem_sim: float | None = None
            if encoder and text_changed:
                try:
                    from sentence_transformers import util
                    e1 = encoder.encode(text,     convert_to_tensor=True)
                    e2 = encoder.encode(attacked, convert_to_tensor=True)
                    sem_sim = round(util.cos_sim(e1, e2).item(), 4)
                except Exception:
                    pass

            orig_correct = (orig_pred == label)
            atk_correct  = (atk_pred  == label)
            flipped      = orig_correct and not atk_correct and atk_pred != "error"

            status = (
                "🔴 FLIPPED" if flipped
                else "⚠️  ERROR  " if "error" in (orig_pred, atk_pred)
                else "✅        "
            )
            sim_part = f"  sim={sem_sim:.3f}" if sem_sim is not None else ""
            tqdm.write(
                f"  [{idx+1:3d}] {status}  "
                f"orig={orig_pred:8s}  atk={atk_pred:8s}{sim_part}"
            )

            attack_records.append({
                "attack":        attack_name,
                "level":         level,
                "idx":           int(idx),
                "original_text": text,
                "attacked_text": attacked,
                "text_changed":  text_changed,
                "label":         label,
                "original_pred": orig_pred,
                "attacked_pred": atk_pred,
                "orig_correct":  orig_correct,
                "atk_correct":   atk_correct,
                "flipped":       flipped,
                "semantic_sim":  sem_sim,
            })

            time.sleep(sleep_sec)

        # ── Attack complete: checkpoint + memory cleanup ────────────────────────
        records.extend(attack_records)
        flips = sum(r["flipped"] for r in attack_records)
        print(f"  ✅ {attack_name} done — {flips} flip(s) / {n_samples} samples")

        if checkpoint_path:
            _save_checkpoint(checkpoint_path, records)
            print(f"  💾 Checkpoint saved → {checkpoint_path}  "
                  f"({len(records):,} rows total)")

        # Free any memory the attack model may be holding between attacks.
        # This matters most for BERTAttack and BackTranslation which load
        # large transformer models that would otherwise stay resident.
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    return pd.DataFrame(records)
