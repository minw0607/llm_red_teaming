"""
evaluate/adversarial_eval.py — End-to-end adversarial evaluation runner.

Wraps the full attack → target → score loop into a single callable,
returning a tidy long-form DataFrame for downstream analysis and reporting.

Usage
-----
    from evaluate.adversarial_eval import run_all_attacks
    from attacks.character import TextBugger
    from targets.azure_openai import AzureOpenAITarget

    results_df = run_all_attacks(
        attacks={"TextBugger": TextBugger(seed=42)},
        target=AzureOpenAITarget(),
        eval_df=dev_df,
        n_samples=50,
    )
"""

from __future__ import annotations

import time
import pandas as pd
from tqdm.auto import tqdm


def run_all_attacks(
    attacks: dict,
    target,
    eval_df: pd.DataFrame,
    n_samples: int = 50,
    sleep_sec: float = 1.2,
    encoder=None,
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
        Dataset with columns ``sentence`` and ``label`` (1=positive, 0=negative).
    n_samples : int
        Number of rows to evaluate (default 50).
    sleep_sec : float
        Delay between API calls to avoid rate limiting (default 1.2 s).
    encoder : SentenceTransformer | None
        Optional encoder for computing semantic similarity between original
        and attacked text.  When provided, ``semantic_sim`` is populated for
        every row where the text actually changed.

    Returns
    -------
    pd.DataFrame
        Columns:
        ``attack``, ``level``, ``idx``, ``original_text``, ``attacked_text``,
        ``text_changed``, ``label``, ``original_pred``, ``attacked_pred``,
        ``orig_correct``, ``atk_correct``, ``flipped``, ``semantic_sim``
    """
    # Attack → level mapping for richer reporting
    ATTACK_LEVELS = {
        "TextBugger":     "character",
        "DeepWordBug":    "character",
        "TextFooler":     "word",
        "BERTAttack":     "word",
        "CheckList":      "sentence",
        "StressTest":     "sentence",
        "SemanticAttack": "semantic",
    }

    sample_df = eval_df.head(n_samples).reset_index(drop=True)
    records: list[dict] = []

    for attack_name, atk in attacks.items():
        level = ATTACK_LEVELS.get(attack_name, "unknown")
        print(f"\n{'═'*54}")
        print(f"  {attack_name:<20}  level={level}  n={n_samples}")
        print(f"{'═'*54}")

        for idx, row in tqdm(sample_df.iterrows(), total=n_samples, leave=False):
            text  = row["sentence"]
            label = "positive" if row["label"] == 1 else "negative"

            # ── original prediction ────────────────────────────────────────
            try:
                orig_pred = target.get_sentiment(text)
            except Exception as e:
                orig_pred = "error"

            # ── attack + prediction ────────────────────────────────────────
            try:
                attacked = atk.attack(text)
                atk_pred = target.get_sentiment(attacked)
            except Exception as e:
                attacked = text
                atk_pred = "error"

            text_changed = (attacked != text)

            # ── semantic similarity (optional) ─────────────────────────────
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

            if flipped:
                status = "🔴 FLIPPED"
            elif "error" in (orig_pred, atk_pred):
                status = "⚠️  ERROR  "
            else:
                status = "✅        "

            tqdm.write(
                f"  [{idx+1:3d}] {status}  "
                f"orig={orig_pred:8s}  atk={atk_pred:8s}  "
                f"sim={sem_sim:.3f}" if sem_sim is not None
                else f"  [{idx+1:3d}] {status}  "
                f"orig={orig_pred:8s}  atk={atk_pred:8s}"
            )

            records.append({
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

    return pd.DataFrame(records)
