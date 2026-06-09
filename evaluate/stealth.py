"""
evaluate/stealth.py — Composite stealth scoring for adversarial attacks.

Problem with a single stealth metric
--------------------------------------
Using semantic cosine similarity alone misses two important imperceptibility
dimensions:

  1. Linguistic naturalness  — does the attacked text read like real English?
     A homoglyph attack has cosine sim ≈ 1.0 but GPT-2 perplexity spikes
     dramatically because the tokeniser sees unknown Unicode codepoints.

  2. Surface edit distance   — how much did the text actually change?
     A back-translation that rewrites the whole sentence may have high cosine
     sim but very low character-level overlap.

Composite stealth score
-----------------------
  composite = α·sem_sim + β·naturalness + γ·edit_sim

Where:
  sem_sim      — SentenceTransformer cosine similarity (0–1)
  naturalness  — min(1, ppl_original / ppl_attacked)  i.e. 1 / ppl_ratio,
                 capped at 1.  Measures how much MORE unnatural the attacked
                 text is relative to the original.
  edit_sim     — difflib SequenceMatcher ratio (0–1).  1 = identical bytes,
                 0 = completely different.

Default weights: α=0.5, β=0.3, γ=0.2
  Semantic similarity carries the most weight because human reviewers
  primarily judge meaning, not byte sequences.  Perplexity is second
  because it captures tokeniser-level anomalies.  Edit distance is a
  useful tiebreaker.

Usage
-----
  from evaluate.stealth import load_perplexity_model, add_stealth_components

  ppl_model, ppl_tokenizer = load_perplexity_model()          # ~500 MB, once
  results_df = add_stealth_components(results_df,
                                      ppl_model, ppl_tokenizer)
  # results_df now has: ppl_original, ppl_attacked, ppl_ratio,
  #                     edit_sim, composite_stealth
"""

from __future__ import annotations

import math
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

# Default composite weights (α, β, γ)
DEFAULT_WEIGHTS: tuple[float, float, float] = (0.5, 0.3, 0.2)


# ── Perplexity helpers ────────────────────────────────────────────────────────

def load_perplexity_model(model_id: str = "gpt2"):
    """
    Load a GPT-2 language model for perplexity scoring.

    Parameters
    ----------
    model_id : str
        Any causal LM available on HuggingFace Hub.
        ``"gpt2"`` (117 M params) is fast and sufficient for relative
        perplexity comparisons.

    Returns
    -------
    tuple
        (model, tokenizer) — pass both to ``add_stealth_components``.
    """
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    except ImportError as e:
        raise ImportError("torch and transformers are required for perplexity scoring.") from e

    print(f"Loading perplexity model: {model_id} …")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    model     = GPT2LMHeadModel.from_pretrained(model_id)
    model.eval()
    print(f"✅ Perplexity model loaded: {model_id}")
    return model, tokenizer


def compute_perplexity(text: str, model, tokenizer, max_length: int = 512) -> float:
    """
    Compute the GPT-2 perplexity of *text*.

    Returns ``float("nan")`` for texts that are too short to score reliably
    (< 2 tokens after encoding).

    Parameters
    ----------
    text : str
    model : GPT2LMHeadModel
    tokenizer : GPT2TokenizerFast
    max_length : int
        Truncation length (default 512 — well within GPT-2's 1024 context).
    """
    import torch

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"]

    if input_ids.shape[1] < 2:
        return float("nan")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    loss = outputs.loss.item()
    return math.exp(loss)


# ── Edit-distance helper ──────────────────────────────────────────────────────

def edit_similarity(s1: str, s2: str) -> float:
    """
    Character-level similarity between *s1* and *s2* using difflib.

    Returns the SequenceMatcher ratio: 2·M / T where M = number of matching
    characters and T = total characters in both strings.
    Range: 0.0 (completely different) … 1.0 (identical).

    Requires no external dependencies (stdlib difflib).
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1, s2).ratio()


# ── Composite stealth ─────────────────────────────────────────────────────────

def composite_stealth_score(
    sem_sim: float,
    ppl_original: float,
    ppl_attacked: float,
    edit_sim: float,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> dict[str, float]:
    """
    Compute all stealth components and the weighted composite.

    Parameters
    ----------
    sem_sim : float
        Cosine similarity from SentenceTransformer (0–1).
    ppl_original : float
        GPT-2 perplexity of the original text.
    ppl_attacked : float
        GPT-2 perplexity of the attacked text.
    edit_sim : float
        Character-level edit similarity (0–1).
    weights : tuple
        (α, β, γ) weights summing to 1 (default 0.5, 0.3, 0.2).

    Returns
    -------
    dict with keys:
        ppl_ratio       — ppl_attacked / ppl_original  (> 1 = less natural)
        naturalness     — min(1, ppl_original / ppl_attacked)
        composite       — α·sem_sim + β·naturalness + γ·edit_sim
    """
    α, β, γ = weights

    if math.isnan(ppl_original) or math.isnan(ppl_attacked) or ppl_original == 0:
        ppl_ratio   = float("nan")
        naturalness = float("nan")
        composite   = float("nan")
    else:
        ppl_ratio   = ppl_attacked / ppl_original
        naturalness = min(1.0, ppl_original / ppl_attacked)
        # Handle nan components gracefully
        sem   = sem_sim   if not math.isnan(sem_sim)   else 0.0
        nat   = naturalness if not math.isnan(naturalness) else 0.0
        esim  = edit_sim  if not math.isnan(edit_sim)  else 0.0
        composite = α * sem + β * nat + γ * esim

    return {
        "ppl_ratio":   round(ppl_ratio,   4) if not math.isnan(ppl_ratio)   else float("nan"),
        "naturalness": round(naturalness, 4) if not math.isnan(naturalness) else float("nan"),
        "composite":   round(composite,   4) if not math.isnan(composite)   else float("nan"),
    }


# ── DataFrame-level helper ────────────────────────────────────────────────────

def add_stealth_components(
    results_df: pd.DataFrame,
    ppl_model=None,
    ppl_tokenizer=None,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """
    Enrich a ``run_all_attacks()`` results DataFrame with composite stealth
    components, computed for every row where text actually changed.

    New columns added
    -----------------
    ppl_original      — GPT-2 perplexity of the original text
    ppl_attacked      — GPT-2 perplexity of the attacked text
    ppl_ratio         — ppl_attacked / ppl_original  (> 1 = less natural)
    naturalness       — min(1, 1 / ppl_ratio)
    edit_sim          — character-level similarity (difflib ratio)
    composite_stealth — weighted composite of sem_sim + naturalness + edit_sim

    Parameters
    ----------
    results_df : pd.DataFrame
        Long-form output from ``run_all_attacks()``.
    ppl_model : GPT2LMHeadModel | None
        If None, perplexity columns are filled with NaN and composite stealth
        uses only sem_sim + edit_sim with renormalised weights (α'=0.7, γ'=0.3).
    ppl_tokenizer : GPT2TokenizerFast | None
        Required when ppl_model is provided.
    weights : tuple
        (α, β, γ) for (sem_sim, naturalness, edit_sim).  Default (0.5, 0.3, 0.2).
    checkpoint_path : str | None
        Path to save intermediate results after each attack group.
        Enables resume-on-restart for large n.  Example:
        ``"../results/01_stealth_ckpt_n872.csv"``

    Returns
    -------
    pd.DataFrame
        Copy of results_df with stealth component columns appended.

    Performance notes (n=872, 10 attacks)
    ---------------------------------------
    * ppl_original is computed once per unique source sentence (~872 calls)
      rather than once per row (~8,720 calls) — a 10× speedup on that pass.
    * ppl_attacked is computed for changed rows only (~6,000–7,000 calls).
    * gc.collect() is called after each attack group to release memory.
    """
    import gc
    import os
    from tqdm.auto import tqdm

    df      = results_df.copy()
    use_ppl = (ppl_model is not None) and (ppl_tokenizer is not None)

    # Renormalised weights when perplexity unavailable
    α_np = weights[0] / (weights[0] + weights[2])
    γ_np = weights[2] / (weights[0] + weights[2])

    # Initialise output columns with NaN
    for col in ("ppl_original", "ppl_attacked", "ppl_ratio",
                "naturalness", "edit_sim", "composite_stealth"):
        df[col] = float("nan")

    # ── Checkpoint: reload already-processed attacks ──────────────────────────
    completed_attacks: set[str] = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = pd.read_csv(checkpoint_path)
            stealth_cols = [c for c in ("ppl_original", "ppl_attacked", "ppl_ratio",
                                        "naturalness", "edit_sim", "composite_stealth")
                            if c in ckpt.columns]
            completed_attacks = set(ckpt["attack"].unique())
            print(f"  📂 Stealth checkpoint: {sorted(completed_attacks)} already done — skipping")
            # Patch stealth columns from checkpoint for completed attacks
            ckpt_mask = ckpt["attack"].isin(completed_attacks)
            df_mask   = df["attack"].isin(completed_attacks)
            for col in stealth_cols:
                df.loc[df_mask, col] = ckpt.loc[ckpt_mask, col].values
        except Exception as e:
            print(f"  ⚠️  Could not read stealth checkpoint ({e}) — recomputing all.")

    # ── Deduplicate ppl_original across attacks ───────────────────────────────
    # Source sentences repeat once per attack (n × num_attacks rows total).
    # Computing ppl_original per-row is (num_attacks)× more work than needed.
    ppl_orig_cache: dict[str, float] = {}
    if use_ppl:
        pending_mask  = ~df["attack"].isin(completed_attacks)
        changed_mask  = df["text_changed"].fillna(False).astype(bool)
        unique_texts  = df.loc[pending_mask & changed_mask, "original_text"].unique()
        print(f"  📊 Pre-computing ppl_original for {len(unique_texts):,} unique source texts "
              f"(shared across all remaining attacks)…")
        for txt in tqdm(unique_texts, desc="ppl_original cache", leave=False):
            ppl_orig_cache[txt] = compute_perplexity(txt, ppl_model, ppl_tokenizer)
        print(f"  ✅ ppl_original cache ready ({len(ppl_orig_cache):,} entries)")

    # ── Process each attack group ─────────────────────────────────────────────
    for attack_name in df["attack"].unique():
        if attack_name in completed_attacks:
            continue

        mask  = df["attack"] == attack_name
        group = df[mask]
        n_changed = int(group["text_changed"].fillna(False).sum())
        print(f"\n  ── {attack_name}  ({mask.sum()} rows, {n_changed} changed) ──")

        for row_idx, row in tqdm(group.iterrows(), total=len(group),
                                 desc=attack_name, leave=False):
            changed = bool(row.get("text_changed", False))
            if not changed:
                continue

            orig     = str(row["original_text"])
            attacked = str(row["attacked_text"])
            sem_sim  = row.get("semantic_sim", float("nan"))
            if sem_sim is None:
                sem_sim = float("nan")

            esim = edit_similarity(orig, attacked)

            if use_ppl:
                ppl_o = ppl_orig_cache.get(orig, float("nan"))
                ppl_a = compute_perplexity(attacked, ppl_model, ppl_tokenizer)
                comps = composite_stealth_score(sem_sim, ppl_o, ppl_a, esim, weights)
                df.at[row_idx, "ppl_original"]     = round(ppl_o, 4) if not math.isnan(ppl_o) else float("nan")
                df.at[row_idx, "ppl_attacked"]      = round(ppl_a, 4) if not math.isnan(ppl_a) else float("nan")
                df.at[row_idx, "ppl_ratio"]         = comps["ppl_ratio"]
                df.at[row_idx, "naturalness"]       = comps["naturalness"]
                df.at[row_idx, "edit_sim"]          = round(esim, 4)
                df.at[row_idx, "composite_stealth"] = comps["composite"]
            else:
                comp_val = (
                    round(α_np * float(sem_sim) + γ_np * esim, 4)
                    if not math.isnan(float(sem_sim)) else float("nan")
                )
                df.at[row_idx, "edit_sim"]          = round(esim, 4)
                df.at[row_idx, "composite_stealth"] = comp_val

        # Save checkpoint and free memory after each attack group
        if checkpoint_path:
            os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
            df.to_csv(checkpoint_path, index=False)
            print(f"  💾 Stealth checkpoint saved → {checkpoint_path}")

        gc.collect()

    return df
