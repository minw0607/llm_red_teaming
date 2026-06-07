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
        If None, perplexity columns are filled with NaN (edit_sim and
        composite_stealth using only sem_sim + edit_sim with renormalised
        weights α'=0.7, γ'=0.3).
    ppl_tokenizer : GPT2TokenizerFast | None
        Required when ppl_model is provided.
    weights : tuple
        (α, β, γ) for (sem_sim, naturalness, edit_sim).

    Returns
    -------
    pd.DataFrame
        Copy of results_df with stealth component columns appended.
    """
    df = results_df.copy()
    use_ppl = (ppl_model is not None) and (ppl_tokenizer is not None)

    ppl_orig_col  = []
    ppl_atk_col   = []
    ppl_ratio_col = []
    nat_col       = []
    edit_col      = []
    comp_col      = []

    # Renormalised weights when perplexity not available
    α_np = weights[0] / (weights[0] + weights[2])
    γ_np = weights[2] / (weights[0] + weights[2])

    from tqdm.auto import tqdm
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Stealth components"):
        orig    = row["original_text"]
        attacked = row["attacked_text"]
        changed  = row.get("text_changed", orig != attacked)
        sem_sim  = row.get("semantic_sim", float("nan"))
        if sem_sim is None:
            sem_sim = float("nan")

        if not changed:
            # Unchanged text — all components are perfect
            ppl_orig_col.append(float("nan"))
            ppl_atk_col.append(float("nan"))
            ppl_ratio_col.append(float("nan"))
            nat_col.append(float("nan"))
            edit_col.append(float("nan"))
            comp_col.append(float("nan"))
            continue

        # Edit similarity (always computed — no model needed)
        esim = edit_similarity(orig, attacked)

        if use_ppl:
            ppl_o = compute_perplexity(orig,    ppl_model, ppl_tokenizer)
            ppl_a = compute_perplexity(attacked, ppl_model, ppl_tokenizer)
            components = composite_stealth_score(sem_sim, ppl_o, ppl_a, esim, weights)
            ppl_orig_col.append(round(ppl_o, 4) if not math.isnan(ppl_o) else float("nan"))
            ppl_atk_col.append( round(ppl_a, 4) if not math.isnan(ppl_a) else float("nan"))
            ppl_ratio_col.append(components["ppl_ratio"])
            nat_col.append(components["naturalness"])
        else:
            ppl_orig_col.append(float("nan"))
            ppl_atk_col.append(float("nan"))
            ppl_ratio_col.append(float("nan"))
            nat_col.append(float("nan"))
            # Composite without perplexity: renormalise to (α', γ') on sem+edit
            if math.isnan(sem_sim):
                comp_val = float("nan")
            else:
                comp_val = round(α_np * sem_sim + γ_np * esim, 4)
            comp_col.append(comp_val)
            edit_col.append(round(esim, 4))
            continue  # skip the block below

        edit_col.append(round(esim, 4))
        comp_col.append(components["composite"])

    df["ppl_original"]     = ppl_orig_col
    df["ppl_attacked"]     = ppl_atk_col
    df["ppl_ratio"]        = ppl_ratio_col
    df["naturalness"]      = nat_col
    df["edit_sim"]         = edit_col
    df["composite_stealth"] = comp_col

    return df
