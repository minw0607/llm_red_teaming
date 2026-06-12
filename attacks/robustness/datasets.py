"""
attacks/robustness/datasets.py — NLI robustness dataset loaders.

Loads three Natural Language Inference (NLI) datasets that share a single 3-way
label space (0 = entailment, 1 = neutral, 2 = contradiction), so a clean
baseline and two adversarial benchmarks can be scored with the same parser:

    MultiNLI   (nyu-mll/multi_nli)   — CLEAN baseline (validation_matched).
    ANLI       (facebook/anli)       — Adversarial NLI, human-in-the-loop,
                                       3 rounds of increasing difficulty
                                       (Nie et al., 2020). Ships a `reason`
                                       field explaining why each item fools
                                       models — used for error analysis.
    AdvGLUE    (adv_glue/adv_mnli*)  — Adversarially-perturbed MNLI (Wang et
                                       al., 2021); tests surface-perturbation
                                       robustness on the same reasoning task.

The headline metric a notebook builds on top of these is the **robustness gap**:
clean accuracy (MNLI) minus adversarial accuracy (ANLI / AdvGLUE).

Each loader returns a list of ``NLIItem`` records with a common shape. Datasets
are fetched once via the ``datasets`` library and cached to parquet under
``eval_datasets/robustness/`` (gitignored). A tiny in-repo fallback keeps the
notebook runnable offline.

References:
  ANLI    — https://arxiv.org/abs/1910.14599
  AdvGLUE — https://arxiv.org/abs/2111.02840
  MNLI    — https://cims.nyu.edu/~sbowman/multinli/
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

# ── Label space (shared across all three datasets) ───────────────────────────────

NLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_TO_ID = {v: k for k, v in NLI_LABELS.items()}

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "eval_datasets", "robustness",
)


@dataclass
class NLIItem:
    """One premise/hypothesis pair with a gold 3-way label."""
    idx: int
    premise: str
    hypothesis: str
    label: int            # 0 entailment · 1 neutral · 2 contradiction
    source: str           # 'mnli' | 'anli_r1' | 'advglue_mnli' | …
    reason: str = ""      # ANLI human annotation (why it fools models); else ""
    meta: dict = field(default_factory=dict)


# ── Offline fallbacks (keep the notebook runnable without network) ───────────────

_FALLBACK = {
    "mnli": [
        ("A man inspects the uniform of a figure.", "The man is sleeping.", 2),
        ("An older and younger man smiling.", "Two men are smiling at cats.", 1),
        ("A soccer game with multiple males playing.", "Some men are playing a sport.", 0),
    ],
    "anli_r3": [
        ("The bullet train can reach speeds of 320 km/h on dedicated track.",
         "The bullet train always travels at 320 km/h.", 1),
        ("Maria has exactly two siblings, both younger than her.",
         "Maria is the eldest of three children.", 0),
    ],
    "advglue_mnli": [
        ("well that would be a help we have got so much landfill here",
         "We have plenty of space in the landfill.", 2),
    ],
}


def _from_fallback(source: str) -> list[NLIItem]:
    rows = _FALLBACK.get(source, _FALLBACK["mnli"])
    return [
        NLIItem(idx=i, premise=p, hypothesis=h, label=l, source=source,
                meta={"fallback": True})
        for i, (p, h, l) in enumerate(rows)
    ]


# ── Cache helpers ────────────────────────────────────────────────────────────────

def _cache_path(source: str) -> str:
    return os.path.join(_CACHE_DIR, f"{source}.parquet")


def _load_cached(source: str) -> pd.DataFrame | None:
    path = _cache_path(source)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def _save_cache(source: str, df: pd.DataFrame) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        df.to_parquet(_cache_path(source))
    except OSError:
        pass  # cache is best-effort


def _fetch(hf_name: str, config: str | None, split: str) -> pd.DataFrame:
    from datasets import load_dataset
    ds = (load_dataset(hf_name, config, split=split)
          if config else load_dataset(hf_name, split=split))
    return ds.to_pandas()


def _to_items(df: pd.DataFrame, source: str, n: int | None,
              prem_col: str, hyp_col: str, shuffle: bool, seed: int) -> list[NLIItem]:
    df = df[df["label"].isin([0, 1, 2])].copy()      # drop unlabeled (-1) rows
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if n:
        df = df.head(n)
    has_reason = "reason" in df.columns
    items = []
    for i, row in enumerate(df.itertuples(index=False)):
        d = row._asdict()
        items.append(NLIItem(
            idx=i,
            premise=str(d[prem_col]).strip(),
            hypothesis=str(d[hyp_col]).strip(),
            label=int(d["label"]),
            source=source,
            reason=str(d.get("reason", "")).strip() if has_reason else "",
        ))
    return items


# ── Public loaders ───────────────────────────────────────────────────────────────

def load_mnli(n: int | None = 100, *, shuffle: bool = True, seed: int = 42,
              force_download: bool = False) -> list[NLIItem]:
    """CLEAN baseline — MultiNLI validation_matched (gold-labelled)."""
    source = "mnli"
    df = None if force_download else _load_cached(source)
    if df is None:
        try:
            df = _fetch("nyu-mll/multi_nli", None, "validation_matched")
            _save_cache(source, df[["premise", "hypothesis", "label"]])
        except Exception:
            return _from_fallback(source)[:n] if n else _from_fallback(source)
    return _to_items(df, source, n, "premise", "hypothesis", shuffle, seed)


def load_anli(round: int = 3, n: int | None = 100, *, split: str = "test",
              shuffle: bool = True, seed: int = 42,
              force_download: bool = False) -> list[NLIItem]:
    """
    Adversarial NLI — ``round`` ∈ {1, 2, 3} (difficulty increases by round).

    Uses the held-out ``test`` split by default (gold-labelled). The ``reason``
    field — a human annotation of why the example fools models — is preserved
    for downstream error analysis.
    """
    if round not in (1, 2, 3):
        raise ValueError("round must be 1, 2 or 3")
    source = f"anli_r{round}"
    df = None if force_download else _load_cached(source)
    if df is None:
        try:
            df = _fetch("facebook/anli", None, f"{split}_r{round}")
            keep = [c for c in ("premise", "hypothesis", "label", "reason") if c in df.columns]
            _save_cache(source, df[keep])
        except Exception:
            return _from_fallback(source)[:n] if n else _from_fallback(source)
    return _to_items(df, source, n, "premise", "hypothesis", shuffle, seed)


def load_advglue(task: str = "mnli", n: int | None = None, *,
                 shuffle: bool = False, seed: int = 42,
                 force_download: bool = False) -> list[NLIItem]:
    """
    AdvGLUE adversarially-perturbed NLI. ``task`` ∈ {"mnli", "mnli_mismatched"}.

    These are small dev sets (AdvGLUE test labels are private), so by default we
    keep the natural order and return all items.
    """
    if task not in ("mnli", "mnli_mismatched"):
        raise ValueError("task must be 'mnli' or 'mnli_mismatched'")
    source = f"advglue_{task}"
    config = f"adv_{task}"
    df = None if force_download else _load_cached(source)
    if df is None:
        try:
            df = _fetch("adv_glue", config, "validation")
            _save_cache(source, df[["premise", "hypothesis", "label"]])
        except Exception:
            return _from_fallback("advglue_mnli")[:n] if n else _from_fallback("advglue_mnli")
    return _to_items(df, source, n, "premise", "hypothesis", shuffle, seed)


# ── Convenience: build a labelled mix in one call ────────────────────────────────

def load_nli_dataset(
    n_clean: int = 100,
    anli_rounds: tuple[int, ...] = (1, 2, 3),
    n_per_anli: int = 100,
    advglue_tasks: tuple[str, ...] = ("mnli", "mnli_mismatched"),
    *, shuffle: bool = True, seed: int = 42, force_download: bool = False,
) -> dict[str, list[NLIItem]]:
    """
    Load the full robustness suite keyed by source name:
        {'mnli': [...], 'anli_r1': [...], 'anli_r2': [...], 'anli_r3': [...],
         'advglue_mnli': [...], 'advglue_mnli_mismatched': [...]}
    """
    suite: dict[str, list[NLIItem]] = {}
    suite["mnli"] = load_mnli(n_clean, shuffle=shuffle, seed=seed, force_download=force_download)
    for r in anli_rounds:
        suite[f"anli_r{r}"] = load_anli(r, n_per_anli, shuffle=shuffle, seed=seed,
                                        force_download=force_download)
    for t in advglue_tasks:
        suite[f"advglue_{t}"] = load_advglue(t, None, force_download=force_download)
    return suite
