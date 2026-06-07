"""
datasets/loader.py — Unified dataset loader for adversarial evaluation.

Supported datasets
------------------
"sst2"
    Stanford Sentiment Treebank 2 (dev split, 872 rows).
    Binary sentiment: 1=positive, 0=negative.
    Loaded from the bundled repo copy (data/sst2/dev.tsv) with an automatic
    HuggingFace fallback.

"advglue_sst2"
    AdvGLUE adversarial SST-2 validation split (Yang et al. 2021).
    Same binary sentiment task but sentences were adversarially modified by
    human annotators to fool BERT/RoBERTa — a much harder test set.
    Downloaded automatically from HuggingFace on first use.

All loaders return a DataFrame with columns: sentence, label, source.
The ``source`` column lets the summary tables identify which dataset
results came from when mixing datasets in a single evaluation.

Usage
-----
    from datasets import load_eval_dataset
    df = load_eval_dataset("sst2",        data_dir="../data")
    df = load_eval_dataset("advglue_sst2")
"""

from __future__ import annotations

import os
import pandas as pd


# ── SST-2 ─────────────────────────────────────────────────────────────────────

def _load_sst2(data_dir: str = "../data") -> pd.DataFrame:
    """
    Load SST-2 dev split.

    Tries the bundled repo copy first (data/sst2/dev.tsv).
    Falls back to a live HuggingFace download if the file is absent.
    """
    dev_path = os.path.join(data_dir, "sst2", "dev.tsv")

    if os.path.exists(dev_path):
        df = pd.read_csv(dev_path, sep="\t")
    else:
        print("SST-2 dev.tsv not found — downloading from HuggingFace …")
        from datasets import load_dataset as hf_load
        hf_ds = hf_load("stanfordnlp/sst2")
        df = hf_ds["validation"].to_pandas()[["sentence", "label"]]
        os.makedirs(os.path.dirname(dev_path), exist_ok=True)
        df.to_csv(dev_path, sep="\t", index=False)
        print(f"Saved to {dev_path}")

    df = df[["sentence", "label"]].copy()
    df["source"] = "sst2"
    return df.reset_index(drop=True)


# ── AdvGLUE ───────────────────────────────────────────────────────────────────

def _load_advglue_sst2() -> pd.DataFrame:
    """
    Load AdvGLUE adversarial SST-2 validation split.

    AdvGLUE (Yang et al. 2021) sentences are designed to fool transformer
    classifiers while preserving human-readable meaning — they are harder
    than vanilla SST-2 and provide a more realistic evaluation surface.

    Reference: https://arxiv.org/abs/2106.09680
    HuggingFace: https://huggingface.co/datasets/adv_glue
    """
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load("adv_glue", "adv_sst2", trust_remote_code=True)
        # AdvGLUE uses the "validation" split
        split = "validation" if "validation" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()

        # Normalise column names to match SST-2 schema
        col_map = {}
        for c in df.columns:
            if c in ("sentence", "text", "review"):
                col_map[c] = "sentence"
            elif c in ("label", "labels"):
                col_map[c] = "label"
        df = df.rename(columns=col_map)

        df = df[["sentence", "label"]].copy()
        df["label"] = df["label"].astype(int)
        df["source"] = "advglue_sst2"
        print(f"✅ AdvGLUE SST-2 loaded: {len(df):,} rows")
        return df.reset_index(drop=True)

    except Exception as e:
        print(f"⚠️  Could not load AdvGLUE from HuggingFace ({e}).")
        print("   Falling back to SST-2.  To use AdvGLUE, ensure you have")
        print("   network access and run: pip install datasets")
        return None


# ── Public entry point ────────────────────────────────────────────────────────

def load_eval_dataset(
    name: str,
    data_dir: str = "../data",
    fallback_to_sst2: bool = True,
) -> pd.DataFrame:
    """
    Load an evaluation dataset by name, normalised to a common schema.

    Parameters
    ----------
    name : str
        Dataset identifier.  Supported values:

        "sst2"
            SST-2 dev split (872 rows, binary sentiment).
        "advglue_sst2"
            AdvGLUE adversarial SST-2 validation split.

    data_dir : str
        Root data directory for file-based datasets (default ``"../data"``).
    fallback_to_sst2 : bool
        If True and the requested dataset fails to load, fall back to SST-2
        rather than raising an exception.  Useful for offline environments.

    Returns
    -------
    pd.DataFrame
        Columns: ``sentence`` (str), ``label`` (int), ``source`` (str).

    Raises
    ------
    ValueError
        If *name* is not a recognised dataset identifier.
    """
    name = name.lower().strip()

    if name == "sst2":
        df = _load_sst2(data_dir)
    elif name == "advglue_sst2":
        df = _load_advglue_sst2()
        if df is None and fallback_to_sst2:
            print("   Using SST-2 as fallback.")
            df = _load_sst2(data_dir)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Supported: 'sst2', 'advglue_sst2'."
        )

    print(f"Dataset : {df['source'].iloc[0]}  |  {len(df):,} rows  |  "
          f"label distribution: {dict(df['label'].value_counts().sort_index())}")
    return df
