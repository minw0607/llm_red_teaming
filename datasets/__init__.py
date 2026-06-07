"""
datasets/
Evaluation dataset loaders — all normalised to a common DataFrame schema.

Every loader returns a pandas DataFrame with at minimum:
  sentence : str   — the input text
  label    : int   — 0 = negative / contradiction / not-entailed
                     1 = positive / entailment
  source   : str   — dataset identifier (e.g. "sst2", "advglue_sst2")

Use ``load_eval_dataset(name)`` as the single entry point from notebooks.
"""

from .loader import load_eval_dataset

__all__ = ["load_eval_dataset"]
