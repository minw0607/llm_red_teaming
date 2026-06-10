"""
attacks/jailbreak/harmbench.py — HarmBench dataset loader.

Loads the HarmBench behavior set (Mazeika et al., 2024) and exposes it with
the same interface as ``jailbreakbench.read_dataset()`` (``.goals``,
``.behaviors``, ``.categories``) so the existing jailbreak runners can consume
it without modification.

HarmBench provides 400 behaviors across 7 semantic categories with published
ASR baselines, vs. JailbreakBench's 100 — broader coverage for benchmarking a
target against known models.

Reference: https://arxiv.org/abs/2402.04249  ·  https://www.harmbench.org/
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

# Official behavior CSV (text-only, all behaviors)
_HARMBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)

# Local cache path (repo eval_datasets/safety/)
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "eval_datasets", "safety",
)
_CACHE_CSV = os.path.join(_CACHE_DIR, "harmbench_behaviors_text_all.csv")


@dataclass
class HarmBenchDataset:
    """Mirror of JailbreakBench's dataset interface used by the runners."""
    goals: list[str]
    behaviors: list[str]          # short BehaviorID labels
    categories: list[str]         # SemanticCategory
    df: pd.DataFrame = field(repr=False, default=None)

    def __len__(self) -> int:
        return len(self.goals)


def _load_csv(force_download: bool = False) -> pd.DataFrame:
    """Load the HarmBench CSV from local cache, downloading once if absent."""
    if os.path.exists(_CACHE_CSV) and not force_download:
        return pd.read_csv(_CACHE_CSV)
    df = pd.read_csv(_HARMBENCH_CSV_URL)
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        df.to_csv(_CACHE_CSV, index=False)
    except OSError:
        pass  # cache is best-effort; proceed with in-memory copy
    return df


def load_harmbench(
    n: int | None = None,
    include_contextual: bool = False,
    force_download: bool = False,
) -> HarmBenchDataset:
    """
    Load HarmBench behaviors as a JBB-compatible dataset.

    Parameters
    ----------
    n : int | None
        Cap the number of behaviors returned (None = all).
    include_contextual : bool
        HarmBench "contextual" behaviors require a separate context string.
        When False (default), only self-contained "standard" + "copyright"
        behaviors are returned — these work as direct goals like JBB. When
        True, contextual behaviors are included with their ContextString
        prepended to the goal.
    force_download : bool
        Re-download the CSV even if a local cache exists.

    Returns
    -------
    HarmBenchDataset
        With ``.goals``, ``.behaviors`` (BehaviorID), ``.categories``
        (SemanticCategory), and the raw ``.df``.
    """
    df = _load_csv(force_download=force_download)

    if not include_contextual:
        df = df[df["FunctionalCategory"] != "contextual"].reset_index(drop=True)

    goals: list[str] = []
    for _, row in df.iterrows():
        goal = str(row["Behavior"]).strip()
        ctx = str(row.get("ContextString", "") or "").strip()
        if include_contextual and ctx and ctx.lower() != "nan":
            goal = f"{ctx}\n\n{goal}"
        goals.append(goal)

    behaviors = df["BehaviorID"].astype(str).tolist()
    categories = df["SemanticCategory"].astype(str).tolist()

    if n is not None:
        goals, behaviors, categories = goals[:n], behaviors[:n], categories[:n]

    return HarmBenchDataset(
        goals=goals,
        behaviors=behaviors,
        categories=categories,
        df=df,
    )
