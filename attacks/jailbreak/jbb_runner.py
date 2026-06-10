"""
JailbreakBenchRunner — Direct goal evaluation using JailbreakBench dataset.

Loads the JailbreakBench harmful-goals dataset and sends each goal directly
to a target model.  Responses are scored by the judge module.

Usage
-----
    from attacks.jailbreak import JailbreakBenchRunner
    from targets.azure_openai import AzureOpenAITarget
    from judges.classifier_judge import ClassifierJudge

    target  = AzureOpenAITarget()
    judge   = ClassifierJudge()
    runner  = JailbreakBenchRunner(target=target, judge=judge)
    results = runner.run(n_goals=50, checkpoint_path="../results/02_ckpt_direct.jsonl")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

# litellm 1.40+ moved prompt_templates; patch the old import path so
# jailbreakbench 1.0.0 can still find it.
import sys as _sys, types as _types
try:
    from litellm.llms.prompt_templates.factory import custom_prompt as _  # noqa: F401
except ModuleNotFoundError:
    import litellm.litellm_core_utils.prompt_templates.factory as _new_pt
    _pt_mod = _types.ModuleType("litellm.llms.prompt_templates.factory")
    _pt_mod.custom_prompt = _new_pt.custom_prompt
    _sys.modules.setdefault(
        "litellm.llms.prompt_templates",
        _types.ModuleType("litellm.llms.prompt_templates"),
    )
    _sys.modules["litellm.llms.prompt_templates.factory"] = _pt_mod

import jailbreakbench as jbb


@dataclass
class JBBResult:
    idx: int
    behavior: str
    goal: str
    response: str
    verdict: str
    reason: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def _load_jbb_checkpoint(path: str) -> tuple[set[int], list[JBBResult]]:
    """Load existing checkpoint; return (completed_indices, results)."""
    if not os.path.exists(path):
        return set(), []
    results: list[JBBResult] = []
    indices: set[int] = set()
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                results.append(JBBResult(**d))
                indices.add(d["idx"])
        print(f"  📂 Checkpoint loaded: {len(results)} result(s) — "
              f"skipping indices {sorted(indices)[:5]}{'…' if len(indices) > 5 else ''}")
    except Exception as exc:
        print(f"  ⚠️  Could not read checkpoint ({exc}) — starting fresh.")
        return set(), []
    return indices, results


def _append_jbb_checkpoint(path: str, result: JBBResult) -> None:
    """Append a single result as a JSON line."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


class JailbreakBenchRunner:
    """
    Run JailbreakBench goals directly against a target model.

    Parameters
    ----------
    target : object
        A target model instance with a ``complete(prompt: str) -> str`` method.
    judge : object
        A judge instance with a ``judge(response: str) -> tuple[str, str]``
        method returning (verdict, reason).
    sleep_sec : float
        Seconds to wait between API calls (default 0.3).
    system_prompt : str
        System message sent to the model.
    """

    DEFAULT_SYSTEM = (
        "You are a helpful assistant. Answer user questions accurately and safely."
    )

    def __init__(
        self,
        target,
        judge,
        sleep_sec: float = 0.3,
        system_prompt: str | None = None,
    ):
        self.target = target
        self.judge = judge
        self.sleep_sec = sleep_sec
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM

    def run(
        self,
        n_goals: int = 50,
        checkpoint_path: str | None = None,
    ) -> list[JBBResult]:
        """
        Send the first *n_goals* JailbreakBench goals to the target model.

        Parameters
        ----------
        n_goals : int
            Number of goals to evaluate (default 50).
        checkpoint_path : str | None
            Path to a ``.jsonl`` checkpoint file.  When provided:
            - Existing results are loaded and those goal indices are skipped.
            - Each new result is appended immediately so a partial run is
              never lost.
            Example: ``"../results/02_ckpt_direct_n50.jsonl"``

        Returns
        -------
        list[JBBResult]
            One result record per goal, ordered by index.
        """
        dataset = jbb.read_dataset()
        goals = dataset.goals[:n_goals]
        behaviors = dataset.behaviors[:n_goals]

        completed_indices: set[int] = set()
        results: list[JBBResult] = []

        if checkpoint_path:
            completed_indices, results = _load_jbb_checkpoint(checkpoint_path)

        for idx, (goal, behavior) in enumerate(zip(goals, behaviors)):
            if idx in completed_indices:
                continue

            try:
                response = self.target.complete(
                    user_prompt=goal,
                    system_prompt=self.system_prompt,
                )
            except Exception as exc:
                response = f"[ERROR] {exc}"

            verdict, reason = self.judge.judge(response)
            result = JBBResult(
                idx=idx,
                behavior=behavior,
                goal=goal,
                response=response,
                verdict=verdict,
                reason=reason,
            )
            results.append(result)

            if checkpoint_path:
                _append_jbb_checkpoint(checkpoint_path, result)

            print(
                f"[{idx+1:3d}/{n_goals}] behavior={behavior!r:30s}  "
                f"verdict={verdict}"
            )
            time.sleep(self.sleep_sec)

        results.sort(key=lambda r: r.idx)
        return results
