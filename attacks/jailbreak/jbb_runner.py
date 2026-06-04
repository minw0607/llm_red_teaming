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
    results = runner.run(n_goals=50)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

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

    def run(self, n_goals: int = 50) -> list[JBBResult]:
        """
        Send the first *n_goals* JailbreakBench goals to the target model.

        Parameters
        ----------
        n_goals : int
            Number of goals to evaluate (default 50).

        Returns
        -------
        list[JBBResult]
            One result record per goal.
        """
        dataset = jbb.read_dataset()
        results: list[JBBResult] = []

        goals = dataset.goals[:n_goals]
        behaviors = dataset.behaviors[:n_goals]

        for idx, (goal, behavior) in enumerate(zip(goals, behaviors)):
            try:
                response = self.target.complete(
                    user_prompt=goal,
                    system_prompt=self.system_prompt,
                )
            except Exception as exc:
                response = f"[ERROR] {exc}"

            verdict, reason = self.judge.judge(response)

            results.append(
                JBBResult(
                    idx=idx,
                    behavior=behavior,
                    goal=goal,
                    response=response,
                    verdict=verdict,
                    reason=reason,
                )
            )

            print(
                f"[{idx+1:3d}/{n_goals}] behavior={behavior!r:30s}  "
                f"verdict={verdict}"
            )
            time.sleep(self.sleep_sec)

        return results
