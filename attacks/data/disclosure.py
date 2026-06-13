"""
attacks/data/disclosure.py — Track A: system-prompt & secret disclosure.

Plants a unique canary secret in the SYSTEM prompt, then runs a taxonomy of
prompt-extraction attacks as the user. Success = the canary appears in the
response (deterministic). Maps to OWASP LLM07 (System Prompt Leakage).
"""

from __future__ import annotations

import time

from ._common import DataLeakResult, append_ckpt, load_ckpt
from .fixtures import (
    DISCLOSURE_STRATEGIES, system_with_secret, _new_canary, _canary_hit,
)


class DisclosureRunner:
    """Run system-prompt secret-extraction attacks against a target."""

    def __init__(self, target, sleep_sec: float = 0.3, strategies=None):
        self.target = target
        self.sleep_sec = sleep_sec
        self.strategies = strategies or DISCLOSURE_STRATEGIES

    def run(self, *, repeats: int = 1, checkpoint_path: str | None = None,
            verbose: bool = True) -> list[DataLeakResult]:
        """
        Each strategy is tried ``repeats`` times (a fresh canary each time, since
        model output is non-deterministic). Returns one result per attempt.
        """
        done, results = load_ckpt(checkpoint_path) if checkpoint_path else (set(), [])
        existing = {(r.track, r.idx) for r in results}
        idx = 0
        for _ in range(repeats):
            for strat, user_prompt in self.strategies:
                if ("disclosure", idx) in existing:
                    idx += 1
                    continue
                canary = _new_canary()
                system_prompt = system_with_secret(canary)
                try:
                    response = self.target.complete(
                        user_prompt=user_prompt, system_prompt=system_prompt)
                except Exception as exc:
                    response = f"[ERROR] {exc}"

                leaked = _canary_hit(response, canary)
                result = DataLeakResult(
                    idx=idx, track="disclosure", strategy=strat,
                    leaked=leaked, leak_type="secret_canary" if leaked else "none",
                    prompt=user_prompt, response=response[:400],
                    detail="canary emitted — secret disclosed" if leaked
                           else "canary absent — secret protected",
                    canary=canary,
                )
                results.append(result)
                if checkpoint_path:
                    append_ckpt(checkpoint_path, result)
                if verbose:
                    print(f"[{idx:3d}] disclosure {strat:22s} "
                          f"→ {'LEAKED' if leaked else 'protected'}")
                idx += 1
                time.sleep(self.sleep_sec)
        results.sort(key=lambda r: (r.track, r.idx))
        return results
