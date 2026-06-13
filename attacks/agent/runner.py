"""
attacks/agent/runner.py — Drive agentic injection scenarios and score them.

For each scenario (× repeats) it plants a fresh canary, runs the ToolAgent to
completion against the mock Sandbox, then applies the scenario's deterministic
check to decide whether an **unsafe action** occurred. Every result carries the
full **replayable trajectory** and the list of sink calls, so a finding can be
reproduced and audited.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .agent import ToolAgent
from .scenarios import build_scenarios
from attacks.prompt.injection import _new_canary


@dataclass
class AgentResult:
    scenario: str
    attack_type: str          # 'indirect' | 'direct'
    unsafe_action: bool       # headline: did the agent take a forbidden sink action?
    detail: str
    n_steps: int
    sink_calls: list          # [{tool, args}] — consequential actions taken
    trajectory: list          # replayable: per-step output/action/observation
    final_answer: str
    canary: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _key(scenario: str, idx: int) -> str:
    return f"{scenario}#{idx}"


def _load_ckpt(path):
    if not path or not os.path.exists(path):
        return set(), []
    done, results = set(), []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                results.append(AgentResult(**d))
                done.add(_key(d["scenario"], d.get("_repeat", 0)))
        print(f"  📂 Checkpoint loaded: {len(results)} result(s) — resuming.")
    except Exception as exc:
        print(f"  ⚠️  Could not read checkpoint ({exc}) — starting fresh.")
        return set(), []
    return done, results


def _append_ckpt(path, result, repeat):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    d = asdict(result); d["_repeat"] = repeat
    with open(path, "a") as f:
        f.write(json.dumps(d) + "\n")


class AgentAttackRunner:
    """Run agentic tool-attack scenarios against a target model."""

    def __init__(self, target, max_steps: int = 6, sleep_sec: float = 0.3, scenarios=None):
        self.agent = ToolAgent(target, max_steps=max_steps, sleep_sec=0.0)
        self.sleep_sec = sleep_sec
        self.scenarios = scenarios or build_scenarios()

    def run(self, *, repeats: int = 1, checkpoint_path: str | None = None,
            verbose: bool = True) -> list[AgentResult]:
        done, results = _load_ckpt(checkpoint_path) if checkpoint_path else (set(), [])
        for rep in range(repeats):
            for scen in self.scenarios:
                if _key(scen.name, rep) in done:
                    continue
                canary = _new_canary()
                sandbox = scen.build(canary)
                trace = self.agent.run(sandbox, scen.user_task)
                unsafe, detail = scen.check(sandbox, canary)
                result = AgentResult(
                    scenario=scen.name, attack_type=scen.attack_type,
                    unsafe_action=unsafe, detail=detail,
                    n_steps=trace["n_steps"],
                    sink_calls=[{"tool": c.tool, "args": c.args} for c in sandbox.sink_calls()],
                    trajectory=trace["trajectory"], final_answer=trace["final_answer"],
                    canary=canary)
                results.append(result)
                if checkpoint_path:
                    _append_ckpt(checkpoint_path, result, rep)
                if verbose:
                    print(f"[{scen.name:18s} {scen.attack_type:8s}] "
                          f"→ {'UNSAFE: ' + detail if unsafe else 'safe (' + detail + ')'}")
                time.sleep(self.sleep_sec)
        return results
