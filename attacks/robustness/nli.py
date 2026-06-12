"""
attacks/robustness/nli.py — Zero-shot NLI classification harness.

Runs a target model as a 3-way Natural Language Inference classifier and scores
its predictions deterministically. Used to measure **reasoning robustness**: the
gap between accuracy on clean data (MultiNLI) and on adversarial data (ANLI /
AdvGLUE).

Unlike the perturbation attacks in NB01, the *dataset* is the adversary here —
ANLI items were written by humans specifically to fool strong models, so a high
clean accuracy paired with a low ANLI accuracy reveals brittle reasoning rather
than a surface-typo weakness.

Success metric — **prediction correctness** (deterministic): the model is asked
to answer with exactly one of {entailment, neutral, contradiction}; the reply is
parsed to a label id and compared to the gold label. Unparseable replies are
recorded as ``pred == -1`` and count as wrong (never silently dropped).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .datasets import NLI_LABELS, NLIItem

LABEL_NAMES = NLI_LABELS  # {0:'entailment', 1:'neutral', 2:'contradiction'}


# ── Prompting ────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a Natural Language Inference classifier. Given a PREMISE and a "
    "HYPOTHESIS, decide the logical relationship between them:\n"
    "- entailment: if the premise is true, the hypothesis must also be true.\n"
    "- neutral: the hypothesis might be true, but the premise does not "
    "guarantee it.\n"
    "- contradiction: if the premise is true, the hypothesis must be false.\n"
    "Reply with EXACTLY one word: entailment, neutral, or contradiction. "
    "No explanation."
)


def _user_prompt(premise: str, hypothesis: str) -> str:
    return f"Premise: {premise}\nHypothesis: {hypothesis}\nAnswer:"


# Word-boundary patterns so 'neutral' isn't matched inside other text, and the
# first explicit label in the reply wins.
_LABEL_PATTERNS = [
    (0, re.compile(r"\bentail(?:ment|s|ed)?\b", re.I)),
    (2, re.compile(r"\bcontradict(?:ion|s|ory|ed)?\b", re.I)),
    (1, re.compile(r"\bneutral\b", re.I)),
]


def parse_label(response: str) -> int:
    """Map a free-text reply to a label id, or -1 if no label is found."""
    if not response:
        return -1
    # Earliest-appearing explicit label wins (handles "neutral, not entailment").
    best_id, best_pos = -1, len(response) + 1
    for lid, pat in _LABEL_PATTERNS:
        m = pat.search(response)
        if m and m.start() < best_pos:
            best_id, best_pos = lid, m.start()
    return best_id


# ── Result record ────────────────────────────────────────────────────────────────

@dataclass
class NLIResult:
    idx: int
    source: str           # dataset key: mnli / anli_r1 / advglue_mnli / …
    premise: str
    hypothesis: str
    gold: int             # gold label id
    pred: int             # predicted label id (-1 = unparseable)
    correct: bool
    response: str
    reason: str = ""      # ANLI human annotation (why the item is hard)
    item_hash: str = ""   # stable fingerprint of premise/hypothesis/label
    target_id: str = ""   # non-secret model identifier used for checkpoint isolation
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── Checkpoint helpers (JSONL, resume-safe) ────────────────────────────────────

def _item_hash(item: NLIItem) -> str:
    payload = f"{item.premise}\n{item.hypothesis}\n{item.label}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _target_id(target) -> str:
    model = getattr(target, "model", None) or target.__class__.__name__
    provider = getattr(target, "provider", None)
    return f"{provider}:{model}" if provider else str(model)


def _key(source: str, idx: int, item_hash: str = "", target_id: str = "") -> str:
    return f"{source}#{idx}#{item_hash}#{target_id}"


def _load_ckpt(path: str) -> tuple[set[str], dict[str, NLIResult]]:
    if not path or not os.path.exists(path):
        return set(), {}
    results_by_key, done = {}, set()
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                result = NLIResult(**d)
                k = _key(d["source"], d["idx"], d.get("item_hash", ""), d.get("target_id", ""))
                results_by_key[k] = result
                done.add(k)
        print(f"  📂 Checkpoint loaded: {len(results_by_key)} result(s) — resuming.")
    except Exception as exc:
        print(f"  ⚠️  Could not read checkpoint ({exc}) — starting fresh.")
        return set(), {}
    return done, results_by_key


def _append_ckpt(path: str, result: NLIResult) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


# ── Runner ───────────────────────────────────────────────────────────────────────

class NLIRunner:
    """
    Run zero-shot NLI classification against a target model.

    Parameters
    ----------
    target : object
        Target with ``complete(user_prompt, system_prompt) -> str``.
    sleep_sec : float
        Pacing delay between API calls.
    """

    def __init__(self, target, sleep_sec: float = 0.2):
        self.target = target
        self.sleep_sec = sleep_sec

    def run(
        self,
        items: list[NLIItem],
        *,
        source: str | None = None,
        checkpoint_path: str | None = None,
        verbose: bool = True,
    ) -> list[NLIResult]:
        """
        Classify every item and score it against its gold label.

        Parameters
        ----------
        items : list[NLIItem]
            Premise/hypothesis pairs (from the dataset loaders).
        source : str | None
            Override the per-item ``source`` label (else taken from each item).
        checkpoint_path : str | None
            Resume-safe ``.jsonl`` checkpoint, keyed by source, item fingerprint,
            and target model so multiple datasets/configs can share one file safely.
        """
        done, results_by_key = _load_ckpt(checkpoint_path) if checkpoint_path else (set(), {})
        target_id = _target_id(self.target)
        results: list[NLIResult] = []

        for item in items:
            src = source or item.source
            item_hash = _item_hash(item)
            k = _key(src, item.idx, item_hash, target_id)
            if k in done:
                results.append(results_by_key[k])
                continue

            try:
                response = self.target.complete(
                    user_prompt=_user_prompt(item.premise, item.hypothesis),
                    system_prompt=_SYSTEM,
                )
            except Exception as exc:
                response = f"[ERROR] {exc}"

            pred = parse_label(response)
            result = NLIResult(
                idx=item.idx, source=src,
                premise=item.premise[:600], hypothesis=item.hypothesis[:400],
                gold=item.label, pred=pred, correct=(pred == item.label),
                response=response[:300], reason=item.reason[:400],
                item_hash=item_hash, target_id=target_id,
            )
            results.append(result)
            done.add(k)
            results_by_key[k] = result
            if checkpoint_path:
                _append_ckpt(checkpoint_path, result)

            if verbose:
                mark = "✓" if result.correct else ("?" if pred == -1 else "✗")
                print(f"[{src:24s} {item.idx:4d}] gold={NLI_LABELS[item.label]:13s} "
                      f"pred={NLI_LABELS.get(pred, 'UNPARSED'):13s} {mark}")
            time.sleep(self.sleep_sec)

        results.sort(key=lambda r: (r.source, r.idx))
        return results

    def run_suite(
        self,
        suite: dict[str, list[NLIItem]],
        *,
        checkpoint_path: str | None = None,
        verbose: bool = True,
    ) -> list[NLIResult]:
        """Run every dataset in a ``{source: [NLIItem, …]}`` suite in one pass."""
        all_results: list[NLIResult] = []
        for src, items in suite.items():
            if verbose:
                print(f"\n── {src}  (n={len(items)}) ──")
            all_results += self.run(items, source=src,
                                    checkpoint_path=checkpoint_path, verbose=verbose)
        return all_results
