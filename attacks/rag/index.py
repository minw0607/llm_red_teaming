"""
attacks/rag/index.py — Vector index with three access-control architectures.

This module is where the use case earns its place. A model-level benchmark cannot
produce the finding here, because the finding is not about the model at all: given
one corpus, one embedding model and one query, **the architecture alone decides
whether a user sees a document they are not entitled to.**

The three architectures
-----------------------
``no_filter``   Retrieve top-k, ignore clearance entirely. The naive build, and
                the *broken-pipeline control* — if this does not leak, the probes
                are too weak and every other result is vacuous.

``post_filter`` Retrieve top-k, then drop the documents the user may not read.
                The most common real build, because it is the easy one to bolt on.
                It leaks in two non-obvious ways:
                  1. **Slot consumption** — restricted documents occupy slots in
                     the top-k and are then discarded, so the user silently gets
                     fewer usable results than they should. Answer quality drops
                     in a way that looks like a model problem and is not.
                  2. **Count disclosure** — the number withheld is itself a signal
                     ("there are 4 documents you cannot see about this").
                Whether it leaks *content* depends entirely on the filter being
                correct, which is exactly the assumption worth testing.

``pre_filter``  Restrict the candidate set *before* the search. The correct build,
                and the *zero-leak control* — if this leaks, the harness is broken
                rather than the system.

Embeddings are all-MiniLM-L6-v2 with numpy cosine similarity. A few hundred
documents does not need a vector database, and avoiding one keeps the package
self-contained and offline-capable after the first model download.

**Scope.** Architectural findings transfer to a production vector store; specific
leak rates do not. A managed service may apply its own filtering semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .corpus import Document, may_read

ARCHITECTURES = ("no_filter", "post_filter", "pre_filter")

#: One-line description used in reports, so the table is readable without the docs.
ARCHITECTURE_NOTE = {
    "no_filter": "top-k, clearance ignored (naive build — broken-pipeline control)",
    "post_filter": "retrieve then drop above-clearance (the common build)",
    "pre_filter": "restrict candidates before search (correct build — zero-leak control)",
}


@dataclass
class Retrieved:
    """One retrieval result, carrying the entitlement verdict alongside the hit."""
    doc: Document
    score: float
    entitled: bool          # was the user allowed to see this?
    withheld: bool = False  # retrieved by search but removed before the model saw it


class VectorIndex:
    """Cosine-similarity index over the corpus, with pluggable access control."""

    def __init__(self, docs: list[Document], model_name: str = "all-MiniLM-L6-v2",
                 embeddings: np.ndarray | None = None):
        self.docs = docs
        self.by_id = {d.doc_id: d for d in docs}
        self._model_name = model_name
        self._model = None
        if embeddings is not None:
            self.emb = embeddings
        else:
            self.emb = self._embed([d.text for d in docs])

    # ── embedding ─────────────────────────────────────────────────────────────
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True,
                                 show_progress_bar=False, batch_size=64)
        # L2-normalise once so similarity is a plain dot product.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)

    # ── retrieval ─────────────────────────────────────────────────────────────
    def search(self, query: str, *, clearance: str, k: int = 5,
               architecture: str = "pre_filter") -> list[Retrieved]:
        """
        Return the documents this architecture would place in the model's context.

        The returned list is what the assistant actually sees. Documents removed by
        ``post_filter`` are *not* included — they are reported separately by
        :meth:`search_debug`, because whether a withheld document counts as a leak
        depends on the architecture and must not be silently conflated.
        """
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {architecture!r}")
        hits, _ = self.search_debug(query, clearance=clearance, k=k,
                                    architecture=architecture)
        return hits

    def search_debug(self, query: str, *, clearance: str, k: int = 5,
                     architecture: str = "pre_filter") -> tuple[list[Retrieved], list[Retrieved]]:
        """
        Return ``(delivered, withheld)``.

        ``withheld`` is only ever non-empty for ``post_filter``; it is what makes
        slot consumption measurable, since those documents occupied top-k slots
        that a correct pre-filter would have given to usable results.
        """
        q = self._embed([query])[0]

        if architecture == "pre_filter":
            # Restrict the candidate set BEFORE ranking.
            idx = [i for i, d in enumerate(self.docs) if may_read(clearance, d.tier)]
            if not idx:
                return [], []
            sims = self.emb[idx] @ q
            order = np.argsort(-sims)[:k]
            return [Retrieved(self.docs[idx[j]], float(sims[j]),
                              entitled=True) for j in order], []

        sims = self.emb @ q
        order = np.argsort(-sims)[:k]
        delivered, withheld = [], []
        for j in order:
            d = self.docs[j]
            ok = may_read(clearance, d.tier)
            hit = Retrieved(d, float(sims[j]), entitled=ok)
            if architecture == "no_filter":
                delivered.append(hit)
            else:                       # post_filter
                if ok:
                    delivered.append(hit)
                else:
                    hit.withheld = True
                    withheld.append(hit)
        return delivered, withheld


# ── Retrieval-only measurement (no LLM involved) ────────────────────────────────

def retrieval_leak_check(index: VectorIndex, queries: list[dict], *,
                         k: int = 5, architectures=ARCHITECTURES) -> list[dict]:
    """
    Measure, per architecture, how often an above-clearance document reaches the
    model's context — using **no model calls at all**.

    This is the deterministic half of the audit, and the analogue of the embedding
    similarity measurement in the hiring use case: it is fully reproducible, costs
    nothing, and isolates the component. Whether the *assistant* then repeats what
    it was given is a separate, later question.

    ``queries`` — dicts of ``{query, clearance, target_doc_id (optional)}``.
    """
    rows = []
    for arch in architectures:
        for qi, q in enumerate(queries):
            delivered, withheld = index.search_debug(
                q["query"], clearance=q["clearance"], k=k, architecture=arch)
            unentitled = [r for r in delivered if not r.entitled]
            target = q.get("target_doc_id")
            rows.append({
                "architecture": arch,
                "query_idx": qi,
                "clearance": q["clearance"],
                "query": q["query"][:120],
                "n_delivered": len(delivered),
                "n_unentitled": len(unentitled),
                "leaked": bool(unentitled),
                # Was the specific document this probe targets in context?
                "target_delivered": bool(target and any(
                    r.doc.doc_id == target for r in delivered)),
                # Reachability: did search find it at all, before any filtering?
                "target_retrieved": bool(target and any(
                    r.doc.doc_id == target for r in delivered + withheld)),
                "n_withheld": len(withheld),
                # Slot consumption — usable results lost to discarded documents.
                "usable_slots": len(delivered),
                "canaries_in_context": [r.doc.canary for r in delivered
                                        if r.doc.canary and not r.entitled],
            })
    return rows
