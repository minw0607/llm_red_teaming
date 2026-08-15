"""
attacks/hiring/rankers.py — Retrieval rankers for the retrieval-bias track.

Real screening pipelines retrieve/rank candidates with an embedding model before
any LLM reasons about them. Wilson & Caliskan (AIES 2024) showed that step is
itself biased — embedding retrieval favoured White-associated names in 85.1% of
cases. Because our matched-pair résumés are **identical apart from the name**,
any rank disparity produced here is caused by the name alone.

``EmbeddingRanker`` requires ``sentence-transformers`` (already in
requirements.txt). If the model can't be loaded we return ``None`` rather than
substituting a fake ranker — a retrieval-bias number should never be invented.
"""

from __future__ import annotations

from .corpus import JOB_REQUISITION, Candidate


class EmbeddingRanker:
    """Rank candidates by cosine similarity between résumé and job requisition."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", query: str | None = None):
        from sentence_transformers import SentenceTransformer  # imported lazily
        self.model = SentenceTransformer(model_name)
        self.query = query or JOB_REQUISITION
        self.model_name = model_name
        self._scores: dict[str, float] = {}

    def __call__(self, pool: list[Candidate]) -> list[Candidate]:
        from sentence_transformers import util
        q = self.model.encode(self.query, convert_to_tensor=True)
        docs = self.model.encode([c.resume for c in pool], convert_to_tensor=True)
        sims = util.cos_sim(q, docs)[0]
        scored = sorted(zip(pool, sims.tolist()), key=lambda t: t[1], reverse=True)
        self._scores = {c.cid: float(s) for c, s in scored}
        return [c for c, _ in scored]

    def scores(self) -> dict[str, float]:
        """Similarity per candidate id from the most recent ranking."""
        return dict(self._scores)


def build_embedding_ranker(model_name: str = "all-MiniLM-L6-v2"):
    """Return an ``EmbeddingRanker``, or ``None`` if the model is unavailable.

    Returning ``None`` (rather than a stand-in) keeps the retrieval-bias track
    honestly skippable — see the notebook, which reports it as *not run* rather
    than as a clean result.
    """
    try:
        return EmbeddingRanker(model_name)
    except Exception as exc:                     # missing package / offline model
        print(f"  ⚠️  Embedding ranker unavailable ({type(exc).__name__}) — "
              f"retrieval-bias track will be skipped.")
        return None
