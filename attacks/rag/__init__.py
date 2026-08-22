"""
attacks.rag — Data leakage in a retrieval-augmented assistant (NB06b use case).

The use-case half of the data red-teaming pair. NB06 tests whether a *model*
discloses secrets placed in its prompt; this tests whether a *deployment* hands
a user documents they are not entitled to — a risk that does not exist until
retrieval is added, and one the model itself never sees.
"""

from .corpus import (
    Document, build_corpus, corpus_summary, may_read,
    CLEARANCES, TIERS, TIER_RANK, ENCODER_CHAR_BUDGET,
)
from .index import (
    VectorIndex, Retrieved, retrieval_leak_check,
    ARCHITECTURES, ARCHITECTURE_NOTE,
)
from .probes import build_probes, probe_summary
from .assistant import (RagAssistant, RagResult, score_response,
                        rescore_rag_results, is_not_found, is_partial_answer)
from .runner import (
    run_boundary_track, run_poison_track, build_poison_docs,
    poison_queries_from_probes, index_with_poison, HARNESS_VERSION,
)

__all__ = [
    "Document", "build_corpus", "corpus_summary", "may_read",
    "CLEARANCES", "TIERS", "TIER_RANK", "ENCODER_CHAR_BUDGET",
    "VectorIndex", "Retrieved", "retrieval_leak_check",
    "ARCHITECTURES", "ARCHITECTURE_NOTE",
    "build_probes", "probe_summary",
    "RagAssistant", "RagResult", "score_response",
    "rescore_rag_results", "is_not_found", "is_partial_answer",
    "run_boundary_track", "run_poison_track", "build_poison_docs",
    "poison_queries_from_probes", "index_with_poison", "HARNESS_VERSION",
]
