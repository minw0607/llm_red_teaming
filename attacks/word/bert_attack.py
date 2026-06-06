"""
BERTAttack — Word-level adversarial attack using masked language modelling.

Strategy
--------
For each token position, masks the word and queries a BERT fill-mask
pipeline for the top-k replacement candidates.  A SentenceTransformer
encoder then filters candidates by cosine similarity to the original
sentence — keeping only substitutions that preserve overall meaning above
a configurable threshold.  The replacement that achieves the highest
similarity (while still changing the text) is returned.

Reference
---------
Li et al. (2020) "BERT-ATTACK: Adversarial Attack Against BERT Using BERT."
EMNLP 2020. https://arxiv.org/abs/2004.09984
"""

from __future__ import annotations

import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util


def _resolve_model_path(env_var: str, default: str) -> str:
    """
    Read a model path from an env var, falling back to *default*.

    python-dotenv does NOT strip inline comments when a value is otherwise
    blank (e.g. ``LOCAL_BERT_PATH=   # e.g. ./models/bert``).  This helper
    discards any value that starts with '#' or is whitespace-only, so a
    commented-out / empty env var correctly resolves to the default.
    """
    raw = os.getenv(env_var, "").strip()
    if not raw or raw.startswith("#"):
        return default
    return raw


class BERTAttack:
    """
    Context-aware word substitution via BERT fill-mask.

    Parameters
    ----------
    bert_model_path : str
        Path or HuggingFace hub name for the masked-LM model.
        Defaults to the ``LOCAL_BERT_PATH`` env variable, then
        ``"bert-base-uncased"``.
    sentence_model_path : str
        Path or HuggingFace hub name for the sentence encoder.
        Defaults to the ``LOCAL_SENTENCE_TRANSFORMER_PATH`` env variable,
        then ``"sentence-transformers/all-MiniLM-L6-v2"``.
    top_k : int
        Number of fill-mask candidates per position (default 5).
    sim_threshold : float
        Minimum cosine similarity required to accept a substitution (0–1).
        Default 0.85.
    """

    def __init__(
        self,
        bert_model_path: str | None = None,
        sentence_model_path: str | None = None,
        top_k: int = 5,
        sim_threshold: float = 0.85,
    ):
        bert_path = bert_model_path or _resolve_model_path(
            "LOCAL_BERT_PATH", "bert-base-uncased"
        )
        sent_path = sentence_model_path or _resolve_model_path(
            "LOCAL_SENTENCE_TRANSFORMER_PATH",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.top_k = top_k
        self.sim_threshold = sim_threshold

        self._mlm = pipeline(
            "fill-mask",
            model=bert_path,
            tokenizer=bert_path,
            top_k=top_k,
        )
        self._mask_token: str = self._mlm.tokenizer.mask_token
        self._encoder = SentenceTransformer(sent_path)

    def attack(self, text: str) -> str:
        """
        Apply the BERTAttack perturbation.

        Parameters
        ----------
        text : str
            The original input sentence.

        Returns
        -------
        str
            The best perturbed sentence found, or the original if no
            valid substitution exceeds the similarity threshold.
        """
        words = text.split()
        if len(words) < 2:
            return text

        original_emb = self._encoder.encode(text, convert_to_tensor=True)
        best_text = text
        best_score: float = 0.0

        for i, original_word in enumerate(words):
            words[i] = self._mask_token
            masked = " ".join(words)

            try:
                predictions = self._mlm(masked)
            except Exception as exc:
                print(f"[BERTAttack] fill-mask error at position {i}: {exc}")
                words[i] = original_word
                continue

            for pred in predictions:
                candidate_word = pred["token_str"]
                words[i] = candidate_word
                candidate_text = " ".join(words)

                sim = util.cos_sim(
                    self._encoder.encode(candidate_text, convert_to_tensor=True),
                    original_emb,
                ).item()

                if (
                    sim >= self.sim_threshold
                    and sim > best_score
                    and candidate_text.lower() != text.lower()
                ):
                    best_text = candidate_text
                    best_score = sim

            words[i] = original_word  # reset for next iteration

        return best_text

    def attack_batch(self, texts: list[str]) -> list[str]:
        """Apply attack to a list of sentences."""
        return [self.attack(t) for t in texts]

    def __repr__(self) -> str:
        return (
            f"BERTAttack(level='word', top_k={self.top_k}, "
            f"sim_threshold={self.sim_threshold})"
        )
