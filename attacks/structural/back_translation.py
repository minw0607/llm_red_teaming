"""
BackTranslation — Structural-level adversarial attack.

Strategy
--------
Translates the input text from English into a pivot language (default:
German) and then back into English using Helsinki-NLP MarianMT models.
The round-trip produces a fluent paraphrase that:

  • Changes word order and lexical choice naturally (no WordNet lookup)
  • Preserves the overall meaning (high semantic similarity)
  • Reads like normal English (low perplexity impact)
  • Has high surface edit distance (tests robustness to genuine rewrites)

This is a cleaner alternative to WordNet-based synonym attacks because
the substitutions are driven by translation statistics rather than
hand-crafted synonym sets, and the output never includes scientific
Latin names or awkward synonyms.

Models used (downloaded automatically on first call, ~300 MB each)
------------------------------------------------------------------
  EN → DE : Helsinki-NLP/opus-mt-en-de
  DE → EN : Helsinki-NLP/opus-mt-de-en

Other pivot languages can be specified via ``pivot_lang``.

Reference
---------
MarianMT: Junczys-Dowmunt et al. (2018) https://arxiv.org/abs/1804.00344
Helsinki-NLP OPUS-MT: https://github.com/Helsinki-NLP/OPUS-MT

Industry relevance
------------------
MITRE ATLAS AML.T0043 (Craft Adversarial Data), AML.T0015 (Evade ML Model)
NIST AI 600-1 — Information Integrity (robustness to distribution shift)
OWASP LLM Top 10 2025 — LLM09 Misinformation
"""

from __future__ import annotations


class BackTranslation:
    """
    Round-trip machine translation paraphrase attack.

    Parameters
    ----------
    pivot_lang : str
        ISO 639-1 code for the pivot language (default ``"de"``).
        Must have a Helsinki-NLP model pair available:
        ``opus-mt-en-{pivot}`` and ``opus-mt-{pivot}-en``.
    max_length : int
        Maximum token length for translation (default 512).
    """

    def __init__(self, pivot_lang: str = "de", max_length: int = 512):
        self.pivot_lang  = pivot_lang
        self.max_length  = max_length
        self._fwd_name   = f"Helsinki-NLP/opus-mt-en-{pivot_lang}"
        self._bwd_name   = f"Helsinki-NLP/opus-mt-{pivot_lang}-en"
        # Lazy-loaded on first call to attack()
        self._fwd_model     = None
        self._fwd_tokenizer = None
        self._bwd_model     = None
        self._bwd_tokenizer = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load(self) -> None:
        """Download and cache both MarianMT models (first call only)."""
        if self._fwd_model is not None:
            return  # already loaded

        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for BackTranslation. "
                "Install with: pip install transformers sentencepiece"
            ) from e

        print(f"Loading BackTranslation models ({self._fwd_name} · {self._bwd_name}) …")
        self._fwd_tokenizer = MarianTokenizer.from_pretrained(self._fwd_name)
        self._fwd_model     = MarianMTModel.from_pretrained(self._fwd_name)
        self._bwd_tokenizer = MarianTokenizer.from_pretrained(self._bwd_name)
        self._bwd_model     = MarianMTModel.from_pretrained(self._bwd_name)
        self._fwd_model.eval()
        self._bwd_model.eval()
        print("✅ BackTranslation models ready.")

    def _translate(self, text: str, model, tokenizer) -> str:
        """Translate *text* with the given MarianMT model/tokenizer pair."""
        import torch

        inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        with torch.no_grad():
            translated_ids = model.generate(**inputs)
        return tokenizer.decode(translated_ids[0], skip_special_tokens=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def attack(self, text: str) -> str:
        """
        Apply the back-translation perturbation.

        Parameters
        ----------
        text : str
            Original input sentence.

        Returns
        -------
        str
            Round-trip translated paraphrase, or the original string if
            translation fails or produces an empty result.
        """
        if not text or not text.strip():
            return text

        self._load()

        try:
            pivot     = self._translate(text, self._fwd_model, self._fwd_tokenizer)
            back      = self._translate(pivot, self._bwd_model, self._bwd_tokenizer)
            return back.strip() if back.strip() else text
        except Exception:
            return text  # graceful fallback — never raise from attack()

    def attack_batch(self, texts: list[str]) -> list[str]:
        """Apply attack to a list of sentences."""
        self._load()
        return [self.attack(t) for t in texts]

    def __repr__(self) -> str:
        return (
            f"BackTranslation(level='structural', pivot='{self.pivot_lang}', "
            f"models='{self._fwd_name}')"
        )
