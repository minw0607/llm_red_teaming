"""
Homoglyph — Structural-level adversarial attack.

Strategy
--------
Replaces one ASCII character in a content word with a visually identical
Unicode lookalike (a homoglyph).  The resulting text is indistinguishable
to a human reader but the underlying byte sequence differs.

This attack targets:
  • Keyword-based content filters  (string matching breaks on the different
    codepoint even though the glyph looks the same)
  • Tokeniser assumptions  (most LLM tokenisers fall back to <unk> or byte
    tokens for non-ASCII codepoints, causing perplexity to spike)

The attack is intentionally minimal — one character in one content word —
to maximise stealth score (semantic similarity stays at ~1.0).

Reference
---------
The Unicode Security Consortium documents homoglyph confusables at:
https://unicode.org/reports/tr39/#Confusable_Detection

Industry relevance
------------------
MITRE ATLAS AML.T0043 (Craft Adversarial Data)
NIST AI 600-1 — Information Security (adversarial inputs)
OWASP LLM Top 10 2025 — LLM01 Prompt Injection, LLM09 Misinformation
"""

from __future__ import annotations

# ASCII → visually identical Unicode lookalike (Cyrillic / Greek / Latin Extended)
# Each replacement is a single codepoint that renders identically in most fonts.
HOMOGLYPH_MAP: dict[str, str] = {
    "a": "а",   # Cyrillic а  (U+0430)
    "c": "с",   # Cyrillic с  (U+0441)
    "e": "е",   # Cyrillic е  (U+0435)
    "i": "і",   # Ukrainian і (U+0456)
    "j": "ј",   # Cyrillic ј  (U+0458)
    "o": "о",   # Cyrillic о  (U+043E)
    "p": "р",   # Cyrillic р  (U+0440)
    "s": "ѕ",   # Cyrillic ѕ  (U+0455)
    "x": "х",   # Cyrillic х  (U+0445)
    "y": "у",   # Cyrillic у  (U+0443)
}

# Short words and punctuation to skip when selecting a target word
_SKIP = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
         "of", "for", "is", "are", "was", "were", "be", "it", "its"}


class Homoglyph:
    """
    Replace one ASCII character in a content word with a visually identical
    Unicode lookalike.

    Parameters
    ----------
    min_word_len : int
        Minimum character length for a word to be eligible (default 4).
        Prevents substituting very short words where the change is more
        noticeable.
    """

    def __init__(self, min_word_len: int = 4):
        self.min_word_len = min_word_len

    def attack(self, text: str) -> str:
        """
        Apply the Homoglyph perturbation.

        Scans words left-to-right, skipping stop words and short tokens,
        and replaces the first eligible character.

        Parameters
        ----------
        text : str
            Original input sentence.

        Returns
        -------
        str
            Perturbed sentence with one homoglyph substitution, or the
            original string if no eligible substitution is found.
        """
        words = text.split()
        for i, word in enumerate(words):
            clean = word.strip(".,!?;:\"'()")
            if (
                len(clean) < self.min_word_len
                or clean.lower() in _SKIP
                or not clean.isascii()
            ):
                continue

            for j, char in enumerate(clean):
                lc = char.lower()
                if lc in HOMOGLYPH_MAP:
                    glyph = HOMOGLYPH_MAP[lc]
                    # Preserve original case position in the stripped word
                    new_clean = clean[:j] + glyph + clean[j + 1:]
                    # Re-attach leading/trailing punctuation
                    prefix = word[: len(word) - len(word.lstrip(".,!?;:\"'()"))]
                    suffix = word[len(word.rstrip(".,!?;:\"'()")):]
                    words[i] = prefix + new_clean + suffix
                    return " ".join(words)

        return text  # no eligible character found

    def attack_batch(self, texts: list[str]) -> list[str]:
        """Apply attack to a list of sentences."""
        return [self.attack(t) for t in texts]

    def __repr__(self) -> str:
        return (
            f"Homoglyph(level='structural', method='unicode_lookalike', "
            f"min_word_len={self.min_word_len})"
        )
