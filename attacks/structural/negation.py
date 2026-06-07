"""
NegationInjection — Structural-level adversarial attack.

Strategy
--------
Inserts or removes negation in the main clause of the input sentence.

  • If a negation is already present (``not`` standalone or ``n't``
    contraction) → remove it to flip logical polarity.
  • If no negation is present:
      - Find a modal or auxiliary verb and insert ``not`` after it.
      - If only a main verb (VBZ / VBP) is found, apply do-support:
        ``"confirms"`` → ``"does not confirm"``.

Why this matters
----------------
LLMs are notoriously brittle at tracking logical negation.  A sentence
like ``"the film is not bad"`` (positive sentiment) can be flipped to
``"the film is bad"`` with a single token removal, and vice versa.
This attack tests whether a model's decision boundary respects negation
or relies on surface sentiment cues (e.g. the word "bad" regardless of
the "not" preceding it).

Note on SST-2 pre-tokenisation
------------------------------
The SST-2 dev set uses space-tokenised text where contractions are
pre-split: ``"ca n't"`` instead of ``"can't"``.  This class handles
both forms.

Industry relevance
------------------
MITRE ATLAS AML.T0043 (Craft Adversarial Data), AML.T0015 (Evade ML Model)
NIST AI 600-1 — Information Integrity (robustness to distribution shift)
OWASP LLM Top 10 2025 — LLM09 Misinformation
"""

from __future__ import annotations

import re

import nltk
from nltk import pos_tag, word_tokenize

# Ensure required NLTK resources are available
for _pkg, _kind in [
    ("punkt_tab",                    "tokenizers"),
    ("averaged_perceptron_tagger_eng", "taggers"),
]:
    try:
        nltk.data.find(f"{_kind}/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)


# ── Lookup tables ──────────────────────────────────────────────────────────

# Contractions that carry negation → their affirmative base form
# Handles both standard and SST-2 pre-tokenised forms
_NEGATED_CONTRACTION: dict[str, str] = {
    "can't":    "can",
    "ca":       "can",       # SST-2: "ca n't" → "can"
    "won't":    "will",
    "wo":       "will",      # SST-2: "wo n't"
    "don't":    "do",
    "doesn't":  "does",
    "didn't":   "did",
    "isn't":    "is",
    "aren't":   "are",
    "wasn't":   "was",
    "weren't":  "were",
    "hasn't":   "has",
    "haven't":  "have",
    "hadn't":   "had",
    "wouldn't": "would",
    "couldn't": "could",
    "shouldn't": "should",
    "mightn't": "might",
    "mustn't":  "must",
    "needn't":  "need",
}

# Auxiliaries / modals → form after inserting "not"
_NEGATION_INSERT: dict[str, str] = {
    "is":     "is not",
    "are":    "are not",
    "was":    "was not",
    "were":   "were not",
    "has":    "has not",
    "have":   "have not",
    "had":    "had not",
    "will":   "will not",
    "would":  "would not",
    "can":    "cannot",
    "could":  "could not",
    "should": "should not",
    "may":    "may not",
    "might":  "might not",
    "must":   "must not",
    "do":     "do not",
    "does":   "does not",
    "did":    "did not",
    "shall":  "shall not",
}

# POS tags for VBZ (3rd person singular present) and VBP (non-3rd person present)
_MAIN_VERB_TAGS = {"VBZ", "VBP"}
_DO_SUPPORT = {"VBZ": "does not", "VBP": "do not"}


class NegationInjection:
    """
    Insert or remove negation to flip the logical polarity of a sentence.

    The attack applies the first matching rule in this priority order:

    1. Remove a negation contraction (``n't`` suffix or ``won't`` / ``can't``).
    2. Remove a standalone ``not`` after an auxiliary.
    3. Insert ``not`` after the first eligible auxiliary / modal.
    4. Apply do-support to the first main verb (VBZ / VBP).

    If none of the rules applies (e.g. a very short or unparseable sentence),
    the original text is returned unchanged.

    Parameters
    ----------
    max_scan_tokens : int
        Maximum number of tokens to scan when searching for an auxiliary or
        verb (default 12).  Prevents modifying deeply-buried clauses.
    """

    def __init__(self, max_scan_tokens: int = 12):
        self.max_scan_tokens = max_scan_tokens

    # ── Public API ────────────────────────────────────────────────────────────

    def attack(self, text: str) -> str:
        """
        Apply the negation perturbation.

        Parameters
        ----------
        text : str
            Original input sentence.

        Returns
        -------
        str
            Negation-modified sentence, or the original if no rule fires.
        """
        result = (
            self._try_remove_contraction(text)
            or self._try_remove_standalone_not(text)
            or self._try_insert_after_auxiliary(text)
            or self._try_do_support(text)
        )
        return result if result else text

    def attack_batch(self, texts: list[str]) -> list[str]:
        return [self.attack(t) for t in texts]

    def __repr__(self) -> str:
        return "NegationInjection(level='structural', method='negation_flip')"

    # ── Rules ─────────────────────────────────────────────────────────────────

    def _try_remove_contraction(self, text: str) -> str | None:
        """
        Rule 1: Remove a negation contraction.

        Handles ``"ca n't"`` (SST-2 format) and ``"can't"`` (standard).

        SST-2 pre-tokenises "can't" as "ca n't" — the "n't" token takes
        priority over the bare "ca" match so we scan for "n't" first.
        """
        tokens = text.split()

        # Pass 1: SST-2 style — "n't" as a separate token
        for i, tok in enumerate(tokens):
            if tok.lower() == "n't" and i > 0:
                base = _NEGATED_CONTRACTION.get(tokens[i - 1].lower())
                if base:
                    new_tokens = tokens[: i - 1] + [base] + tokens[i + 1:]
                else:
                    new_tokens = tokens[: i - 1] + tokens[i + 1:]
                return " ".join(new_tokens)

        # Pass 2: standard contractions as a single token ("can't", "won't", …)
        for i, tok in enumerate(tokens):
            low = tok.lower().rstrip(".,!?;:")
            if low in _NEGATED_CONTRACTION:
                new_tokens = tokens[:]
                new_tokens[i] = _NEGATED_CONTRACTION[low]
                return " ".join(new_tokens)

        return None

    def _try_remove_standalone_not(self, text: str) -> str | None:
        """
        Rule 2: Remove a standalone ``not`` that follows an auxiliary.
        """
        tokens = text.split()
        for i, tok in enumerate(tokens):
            if tok.lower() == "not" and i > 0:
                prev = tokens[i - 1].lower()
                if prev in _NEGATION_INSERT:
                    new_tokens = tokens[:i] + tokens[i + 1:]
                    return " ".join(new_tokens)
        return None

    def _try_insert_after_auxiliary(self, text: str) -> str | None:
        """
        Rule 3: Insert ``not`` after the first auxiliary / modal verb.
        """
        tokens = text.split()
        for i, tok in enumerate(tokens[: self.max_scan_tokens]):
            low = tok.lower().rstrip(".,!?;:")
            if low in _NEGATION_INSERT:
                negated = _NEGATION_INSERT[low]
                new_tokens = tokens[:i] + [negated] + tokens[i + 1:]
                return " ".join(new_tokens)
        return None

    def _try_do_support(self, text: str) -> str | None:
        """
        Rule 4: Apply do-support to the first present-tense main verb.

        ``"it confirms fincher's status"``
        → ``"it does not confirm fincher's status"``

        The verb is lemmatised to its base form so "confirms" (VBZ) becomes
        "confirm" rather than the grammatically incorrect "does not confirms".
        """
        try:
            from nltk.stem import WordNetLemmatizer
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            lemmatiser = WordNetLemmatizer()
        except Exception:
            return None

        space_tokens = text.split()

        for i, (word, tag) in enumerate(tagged[: self.max_scan_tokens]):
            if tag not in _MAIN_VERB_TAGS:
                continue
            do_str   = _DO_SUPPORT[tag]                             # "does not" or "do not"
            base     = lemmatiser.lemmatize(word.lower(), pos="v")  # "confirms" → "confirm"

            # Match against the original space-split tokens
            for j, stok in enumerate(space_tokens[: self.max_scan_tokens + 2]):
                if stok.lower().rstrip(".,") == word.lower():
                    new_tokens = space_tokens[:j] + [do_str, base] + space_tokens[j + 1:]
                    return " ".join(new_tokens)
        return None
