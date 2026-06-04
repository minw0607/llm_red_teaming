"""
SemanticAttack — Semantic-level adversarial attack.

Strategy
--------
Uses NLTK part-of-speech tagging to identify the most important content word
(noun, verb, adjective, or adverb) and replaces it with a WordNet synonym
that matches the same POS.  Single-word, non-underscore synonyms are
preferred to maintain fluency.  Limited to one substitution for clarity.

This attack preserves meaning better than character- or word-level attacks
and is closer to human paraphrasing.
"""

from __future__ import annotations

import nltk
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize

for pkg in ("punkt", "averaged_perceptron_tagger", "wordnet", "omw-1.4",
            "averaged_perceptron_tagger_eng", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}") if "punkt" in pkg else nltk.data.find(f"taggers/{pkg}") if "tagger" in pkg else nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)


def _wordnet_pos(treebank_tag: str):
    """Convert a Penn Treebank POS tag to a WordNet POS constant."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return None


class SemanticAttack:
    """
    POS-aware synonym substitution preserving overall sentence meaning.

    Parameters
    ----------
    min_word_len : int
        Minimum character length of a word to be eligible (default 3).
    """

    def __init__(self, min_word_len: int = 3):
        self.min_word_len = min_word_len

    def attack(self, text: str) -> str:
        """
        Replace the first eligible content word with a WordNet synonym.

        Parameters
        ----------
        text : str
            The original input sentence.

        Returns
        -------
        str
            The perturbed sentence, or the original if no substitution found.
        """
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        new_tokens = tokens[:]

        for i, (word, tag) in enumerate(tagged):
            if len(word) < self.min_word_len:
                continue
            wn_pos = _wordnet_pos(tag)
            if wn_pos is None:
                continue

            synsets = wordnet.synsets(word, pos=wn_pos)
            candidates = [
                lemma.name()
                for s in synsets
                for lemma in s.lemmas()
                if lemma.name().lower() != word.lower()
                and "_" not in lemma.name()
                and len(lemma.name()) > 2
            ]
            if candidates:
                new_tokens[i] = candidates[0]
                break  # one substitution only

        return " ".join(new_tokens)

    def attack_batch(self, texts: list[str]) -> list[str]:
        """Apply attack to a list of sentences."""
        return [self.attack(t) for t in texts]

    def __repr__(self) -> str:
        return "SemanticAttack(level='semantic', method='pos_aware_synonym')"
