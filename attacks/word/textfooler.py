"""
TextFooler — Word-level adversarial attack.

Strategy
--------
Scans the input words and replaces the first non-stopword, non-short word
with a WordNet synonym chosen from the top-3 candidates.  Preserves
approximate semantics while changing surface form.

Reference
---------
Jin et al. (2019) "Is BERT Really Robust? A Strong Baseline for Natural
Language Attack on Text Classification and Entailment."
AAAI 2020. https://arxiv.org/abs/1907.11932
"""

import random
from nltk.corpus import wordnet
import nltk

# Ensure required NLTK data is present
for pkg in ("wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = {
    "the", "is", "a", "an", "and", "to", "in", "of",
    "on", "for", "with", "it", "this", "that", "was",
    "are", "be", "has", "had", "at", "by", "from",
}


class TextFooler:
    """
    Replace the first eligible word with a WordNet synonym.

    Parameters
    ----------
    stopwords : set[str] | None
        Words to skip. Defaults to a built-in English stopword list.
    min_word_len : int
        Minimum word length to consider for substitution (default 3).
    top_k : int
        Number of synonym candidates to sample from (default 3).
    seed : int | None
        Optional random seed.
    """

    def __init__(
        self,
        stopwords: set[str] | None = None,
        min_word_len: int = 3,
        top_k: int = 3,
        seed: int | None = None,
    ):
        self.stopwords = stopwords if stopwords is not None else STOPWORDS
        self.min_word_len = min_word_len
        self.top_k = top_k
        self._rng = random.Random(seed)

    def _get_synonyms(self, word: str) -> list[str]:
        """Return a deduplicated list of synonyms for *word* via WordNet."""
        synsets = wordnet.synsets(word)
        seen: set[str] = set()
        results: list[str] = []
        for s in synsets:
            for lemma in s.lemmas():
                candidate = lemma.name().replace("_", " ")
                if candidate.lower() != word.lower() and candidate not in seen:
                    seen.add(candidate)
                    results.append(candidate)
        return results

    def attack(self, text: str) -> str:
        """
        Apply the TextFooler perturbation.

        Parameters
        ----------
        text : str
            The original input sentence.

        Returns
        -------
        str
            The perturbed sentence (first eligible word replaced).
        """
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in self.stopwords or len(word) < self.min_word_len:
                continue
            synonyms = self._get_synonyms(word)
            if synonyms:
                words[i] = self._rng.choice(synonyms[: self.top_k])
                break
        return " ".join(words)

    def attack_batch(self, texts: list[str]) -> list[str]:
        """Apply attack to a list of sentences."""
        return [self.attack(t) for t in texts]

    def __repr__(self) -> str:
        return f"TextFooler(level='word', top_k={self.top_k})"
