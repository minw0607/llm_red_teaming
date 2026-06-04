"""
StressTest — Sentence-level adversarial attack.

Strategy
--------
Appends a tautological or logically redundant phrase to the sentence,
simulating the kind of noise injection that can confuse models relying
on shallow pattern matching or sentence-level statistics.

Reference
---------
Naik et al. (2018) "Stress Test Evaluation for Natural Language Inference."
COLING 2018. https://arxiv.org/abs/1806.00692
"""

import random

# Tautological suffixes — logically true but semantically vacuous
TAUTOLOGIES = [
    " and true is true",
    " and false is not true",
    " and true is true and true is true",
    " and true is true and true is true and true is true",
    " and one plus one is not three",
]


class StressTest:
    """
    Append a tautological phrase to stress-test inference robustness.

    Parameters
    ----------
    tautologies : list[str] | None
        Custom list of suffix strings to choose from.
        Defaults to the built-in tautology set.
    seed : int | None
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        tautologies: list[str] | None = None,
        seed: int | None = None,
    ):
        self.tautologies = tautologies if tautologies is not None else TAUTOLOGIES
        self._rng = random.Random(seed)

    def attack(self, text: str) -> str:
        """
        Append a random tautological suffix.

        Parameters
        ----------
        text : str
            The original input sentence.

        Returns
        -------
        str
            The input with a tautological suffix appended.
        """
        suffix = self._rng.choice(self.tautologies)
        return text.rstrip() + suffix

    def attack_batch(self, texts: list[str]) -> list[str]:
        """Apply attack to a list of sentences."""
        return [self.attack(t) for t in texts]

    def __repr__(self) -> str:
        return "StressTest(level='sentence', method='tautology_append')"
