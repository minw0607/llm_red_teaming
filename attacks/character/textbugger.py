"""
TextBugger — Character-level adversarial attack.

Strategy
--------
Substitutes a single character near the start of the input with a random
ASCII letter, producing a plausible-looking typo that can fool models
relying on surface-form features.

Reference
---------
Li et al. (2019) "TextBugger: Generating Adversarial Text Against Real-world
Applications." NDSS 2019. https://arxiv.org/abs/1812.05271
"""

import random
import string


class TextBugger:
    """
    Introduce a single random character substitution at position 3
    (near the start of the string).

    Parameters
    ----------
    seed : int | None
        Optional random seed for reproducibility.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def attack(self, text: str) -> str:
        """
        Apply the TextBugger perturbation.

        Parameters
        ----------
        text : str
            The original input sentence.

        Returns
        -------
        str
            The perturbed sentence.
        """
        if len(text) <= 3:
            return text
        replacement = self._rng.choice(string.ascii_letters)
        return text[:3] + replacement + text[4:]

    def attack_batch(self, texts: list[str]) -> list[str]:
        """Apply attack to a list of sentences."""
        return [self.attack(t) for t in texts]

    def __repr__(self) -> str:
        return "TextBugger(level='character', method='char_substitution')"
