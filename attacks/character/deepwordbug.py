"""
DeepWordBug — Character-level adversarial attack.

Strategy
--------
Selects a random position in the character sequence and applies one of three
operations: insert a random character, delete a character, or swap two
adjacent characters. Mimics realistic typing errors.

Reference
---------
Gao et al. (2018) "Black-box Generation of Adversarial Text Sequences to
Evade Deep Learning Classifiers." IEEE S&P Workshops 2018.
https://arxiv.org/abs/1801.04354
"""

import random
import string


class DeepWordBug:
    """
    Apply a single character-level operation (insert / delete / swap)
    at a randomly chosen position.

    Parameters
    ----------
    seed : int | None
        Optional random seed for reproducibility.
    """

    OPERATIONS = ("insert", "delete", "swap")

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def attack(self, text: str) -> str:
        """
        Apply one random character-level edit.

        Parameters
        ----------
        text : str
            The original input sentence.

        Returns
        -------
        str
            The perturbed sentence.
        """
        chars = list(text)
        if len(chars) <= 3:
            return text

        idx = self._rng.randint(1, len(chars) - 2)
        op = self._rng.choice(self.OPERATIONS)

        if op == "insert":
            chars.insert(idx, self._rng.choice(string.ascii_letters))
        elif op == "delete":
            chars.pop(idx)
        elif op == "swap" and idx < len(chars) - 1:
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

        return "".join(chars)

    def attack_batch(self, texts: list[str]) -> list[str]:
        """Apply attack to a list of sentences."""
        return [self.attack(t) for t in texts]

    def __repr__(self) -> str:
        return "DeepWordBug(level='character', method='insert|delete|swap')"
