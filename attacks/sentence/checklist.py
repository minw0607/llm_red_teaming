"""
CheckList — Sentence-level adversarial attack.

Strategy
--------
Appends a random alphanumeric token to the end of the sentence.  Tests
whether a model is robust to trailing noise that has no semantic content.

Reference
---------
Ribeiro et al. (2020) "Beyond Accuracy: Behavioral Testing of NLP Models
with CheckList." ACL 2020. https://arxiv.org/abs/2005.04118
"""

import random
import string


class CheckList:
    """
    Append a random alphanumeric string to the end of the input.

    Parameters
    ----------
    token_length : int
        Length of the appended noise token (default 10).
    seed : int | None
        Optional random seed for reproducibility.
    """

    def __init__(self, token_length: int = 10, seed: int | None = None):
        self.token_length = token_length
        self._rng = random.Random(seed)

    def _random_token(self) -> str:
        alphabet = string.ascii_letters + string.digits
        return "".join(self._rng.choices(alphabet, k=self.token_length))

    def attack(self, text: str) -> str:
        """
        Append a random noise token.

        Parameters
        ----------
        text : str
            The original input sentence.

        Returns
        -------
        str
            The input with a trailing noise token.
        """
        return f"{text.rstrip()} {self._random_token()}"

    def attack_batch(self, texts: list[str]) -> list[str]:
        """Apply attack to a list of sentences."""
        return [self.attack(t) for t in texts]

    def __repr__(self) -> str:
        return f"CheckList(level='sentence', token_length={self.token_length})"
