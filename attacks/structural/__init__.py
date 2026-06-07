"""
attacks/structural/
Structural-level adversarial attacks that modify text at the byte,
character, or linguistic-structure level without changing lexical content.

Attacks
-------
Homoglyph        — Replace ASCII characters with Unicode lookalikes
BackTranslation  — EN → pivot language → EN paraphrase
NegationInjection — Insert or remove negation to flip logical polarity
"""

from .homoglyph import Homoglyph
from .back_translation import BackTranslation
from .negation import NegationInjection

__all__ = ["Homoglyph", "BackTranslation", "NegationInjection"]
