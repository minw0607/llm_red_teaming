from .metrics import (
    compute_attack_summary,
    flag_human_review,
    risk_score,
    accuracy_drop,
    attack_success_rate,
    verdict_summary,
    adversarial_report,
    jailbreak_report,
)
from .adversarial_eval import run_all_attacks
from .stealth import (
    load_perplexity_model,
    compute_perplexity,
    edit_similarity,
    composite_stealth_score,
    add_stealth_components,
)
from .plots import plot_adversarial_summary, DEFAULT_LEVEL_COLORS
from .display import display_human_review
from .regulatory import map_to_regulations, regulatory_report, render_regulatory_heatmap

__all__ = [
    # Core evaluation pipeline
    "run_all_attacks",
    "compute_attack_summary",
    "flag_human_review",
    "risk_score",
    # Composite stealth
    "load_perplexity_model",
    "compute_perplexity",
    "edit_similarity",
    "composite_stealth_score",
    "add_stealth_components",
    # Visualisation
    "plot_adversarial_summary",
    "DEFAULT_LEVEL_COLORS",
    # Notebook display
    "display_human_review",
    # Regulatory mapping
    "map_to_regulations",
    "regulatory_report",
    "render_regulatory_heatmap",
    # Legacy / jailbreak helpers
    "accuracy_drop",
    "attack_success_rate",
    "verdict_summary",
    "adversarial_report",
    "jailbreak_report",
]
