from .metrics import (
    compute_attack_summary,
    flag_human_review,
    risk_score,
    accuracy_drop,
    attack_success_rate,
    asr_by_category,
    verdict_summary,
    adversarial_report,
    jailbreak_report,
)
from .strongreject import strongreject_report, strongreject_score
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
from .executive import generate_executive_summary, render_executive_html
from .jb_executive import generate_jailbreak_summary, render_jailbreak_html, compute_jailbreak_metrics
from .injection_metrics import (
    override_rate, override_by, injection_summary, injection_report,
    explain_overrides, print_override_report,
)
from .injection_executive import generate_injection_summary, render_injection_html, compute_injection_metrics
from .fairness_metrics import (
    bbq_category_summary, bbq_overall, bbq_report,
    cf_flip_summary, cf_flip_rate, cf_parity_by_dimension, explain_cf_flips, print_bias_report,
    stereotype_errors, stereotype_error_rate, explain_bias_cases, explain_notable_cases,
)
from .fairness_executive import generate_fairness_summary, render_fairness_html, compute_fairness_metrics
from .sanity import sanity_check, render_sanity_html

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
    # Executive summary
    "generate_executive_summary",
    "render_executive_html",
    # Jailbreak executive summary
    "generate_jailbreak_summary",
    "render_jailbreak_html",
    "compute_jailbreak_metrics",
    # Prompt injection
    "override_rate",
    "override_by",
    "injection_summary",
    "injection_report",
    "explain_overrides",
    "print_override_report",
    "generate_injection_summary",
    "render_injection_html",
    "compute_injection_metrics",
    # Sanity check
    "sanity_check",
    "render_sanity_html",
    # Legacy / jailbreak helpers
    "accuracy_drop",
    "attack_success_rate",
    "asr_by_category",
    "verdict_summary",
    "adversarial_report",
    "jailbreak_report",
    # StrongREJECT graded scoring
    "strongreject_report",
    "strongreject_score",
    # Fairness & bias
    "bbq_category_summary",
    "bbq_overall",
    "bbq_report",
    "cf_flip_summary",
    "cf_flip_rate",
    "cf_parity_by_dimension",
    "explain_cf_flips",
    "print_bias_report",
    "stereotype_errors",
    "stereotype_error_rate",
    "explain_bias_cases",
    "explain_notable_cases",
    "generate_fairness_summary",
    "render_fairness_html",
    "compute_fairness_metrics",
]
