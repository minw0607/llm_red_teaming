"""
evaluate/regulatory.py — Dynamic regulatory impact mapping from adversarial test results.

Translates actual evaluation findings into specific citations from:
  • NIST AI 600-1  (Jul 2024) — GenAI Profile
  • MITRE ATLAS              — Adversarial Threat Landscape for AI Systems
  • OWASP LLM Top 10 (2025)  — Application-layer LLM security risks
  • EU AI Act (2024/1689)    — High-risk AI system obligations

Unlike the static attack-to-framework tables in the notebook overview,
these functions consume live run results (summary_df, review_df) and
emit *finding-level* entries that include actual metric values, severity
ratings drawn from the data, and recommended remediation actions.

Functions
---------
map_to_regulations       : summary_df + review_df → findings DataFrame
regulatory_report        : pretty-print the findings report to stdout,
                           return the findings DataFrame
render_regulatory_heatmap: matplotlib heatmap — which frameworks each
                           finding implicates (HIGH / MEDIUM / —)

Usage
-----
    from evaluate.regulatory import map_to_regulations, regulatory_report

    reg_df = regulatory_report(summary_df, review_df=review_df)
"""

from __future__ import annotations

import textwrap
from typing import Optional

import numpy as np
import pandas as pd


# ── Per-level baseline framework mappings ─────────────────────────────────────
# These capture the structural relationship between attack class and standard.
# The dynamic mapper then enriches each entry with actual metric values.

_LEVEL_MAP: dict[str, dict[str, str]] = {
    "character": {
        "nist":  "Information Security (§2.6)",
        "mitre": "AML.T0043 · AML.T0015",
        "owasp": "LLM09 – Misinformation",
        "eu":    "Art. 15 §4 – Robustness under adversarial inputs",
    },
    "word": {
        "nist":  "Information Security (§2.6) · Information Integrity (§2.5)",
        "mitre": "AML.T0043 · AML.T0015 · AML.T0040",
        "owasp": "LLM09 – Misinformation",
        "eu":    "Art. 15 §4 – Robustness; Art. 13 – Transparency",
    },
    "sentence": {
        "nist":  "Information Integrity (§2.5) · Confabulation (§2.3)",
        "mitre": "AML.T0043",
        "owasp": "LLM09 – Misinformation",
        "eu":    "Art. 13 – Transparency · Art. 15 §4 – Robustness",
    },
    "semantic": {
        "nist":  "Information Integrity (§2.5) · Information Security (§2.6)",
        "mitre": "AML.T0043 · AML.T0015",
        "owasp": "LLM09 – Misinformation",
        "eu":    "Art. 15 §4 – Robustness",
    },
    "structural": {
        "nist":  "Information Integrity (§2.5) · Information Security (§2.6)",
        "mitre": "AML.T0043 · AML.T0015 · AML.T0016",
        "owasp": "LLM01 – Prompt Injection · LLM09 – Misinformation",
        "eu":    "Art. 15 §4 – Robustness; Art. 63 – Documentation & traceability",
    },
}

# ── Specific overrides for named attacks ──────────────────────────────────────
_ATTACK_NIST_OVERRIDE: dict[str, str] = {
    "NegationInjection": (
        "Information Integrity (§2.5) — logical negation flips classification "
        "without changing surface sentiment; model relies on token patterns "
        "rather than semantic understanding; Confabulation risk in summarisation pipelines"
    ),
    "Homoglyph": (
        "Information Security (§2.6) — Unicode homoglyph substitution bypasses "
        "ASCII-only content filters and keyword blocklists; imperceptible to human reviewers"
    ),
    "BackTranslation": (
        "Information Integrity (§2.5) — MT round-trip as data normalisation signal: "
        "back-translated text is *more* fluent than noisy pre-tokenised source; "
        "negative acc_drop indicates data quality issue in evaluation set, not model robustness"
    ),
    "BERTAttack": (
        "Information Security (§2.6) · Information Integrity (§2.5) — "
        "contextual BERT substitution produces highest-stealth word-level attack; "
        "non-determinism in model outputs (text_changed=False flips) implicates "
        "Confabulation (§2.3) and reproducibility obligations"
    ),
    "StressTest": (
        "Information Integrity (§2.5) — tautology append probes attention drift; "
        "Confabulation (§2.3) risk when model redistributes attention to vacuous suffix"
    ),
    "SemanticAttack": (
        "Information Integrity (§2.5) — meaning-preserving paraphrase that flips "
        "decision boundary; reveals brittle surface-token reliance over semantic reasoning"
    ),
}

_ATTACK_MITRE_OVERRIDE: dict[str, str] = {
    "Homoglyph":       "AML.T0043 – Craft Adversarial Data (Unicode manipulation)",
    "NegationInjection": "AML.T0043 · AML.T0015 · AML.T0016 – Craft, Evade, Verify",
    "BackTranslation": "AML.T0043 – Craft Adversarial Data (MT paraphrase pipeline)",
    "BERTAttack":      "AML.T0043 · AML.T0015 · AML.T0040 – Craft, Evade, API Access",
}

_ATTACK_EU_OVERRIDE: dict[str, str] = {
    "BackTranslation": (
        "Art. 10 §3 – Data governance: MT normalisation reveals pre-tokenisation "
        "noise in the training/evaluation corpus; document as data quality finding"
    ),
    "BERTAttack": (
        "Art. 13 – Transparency · Art. 17 – Quality management: "
        "non-deterministic outputs on identical inputs must be documented "
        "and managed in high-risk AI systems"
    ),
    "NegationInjection": (
        "Art. 15 §4 – High-risk AI must remain accurate and robust to adversarial inputs; "
        "12% accuracy drop at high stealth triggers mandatory risk documentation"
    ),
}


# ── Helper: recommended action per finding ────────────────────────────────────

def _action(attack: str, level: str, acc_drop: float, stealth: float, severity: str) -> str:
    """Generate a concrete recommended action for a finding."""
    if severity == "HIGH":
        if "Negation" in attack:
            return (
                f"Priority: add negation-aware pre-processing (detect negation scope before "
                f"classification). Include negation robustness in regression test suite. "
                f"Consider fine-tuning on negated examples or using an ensemble that "
                f"explicitly models logical operators. ({acc_drop:.0%} drop at {stealth:.3f} stealth)"
            )
        if level == "structural":
            return (
                f"Priority: implement Unicode normalisation (NFC/NFKC) and homoglyph "
                f"detection at the API gateway layer. Add structural perturbation test cases "
                f"to CI regression suite. ({acc_drop:.0%} drop at {stealth:.3f} stealth)"
            )
        if level in ("word", "semantic"):
            return (
                f"Priority: adversarial training on synonym-substituted examples; "
                f"consider semantic-similarity guard at inference (flag inputs with "
                f"cosine sim < 0.85 to known safe examples). "
                f"({acc_drop:.0%} drop at {stealth:.3f} stealth)"
            )
        return (
            f"Priority: add {attack} test cases to regression suite; "
            f"investigate model brittleness at this perturbation level. "
            f"({acc_drop:.0%} drop at {stealth:.3f} stealth)"
        )
    if severity == "MEDIUM":
        return (
            f"Retest at n≥200 to confirm — current n=50 has ±2% swing per flip. "
            f"Add to monitoring dashboard. Consider targeted adversarial data augmentation "
            f"for {level}-level inputs. ({acc_drop:.0%} drop)"
        )
    if severity == "INFO":
        return (
            "Document as data quality finding. Consider MT-based normalisation "
            "as a pre-processing step. Confirm on full dataset (n=872) before "
            "drawing conclusions."
        )
    return "Monitor at larger sample size. Low immediate risk."


# ── Main mapping function ──────────────────────────────────────────────────────

def map_to_regulations(
    summary_df: pd.DataFrame,
    review_df: Optional[pd.DataFrame] = None,
    risk_threshold: float = 0.03,
    min_acc_drop: float = 0.0,
) -> pd.DataFrame:
    """
    Map actual test findings to regulatory frameworks.

    Only findings that cross a significance threshold (risk_score ≥
    ``risk_threshold`` or acc_drop > ``min_acc_drop``) are reported.
    Special findings (data cleaning, model non-determinism) are always
    included when detected.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of ``compute_attack_summary()``.
    review_df : pd.DataFrame | None
        Output of ``display_human_review()`` — used to detect
        non-determinism (text_changed=False + flipped=True) cases.
    risk_threshold : float
        Minimum risk_score to report a finding as HIGH or MEDIUM
        (default 0.03).  Attacks below this with acc_drop > 0 are
        reported as LOW.
    min_acc_drop : float
        Minimum acc_drop (absolute) to include a LOW finding (default 0).

    Returns
    -------
    pd.DataFrame
        One row per finding.  Columns:
        finding, type, severity, attacks, level,
        acc_drop, asr, stealth, risk_score,
        nist_ai_600_1, mitre_atlas, owasp_llm, eu_ai_act,
        recommended_action
    """
    findings: list[dict] = []

    for _, row in summary_df.iterrows():
        attack  = row["attack"]
        level   = row.get("level", "unknown")
        acc_drop = float(row.get("acc_drop", 0) or 0)
        asr      = float(row.get("asr", 0) or 0)
        stealth  = row.get("stealth_score", float("nan"))
        risk     = row.get("risk_score",    float("nan"))

        lmap = _LEVEL_MAP.get(level, _LEVEL_MAP["structural"])
        nist = _ATTACK_NIST_OVERRIDE.get(attack, lmap["nist"])
        mitre = _ATTACK_MITRE_OVERRIDE.get(attack, lmap["mitre"])
        owasp = lmap["owasp"]
        eu   = _ATTACK_EU_OVERRIDE.get(attack, lmap["eu"])

        # ── Special case: data-cleaning effect (negative acc_drop) ──────────
        if acc_drop < -0.005:
            findings.append({
                "finding": (
                    f"{attack}: accuracy *improved* by {abs(acc_drop):.0%} "
                    f"(stealth {stealth:.3f} — MT output is cleaner than noisy source text)"
                ),
                "type":     "data_cleaning",
                "severity": "INFO",
                "attacks":  attack,
                "level":    level,
                "acc_drop": acc_drop,
                "asr":      asr,
                "stealth":  stealth,
                "risk_score": risk,
                "nist_ai_600_1":      nist,
                "mitre_atlas":        mitre,
                "owasp_llm":          owasp,
                "eu_ai_act":          eu,
                "recommended_action": _action(attack, level, acc_drop, stealth, "INFO"),
            })
            continue

        # ── Skip zero-impact, zero-risk attacks ──────────────────────────────
        if acc_drop <= min_acc_drop and (pd.isna(risk) or abs(float(risk)) < 0.001):
            continue

        # ── Determine severity from risk_score ───────────────────────────────
        risk_val = float(risk) if not pd.isna(risk) else 0.0
        if risk_val >= 0.07:
            severity = "HIGH"
        elif risk_val >= risk_threshold:
            severity = "MEDIUM"
        elif acc_drop > min_acc_drop:
            severity = "LOW"
        else:
            continue

        stealth_str = f"{stealth:.3f}" if not (isinstance(stealth, float) and np.isnan(stealth)) else "n/a"
        findings.append({
            "finding": (
                f"{attack}: {acc_drop:.0%} accuracy drop, "
                f"ASR {asr:.0%}, stealth {stealth_str}, "
                f"risk score {risk_val:.4f}"
            ),
            "type":     "impact",
            "severity": severity,
            "attacks":  attack,
            "level":    level,
            "acc_drop": acc_drop,
            "asr":      asr,
            "stealth":  stealth,
            "risk_score": risk,
            "nist_ai_600_1":      nist,
            "mitre_atlas":        mitre,
            "owasp_llm":          owasp,
            "eu_ai_act":          eu,
            "recommended_action": _action(attack, level, acc_drop,
                                          float(stealth) if not pd.isna(stealth) else 0.0,
                                          severity),
        })

    # ── Non-determinism finding from review_df ───────────────────────────────
    if review_df is not None and "text_changed" in review_df.columns:
        nd = review_df[
            (review_df["review_priority"] == "MEDIUM") & (~review_df["text_changed"])
        ]
        if len(nd):
            affected = ", ".join(nd["attack"].unique())
            findings.append({
                "finding": (
                    f"Model non-determinism: {len(nd)} case(s) — identical input "
                    f"produced different outputs across two calls ({affected})"
                ),
                "type":     "non_determinism",
                "severity": "MEDIUM",
                "attacks":  affected,
                "level":    "—",
                "acc_drop": float("nan"),
                "asr":      float("nan"),
                "stealth":  float("nan"),
                "risk_score": float("nan"),
                "nist_ai_600_1": (
                    "Confabulation (§2.3) — non-deterministic outputs on identical inputs "
                    "undermine auditability and reproducibility of GenAI systems; "
                    "Information Integrity (§2.5) — output consistency is a measurable "
                    "integrity property under AI 600-1 MEASURE actions"
                ),
                "mitre_atlas": "AML.T0040 – ML Model Inference API Access (probabilistic sampling side-effect)",
                "owasp_llm":   "LLM09 – Misinformation (inconsistent outputs may mislead users across sessions)",
                "eu_ai_act": (
                    "Art. 13 – Transparency: non-determinism range must be disclosed; "
                    "Art. 17 – Quality management: reproducibility requirements for high-risk AI; "
                    "Art. 9 – Risk management: decision-boundary instability is a documented risk"
                ),
                "recommended_action": (
                    "Set temperature=0 / use fixed seed for reproducible evaluation runs. "
                    "Document non-determinism range (observed flip variance) in model card. "
                    "For high-stakes decisions, use ensemble voting or majority-vote across k calls. "
                    "Flag in NIST AI 600-1 MEASURE 2.5 audit log."
                ),
            })

    if not findings:
        return pd.DataFrame(columns=[
            "finding", "type", "severity", "attacks", "level",
            "acc_drop", "asr", "stealth", "risk_score",
            "nist_ai_600_1", "mitre_atlas", "owasp_llm", "eu_ai_act",
            "recommended_action",
        ])

    df = pd.DataFrame(findings)
    _sev_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    df["_rank"] = df["severity"].map(_sev_rank).fillna(9)
    df = df.sort_values(["_rank", "risk_score"], ascending=[True, False], na_position="last")
    return df.drop(columns=["_rank"]).reset_index(drop=True)


# ── Console report ─────────────────────────────────────────────────────────────

def regulatory_report(
    summary_df: pd.DataFrame,
    review_df: Optional[pd.DataFrame] = None,
    risk_threshold: float = 0.03,
    min_acc_drop: float = 0.0,
    width: int = 72,
) -> pd.DataFrame:
    """
    Print a formatted regulatory impact report and return the findings DataFrame.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of ``compute_attack_summary()``.
    review_df : pd.DataFrame | None
        Output of ``display_human_review()`` — used for non-determinism detection.
    risk_threshold : float
        Minimum risk_score for MEDIUM/HIGH classification (default 0.03).
    min_acc_drop : float
        Minimum acc_drop to include LOW findings (default 0).
    width : int
        Console print width (default 72).

    Returns
    -------
    pd.DataFrame
        The findings DataFrame (same as ``map_to_regulations()``).
    """
    df = map_to_regulations(
        summary_df, review_df=review_df,
        risk_threshold=risk_threshold, min_acc_drop=min_acc_drop,
    )

    _header = "Regulatory Impact Assessment — Findings from This Run"
    _sub    = "Frameworks: NIST AI 600-1 · MITRE ATLAS · OWASP LLM Top 10 · EU AI Act"
    print(f"\n{'=' * width}")
    print(f"  {_header}")
    print(f"  {_sub}")
    print(f"{'=' * width}")

    if df.empty:
        print("  No findings above threshold — all attacks within acceptable risk bounds.")
        print(f"{'=' * width}\n")
        return df

    _icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡", "INFO": "ℹ️ "}
    for sev in ["HIGH", "MEDIUM", "INFO", "LOW"]:
        grp = df[df["severity"] == sev]
        if grp.empty:
            continue
        print(f"\n{_icon.get(sev, '  ')} {sev}  ({len(grp)} finding{'s' if len(grp) > 1 else ''})")
        print("─" * width)
        for _, r in grp.iterrows():
            _wrap = lambda s: "\n".join(
                "              " + line
                for line in textwrap.wrap(str(s), width - 16)
            )
            print(f"  Finding     : {r['finding']}")
            print(f"  NIST 600-1  : {textwrap.fill(r['nist_ai_600_1'],  width - 16, subsequent_indent=' ' * 16)}")
            print(f"  MITRE ATLAS : {r['mitre_atlas']}")
            print(f"  OWASP LLM   : {r['owasp_llm']}")
            print(f"  EU AI Act   : {textwrap.fill(r['eu_ai_act'],      width - 16, subsequent_indent=' ' * 16)}")
            print(f"  Action      : {textwrap.fill(r['recommended_action'], width - 16, subsequent_indent=' ' * 16)}")
            print()

    counts = df["severity"].value_counts()
    print(f"{'─' * width}")
    print(
        f"  Summary: "
        + "  ".join(
            f"{_icon.get(s, s)} {s}: {counts.get(s, 0)}"
            for s in ["HIGH", "MEDIUM", "LOW", "INFO"]
            if counts.get(s, 0) > 0
        )
    )
    print(f"{'=' * width}\n")
    return df


# ── Short-tag extractors for the heatmap ─────────────────────────────────────

def _nist_tags(text: str) -> str:
    """Extract short NIST AI 600-1 category names from a full citation string."""
    import re
    tags = []
    if re.search(r"Info(?:rmation)? Integrity|§2\.5", text, re.I):
        tags.append("Info Integrity §2.5")
    if re.search(r"Info(?:rmation)? Security|§2\.6", text, re.I):
        tags.append("Info Security §2.6")
    if re.search(r"Confabulation|§2\.3", text, re.I):
        tags.append("Confabulation §2.3")
    if re.search(r"Data (?:Privacy|Provenance)|§2\.1", text, re.I):
        tags.append("Data Privacy §2.1")
    if re.search(r"Harmful Bias|§2\.2", text, re.I):
        tags.append("Harmful Bias §2.2")
    if re.search(r"data (?:governance|quality)|Art\. 10", text, re.I):
        tags.append("Data Governance")
    return "\n".join(tags) if tags else text[:40]


def _mitre_tags(text: str) -> str:
    """Extract MITRE ATT&CK IDs from a full citation string."""
    import re
    ids = re.findall(r"AML\.T\d{4}", text)
    seen: list[str] = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    return "  ".join(seen) if seen else text[:30]


def _owasp_tags(text: str) -> str:
    """Extract OWASP LLM Top 10 IDs from a full citation string."""
    import re
    ids = re.findall(r"LLM\d{2}", text)
    seen: list[str] = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    return "  ".join(seen) if seen else text[:20]


def _eu_tags(text: str) -> str:
    """Extract EU AI Act article references from a full citation string."""
    import re
    arts = re.findall(r"Art\.\s*\d+(?:\s*§\d+)?", text)
    seen: list[str] = []
    for a in arts:
        a = a.replace("  ", " ")
        if a not in seen:
            seen.append(a)
    return "  ".join(seen) if seen else text[:30]


# ── Regulatory heatmap ────────────────────────────────────────────────────────

def render_regulatory_heatmap(
    reg_df: pd.DataFrame,
    figsize: tuple = (14, None),
    save_path: str | None = None,
    dpi: int = 150,
):
    """
    Render a heatmap showing which regulatory frameworks each finding implicates.

    Rows = findings; columns = NIST AI 600-1 / MITRE ATLAS /
    OWASP LLM Top 10 / EU AI Act.  Cell colour = severity (HIGH=red,
    MEDIUM=orange, LOW=yellow, INFO=blue).  Cells show short regulatory
    tags (§ references, ATT&CK IDs, article numbers) rather than truncated
    prose, making the heatmap scannable at a glance.

    Parameters
    ----------
    reg_df : pd.DataFrame
        Output of ``map_to_regulations()``.
    figsize : tuple
        ``(width, height)`` — height auto-computed from row count if None.
    save_path : str | None
        File path to save the figure, or None to skip.
    dpi : int
        Output resolution (default 150).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if reg_df.empty:
        print("No findings to plot.")
        return None

    _sev_color = {
        "HIGH":   "#D32F2F",
        "MEDIUM": "#F57C00",
        "LOW":    "#FBC02D",
        "INFO":   "#1976D2",
        "—":      "#E0E0E0",
    }

    frameworks  = ["nist_ai_600_1", "mitre_atlas", "owasp_llm", "eu_ai_act"]
    col_headers = ["NIST AI 600-1", "MITRE ATLAS", "OWASP LLM\nTop 10", "EU AI Act"]
    # Short-tag extractor per column
    _tagfn = [_nist_tags, _mitre_tags, _owasp_tags, _eu_tags]

    # Build row labels — attack + metric snapshot
    rows = []
    for _, r in reg_df.iterrows():
        atk  = r.get("attacks", "?")
        sev  = r.get("severity", "—")
        drop = r.get("acc_drop", float("nan"))
        asr  = r.get("asr",      float("nan"))
        if isinstance(drop, float) and np.isnan(drop):
            metric = ""
        else:
            metric = f"  Δacc {drop:+.0%}"
            if not (isinstance(asr, float) and np.isnan(asr)) and asr > 0:
                metric += f"  ASR {asr:.0%}"
        rows.append(f"{atk}{metric}  [{sev}]")

    n_rows  = len(rows)
    row_h   = 0.90           # cell height in data units
    col_w   = 1.0
    height  = figsize[1] if figsize[1] else max(4, n_rows * row_h * 1.2 + 2.0)
    fig, ax = plt.subplots(figsize=(figsize[0], height))

    for col_i, (col, tagfn) in enumerate(zip(frameworks, _tagfn)):
        for row_i, (_, r) in enumerate(reg_df.iterrows()):
            sev  = r.get("severity", "—")
            text = str(r.get(col, ""))
            has_content = bool(text.strip() and text != "—" and text != "nan")
            cell_color  = _sev_color.get(sev, "#E0E0E0") if has_content else "#F5F5F5"
            text_color  = "white" if sev in ("HIGH", "MEDIUM") else "#222222"

            y0 = n_rows - row_i - row_h
            rect = plt.Rectangle(
                (col_i * col_w + 0.05, y0),
                col_w - 0.10, row_h - 0.05,
                linewidth=0, edgecolor="none",
                facecolor=cell_color, alpha=0.90, zorder=2,
            )
            ax.add_patch(rect)

            # Extract short tags for this cell
            tag_text = tagfn(text) if has_content else ""
            if tag_text:
                ax.text(
                    col_i * col_w + col_w / 2,
                    y0 + (row_h - 0.05) / 2,
                    tag_text,
                    ha="center", va="center",
                    fontsize=7.5, fontweight="semibold",
                    color=text_color, zorder=3,
                    linespacing=1.4,
                )

    # Grid lines between rows
    for i in range(n_rows + 1):
        ax.axhline(n_rows - i, color="#CCCCCC", linewidth=0.4, zorder=1)
    for j in range(len(frameworks) + 1):
        ax.axvline(j * col_w, color="#CCCCCC", linewidth=0.4, zorder=1)

    # Axes formatting
    ax.set_xlim(0, len(frameworks) * col_w)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([i * col_w + col_w / 2 for i in range(len(frameworks))])
    ax.set_xticklabels(col_headers, fontsize=10, fontweight="bold", linespacing=1.3)
    ax.set_yticks([n_rows - i - row_h / 2 for i in range(n_rows)])
    ax.set_yticklabels(rows, fontsize=8.5)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False,
                   length=0, pad=6)
    ax.tick_params(axis="y", left=True, right=False, length=0, pad=6)
    ax.spines[:].set_visible(False)
    ax.set_title(
        "Regulatory Impact Heatmap — Findings from This Run",
        fontsize=12, fontweight="bold", pad=18,
    )

    legend_patches = [
        mpatches.Patch(color=_sev_color[s], alpha=0.85, label=s)
        for s in ["HIGH", "MEDIUM", "LOW", "INFO"]
        if s in reg_df["severity"].values
    ]
    ax.legend(
        handles=legend_patches, title="Severity", loc="lower right",
        fontsize=8, title_fontsize=9, framealpha=0.92,
        bbox_to_anchor=(1.0, -0.02),
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✅ Regulatory heatmap saved → {save_path}")

    return fig
