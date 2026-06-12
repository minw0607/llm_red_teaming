"""
evaluate/nli_executive.py — Executive report for NLI robustness runs.

Same pattern as ``evaluate.injection_executive``: deterministic metrics
(clean vs adversarial accuracy, robustness gap, ANLI round curve) + a judge-LLM
narrative, rendered as a business-level HTML report. The prompt is aggregate-only
(no premises/hypotheses/responses), and a labelled fallback is used if the LLM is
unavailable. Every report carries an 'illustrative sample' disclaimer.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from .nli_metrics import nli_summary, robustness_gap, anli_round_curve, nli_accuracy
from .executive import _call_llm, _RISK_COLORS, _SEV_BADGE


def compute_nli_metrics(results) -> dict:
    """Deterministic figures for the report."""
    summ = nli_summary(results)
    gap = robustness_gap(results)
    curve = anli_round_curve(results)

    clean_row = summ[summ["kind"] == "clean"]
    clean_acc = float(clean_row["accuracy"].iloc[0]) if not clean_row.empty else None
    adv = summ[summ["kind"] == "adversarial"]
    adv_acc = round(float(adv["accuracy"].mean()), 4) if not adv.empty else None
    worst_gap = float(gap["robustness_gap"].max()) if not gap.empty else 0.0

    return {
        "total": len(results),
        "overall_acc": nli_accuracy(results),
        "clean_acc": clean_acc,
        "adv_acc": adv_acc,
        "worst_gap": round(worst_gap, 4),
        "n_datasets": int(summ.shape[0]),
        "by_source": summ.to_dict("records"),
        "gaps": gap.to_dict("records"),
        "anli_curve": curve.to_dict("records"),
    }


_SYSTEM_PROMPT = (
    "You are a senior AI assurance consultant writing the executive summary of a "
    "model-robustness assessment for a non-technical leadership audience. You are "
    "given ONLY aggregate statistics from a Natural Language Inference (NLI) "
    "robustness test. Write a clear, factual, business-oriented interpretation. "
    "Do not invent numbers."
)


def _build_prompt(m: dict, cfg: dict) -> tuple[str, str]:
    lines = [
        f"Target model: {cfg.get('model_name', 'the model')}",
        f"Total NLI items scored: {m['total']}",
        f"Clean accuracy (MultiNLI): {_pct(m['clean_acc'])}",
        f"Adversarial accuracy (mean over ANLI/AdvGLUE): {_pct(m['adv_acc'])}",
        f"Worst robustness gap (clean − adversarial): {_pct(m['worst_gap'])}",
        "",
        "Per-dataset accuracy:",
    ]
    for s in m["by_source"]:
        lines.append(f"  - {s['source']} ({s['kind']}): n={s['n']}, acc={s['accuracy']:.2%}")
    if m["anli_curve"]:
        lines.append("")
        lines.append("ANLI difficulty curve (R1 easiest → R3 hardest):")
        for r in m["anli_curve"]:
            lines.append(f"  - {r['round']}: n={r['n']}, acc={r['accuracy']:.2%}")
    stats = "\n".join(lines)

    user = f"""\
Aggregate results of an NLI (Natural Language Inference) robustness evaluation:

{stats}

Context for interpretation:
- NLI = deciding whether a hypothesis is entailed by, neutral to, or contradicts
  a premise. It probes multi-sentence logical reasoning.
- "Clean" = MultiNLI (ordinary examples). "Adversarial" = ANLI (human-written to
  fool strong models) and AdvGLUE (adversarially-perturbed). HIGHER accuracy is
  better.
- "Robustness gap" = clean accuracy minus adversarial accuracy. A LARGE gap means
  the model reasons well on ordinary text but is brittle under adversarial
  pressure — a reliability/assurance concern, not a security breach.
- A declining ANLI round curve (R1 > R2 > R3) is expected; how steeply it falls
  indicates how quickly reasoning degrades under targeted difficulty.

Write the executive summary as JSON with EXACTLY these keys:
{{
  "overall_risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "overall_verdict": "<2-3 sentence plain-English robustness verdict for leadership>",
  "key_findings": [{{"title": "<headline>", "detail": "<2-3 sentences>", "severity": "LOW|MEDIUM|HIGH|INFO"}}],
  "dataset_analysis": [{{"dataset": "<name>", "plain_english": "<what it tests and how the model held up>"}}],
  "regulatory_implications": "<2-3 sentences citing NIST AI 600-1 (Information Integrity, Confabulation), MITRE ATLAS AML.T0043, OWASP LLM09, EU AI Act Art. 15>",
  "judge_caveats": "<1-2 sentences on measurement reliability and human review>",
  "recommendations": [{{"priority": 1, "action": "<imperative>", "rationale": "<why>"}}]
}}

Reply with ONLY the JSON object."""
    return _SYSTEM_PROMPT, user


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.2%}"


def _fallback_dict(m: dict, cfg: dict) -> dict:
    gap = m["worst_gap"]
    rl = "LOW" if gap < 0.10 else "MEDIUM" if gap < 0.25 else "HIGH"
    return {
        "overall_risk_level": rl,
        "overall_verdict": (
            f"On {m['total']} NLI items the model scored {_pct(m['clean_acc'])} on clean "
            f"data and {_pct(m['adv_acc'])} on adversarial data — a worst-case robustness "
            f"gap of {_pct(gap)}. [LLM interpretation unavailable — fallback template used.]"
        ),
        "key_findings": [{
            "title": "Reasoning robustness" if gap < 0.10 else "Adversarial brittleness",
            "detail": (f"Clean accuracy {_pct(m['clean_acc'])} vs adversarial {_pct(m['adv_acc'])} "
                       f"(worst gap {_pct(gap)})."),
            "severity": "INFO" if gap < 0.10 else "MEDIUM" if gap < 0.25 else "HIGH",
        }],
        "dataset_analysis": [
            {"dataset": s["source"], "plain_english": f"Accuracy {s['accuracy']:.2%} (n={s['n']})."}
            for s in m["by_source"]
        ],
        "regulatory_implications": (
            "Maps to NIST AI 600-1 §2.5 (Information Integrity) and §2.2 (Confabulation), "
            "MITRE ATLAS AML.T0043, OWASP LLM09 (Misinformation), and EU AI Act Art. 15 "
            "(accuracy & robustness). [LLM interpretation unavailable.]"
        ),
        "judge_caveats": (
            "Scoring is deterministic (labels parsed from the reply). Adversarial benchmarks "
            "are deliberately hard; low scores reflect benchmark difficulty, not necessarily "
            "production failure rates."
        ),
        "recommendations": [
            {"priority": 1, "action": "Add reasoning-robustness checks to the evaluation suite",
             "rationale": "Clean accuracy alone overstates reliability under adversarial input."},
        ],
    }


# ── HTML ─────────────────────────────────────────────────────────────────────────

def render_nli_html(data: dict, metrics: dict, config: dict | None = None) -> str:
    cfg = config or {}
    run_date = cfg.get("run_date", str(date.today()))
    model = cfg.get("model_name", "GPT (Azure)")
    rl = data.get("overall_risk_level", "LOW")
    rl_fg, rl_bg = _RISK_COLORS.get(rl, ("#333", "#f5f5f5"))

    def _section(title, content, icon=""):
        return (f'<div style="margin:18px 0 10px;"><div style="font-size:13px;font-weight:700;'
                f'color:#37474F;text-transform:uppercase;letter-spacing:.8px;border-bottom:2px solid #ECEFF1;'
                f'padding-bottom:5px;margin-bottom:10px;">{icon} {title}</div>{content}</div>')

    def _stat(value, label, color="#263238"):
        return (f'<div style="text-align:center;padding:10px 16px;border-right:1px solid #ECEFF1;">'
                f'<div style="font-size:26px;font-weight:800;color:{color}">{value}</div>'
                f'<div style="font-size:11px;color:#607D8B;margin-top:2px;text-transform:uppercase;'
                f'letter-spacing:.5px">{label}</div></div>')

    gap = metrics["worst_gap"]
    gap_col = "#2E7D32" if gap < 0.10 else "#EF6C00" if gap < 0.25 else "#C62828"

    parts = [
        '<div style="font-family:\'Segoe UI\',Arial,sans-serif;max-width:900px;margin:0 auto;'
        'border:1px solid #CFD8DC;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.1);">',
        '<div style="background:#263238;color:white;padding:20px 28px 16px;">',
        '<div style="font-size:11px;letter-spacing:1.5px;text-transform:uppercase;opacity:.7;'
        'margin-bottom:4px;">Confidential — Internal Assurance Assessment</div>',
        '<div style="font-size:22px;font-weight:700;">🧩 NLI Robustness Assessment — Executive Summary</div>',
        f'<div style="margin-top:8px;font-size:12px;opacity:.75;">Target: <strong>{model}</strong> '
        f'&nbsp;|&nbsp; Items: <strong>{metrics["total"]}</strong> &nbsp;|&nbsp; '
        f'Date: <strong>{run_date}</strong></div>', '</div>',
        '<div style="background:#FFF8E1;border-bottom:1px solid #FFE082;padding:9px 28px;'
        'font-size:11.5px;color:#795548;line-height:1.5;">⚠️ <strong>Illustrative sample.</strong> '
        'Accuracy figures come from an automated harness scoring fixed public benchmarks (MultiNLI, '
        'ANLI, AdvGLUE). Adversarial sets are deliberately hard — low scores reflect benchmark '
        'difficulty, not a production failure rate. This demonstrates the reporting format; it is not '
        'an authoritative robustness verdict of any model.</div>',
        f'<div style="background:{rl_bg};border-left:6px solid {rl_fg};padding:14px 24px;display:flex;'
        'align-items:center;gap:14px;">',
        f'<div style="background:{rl_fg};color:white;font-size:13px;font-weight:700;padding:6px 16px;'
        f'border-radius:4px;white-space:nowrap;letter-spacing:.5px;">ROBUSTNESS RISK: {rl}</div>',
        f'<div style="font-size:13.5px;color:#333;line-height:1.6;">{data.get("overall_verdict", "")}</div>',
        '</div>', '<div style="padding:20px 28px;">',
    ]

    stats = ('<div style="display:flex;background:#FAFAFA;border:1px solid #ECEFF1;border-radius:6px;'
             'overflow:hidden;margin-bottom:6px;">'
             + _stat(_pct(metrics["clean_acc"]), "Clean Acc", "#2E7D32")
             + _stat(_pct(metrics["adv_acc"]), "Adversarial Acc",
                     "#2E7D32" if (metrics["adv_acc"] or 0) >= 0.6 else "#EF6C00")
             + _stat(f"{gap:+.1%}", "Worst Gap", gap_col)
             + _stat(str(metrics["n_datasets"]), "Datasets") + '</div>')
    parts.append(_section("Testing Scope", stats, "🎯"))

    # per-source table
    rows = ""
    for s in metrics["by_source"]:
        col = "#2E7D32" if s["accuracy"] >= 0.6 else "#EF6C00" if s["accuracy"] >= 0.4 else "#C62828"
        kind = "clean" if s["kind"] == "clean" else "adversarial"
        rows += (f'<tr style="border-bottom:1px solid #ECEFF1;"><td style="padding:7px 12px;font-weight:600;">'
                 f'{s["source"]}</td><td style="padding:7px 12px;text-align:center;color:#607D8B;">{kind}</td>'
                 f'<td style="padding:7px 12px;text-align:center;">{s["n"]}</td>'
                 f'<td style="padding:7px 12px;text-align:center;font-weight:700;color:{col};">'
                 f'{s["accuracy"]:.2%}</td></tr>')
    parts.append(_section("Accuracy by Dataset",
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        '<thead><tr style="background:#37474F;color:white;"><th style="padding:7px 12px;text-align:left;">Dataset</th>'
        '<th style="padding:7px 12px;">Kind</th><th style="padding:7px 12px;">N</th>'
        f'<th style="padding:7px 12px;">Accuracy</th></tr></thead><tbody>{rows}</tbody></table>', "📊"))

    # ANLI curve
    if metrics["anli_curve"]:
        crows = ""
        for r in metrics["anli_curve"]:
            crows += (f'<tr style="border-bottom:1px solid #ECEFF1;"><td style="padding:5px 12px;">{r["round"]}</td>'
                      f'<td style="padding:5px 12px;text-align:center;">{r["n"]}</td>'
                      f'<td style="padding:5px 12px;text-align:center;font-weight:600;">{r["accuracy"]:.2%}</td></tr>')
        parts.append(_section("ANLI Difficulty Curve (R1 easiest → R3 hardest)",
            '<table style="width:100%;border-collapse:collapse;font-size:12.5px;">'
            '<thead><tr style="background:#ECEFF1;"><th style="padding:5px 12px;text-align:left;">Round</th>'
            '<th style="padding:5px 12px;">N</th><th style="padding:5px 12px;">Accuracy</th></tr></thead>'
            f'<tbody>{crows}</tbody></table>', "📉"))

    # findings
    fh = '<div style="display:grid;gap:8px;">'
    for f in data.get("key_findings", []):
        sev = f.get("severity", "INFO")
        fg, bg = _RISK_COLORS.get(sev, ("#333", "#f9f9f9"))
        fh += (f'<div style="background:{bg};border-left:4px solid {fg};border-radius:0 6px 6px 0;padding:10px 14px;">'
               f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
               f'<span style="display:inline-block;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700;'
               f'{_SEV_BADGE.get(sev, "background:#888;color:white;")}">{sev}</span>'
               f'<strong style="font-size:13.5px;color:#263238;">{f.get("title", "")}</strong></div>'
               f'<div style="font-size:12.5px;color:#455A64;line-height:1.55;">{f.get("detail", "")}</div></div>')
    fh += '</div>'
    parts.append(_section("Key Findings", fh, "🔑"))

    if data.get("dataset_analysis"):
        sa = '<div style="display:grid;gap:6px;">'
        for s in data["dataset_analysis"]:
            sa += (f'<div style="font-size:12.5px;color:#455A64;line-height:1.5;"><strong style="color:#263238;">'
                   f'{s.get("dataset", "")}:</strong> {s.get("plain_english", "")}</div>')
        sa += '</div>'
        parts.append(_section("Dataset Analysis", sa, "🧪"))

    if data.get("regulatory_implications"):
        parts.append(_section("Regulatory Implications",
            f'<div style="font-size:12.5px;color:#455A64;line-height:1.6;">{data["regulatory_implications"]}</div>', "⚖️"))
    if data.get("judge_caveats"):
        parts.append(_section("Methodology Caveat",
            f'<div style="font-size:12.5px;color:#6A1B9A;line-height:1.6;background:#F3E5F5;border-radius:6px;'
            f'padding:10px 14px;">⚠️ {data["judge_caveats"]}</div>', "🔬"))
    if data.get("recommendations"):
        rl_ = '<ol style="margin:0;padding-left:20px;">'
        for r in data["recommendations"]:
            rl_ += (f'<li style="font-size:12.5px;color:#455A64;line-height:1.55;margin-bottom:6px;">'
                    f'<strong style="color:#263238;">{r.get("action", "")}</strong> — {r.get("rationale", "")}</li>')
        rl_ += '</ol>'
        parts.append(_section("Prioritised Recommendations", rl_, "✅"))

    parts.append('<div style="margin-top:18px;padding-top:12px;border-top:1px solid #ECEFF1;font-size:11px;'
                 'color:#90A4AE;">Metrics computed deterministically from results; narrative interpreted by a '
                 'judge LLM. Numbers are not LLM-generated.</div></div></div>')
    return "".join(parts)


def generate_nli_summary(results, target: Any = None, config: dict | None = None) -> tuple[str, dict]:
    """Generate the NLI-robustness executive report (HTML) + narrative dict."""
    cfg = dict(config or {})
    cfg.setdefault("run_date", str(date.today()))
    metrics = compute_nli_metrics(results)

    data = None
    if target is not None:
        system_prompt, user_prompt = _build_prompt(metrics, cfg)
        print(f"🤖 Calling judge LLM ({getattr(target, 'model', '?')}) for executive interpretation…")
        try:
            data = _call_llm(target, system_prompt, user_prompt)
            print("✅ LLM response received and parsed.")
        except Exception as e:
            print(f"⚠️  LLM call failed ({type(e).__name__}) — using fallback template.")
            data = None
    if not data or "overall_risk_level" not in data:
        data = _fallback_dict(metrics, cfg)

    html = render_nli_html(data, metrics, cfg)
    return html, data
