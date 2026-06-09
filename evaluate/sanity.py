"""
evaluate/sanity.py — Pre-flight readiness validator for the adversarial eval pipeline.

Runs structured checks across every component (dataset, target model, encoder,
and each attack) before the expensive full evaluation, then renders a clear
GO / HOLD HTML dashboard so users know whether it is safe to proceed.

Usage (notebook)
----------------
    from evaluate.sanity import sanity_check

    checks, atk_rows = sanity_check(dev_df, target, encoder, attacks)

    # checks  : dict[label, {"status": "ok"|"warn"|"fail", "detail": str}]
    # atk_rows: list[(attack_name, [(orig, attacked, changed), ...])]

The function automatically calls IPython's ``display(HTML(...))`` when running
inside a Jupyter kernel.  In plain Python it just returns the data structures.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass


# ── Colour constants ───────────────────────────────────────────────────────────

_ICONS: dict[str, tuple[str, str, str]] = {
    "ok":   ("✅", "#E8F5E9", "#2E7D32"),
    "warn": ("⚠️",  "#FFF8E1", "#F57F17"),
    "fail": ("❌", "#FFEBEE", "#B71C1C"),
}

_LEVEL_CLR: dict[str, str] = {
    "character":  "#E84393",
    "word":       "#FF7F0E",
    "sentence":   "#2CA02C",
    "semantic":   "#1F77B4",
    "structural": "#9467BD",
    "unknown":    "#78909C",
}

# Known sentences with ground-truth labels for the target-model probe
_PROBE_POS = "a wonderful film that is touching and funny"
_PROBE_NEG = "a terrible waste of time and money"


# ── Internal helpers ───────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Collapse whitespace and strip edges (mirrors adversarial_eval._norm)."""
    return re.sub(r"\s+", " ", s).strip()


def _highlight_diff(words_a: list[str], words_b: list[str], max_words: int = 18) -> str:
    """
    Return an HTML fragment highlighting words in *words_a* that are absent
    from *words_b* with a yellow background.  Truncated to *max_words*.
    """
    set_b = set(words_b)
    out: list[str] = []
    for w in words_a[:max_words]:
        if w not in set_b:
            out.append(f'<span style="background:#FFF9C4;font-weight:700;">{w}</span>')
        else:
            out.append(w)
    if len(words_a) > max_words:
        out.append('<span style="color:#9E9E9E">…</span>')
    return " ".join(out)


# ── Check runners ──────────────────────────────────────────────────────────────

def _check_dataset(dev_df: pd.DataFrame) -> dict:
    try:
        n_rows = len(dev_df)
        n_pos  = int((dev_df["label"] == 1).sum())
        n_neg  = int((dev_df["label"] == 0).sum())
        if "sentence" in dev_df.columns and "label" in dev_df.columns and n_rows > 0:
            return {
                "status": "ok",
                "detail": f"{n_rows:,} rows · {n_pos:,} positive · {n_neg:,} negative",
            }
        return {"status": "fail", "detail": "Missing 'sentence' or 'label' column"}
    except Exception as exc:
        return {"status": "fail", "detail": str(exc)[:140]}


def _check_target(target) -> dict:
    try:
        pred_pos = target.get_sentiment(_PROBE_POS)
        pred_neg = target.get_sentiment(_PROBE_NEG)
        ok = (pred_pos == "positive") and (pred_neg == "negative")
        if ok:
            return {
                "status": "ok",
                "detail": (
                    f"✓ '{_PROBE_POS[:45]}…' → {pred_pos}  |  "
                    f"✓ '{_PROBE_NEG[:35]}…' → {pred_neg}"
                ),
            }
        return {
            "status": "warn",
            "detail": (
                f"Unexpected responses: positive-sentence → {pred_pos}, "
                f"negative-sentence → {pred_neg}"
            ),
        }
    except Exception as exc:
        return {"status": "fail", "detail": str(exc)[:140]}


def _check_encoder(encoder) -> dict:
    if encoder is None:
        return {
            "status": "warn",
            "detail": "encoder=None — semantic_sim will not be computed; stealth scoring degraded",
        }
    try:
        from sentence_transformers import util as st_util

        e1 = encoder.encode("hello world", convert_to_tensor=True)
        e2 = encoder.encode("hi there",    convert_to_tensor=True)
        sim = round(st_util.cos_sim(e1, e2).item(), 3)
        return {
            "status": "ok",
            "detail": f"'hello world' ↔ 'hi there' = {sim}  (expected 0.6–0.85)",
        }
    except Exception as exc:
        return {"status": "fail", "detail": str(exc)[:140]}


def _check_attacks(
    attacks: dict,
    sample: list[str],
) -> tuple[dict[str, dict], list[tuple]]:
    """
    Test each attack on *sample* sentences.

    Returns
    -------
    checks   : dict[f"Attack · {name}", status_dict]
    atk_rows : list[(attack_name, [(orig, attacked, changed), ...])]
    """
    from evaluate.adversarial_eval import ATTACK_LEVELS  # local import avoids circular

    checks: dict[str, dict] = {}
    atk_rows: list[tuple] = []

    for name, atk in attacks.items():
        try:
            pairs: list[tuple[str, str, bool]] = []
            any_err = False

            for sent in sample:
                out = atk.attack(sent)
                if not out or not out.strip():
                    any_err = True
                    out = sent
                changed = _norm(out) != _norm(sent)
                pairs.append((sent, out, changed))

            bt_err = hasattr(atk, "_load_error") and atk._load_error

            if bt_err:
                checks[f"Attack · {name}"] = {
                    "status": "warn",
                    "detail": (
                        "MarianMT models not loaded (offline / cache miss) — "
                        f"will pass through text unchanged.  Error: {str(atk._load_error)[:80]}"
                    ),
                }
            elif any_err:
                checks[f"Attack · {name}"] = {
                    "status": "warn",
                    "detail": "One or more samples returned empty output",
                }
            else:
                n_changed = sum(1 for *_, c in pairs if c)
                checks[f"Attack · {name}"] = {
                    "status": "ok",
                    "detail": f"{n_changed}/{len(sample)} samples modified",
                }

            atk_rows.append((name, pairs))

        except Exception as exc:
            checks[f"Attack · {name}"] = {"status": "fail", "detail": str(exc)[:140]}
            atk_rows.append((name, []))

    return checks, atk_rows


# ── HTML renderer ──────────────────────────────────────────────────────────────

def render_sanity_html(
    checks: dict[str, dict],
    atk_rows: list[tuple],
) -> str:
    """
    Build and return a self-contained HTML string for the sanity-check dashboard.

    Parameters
    ----------
    checks   : mapping of component label → {"status", "detail"}
    atk_rows : list of (attack_name, [(orig, attacked, changed), ...])
    """
    from evaluate.adversarial_eval import ATTACK_LEVELS

    n_ok   = sum(1 for v in checks.values() if v["status"] == "ok")
    n_warn = sum(1 for v in checks.values() if v["status"] == "warn")
    n_fail = sum(1 for v in checks.values() if v["status"] == "fail")

    if n_fail:
        verdict = "❌  HOLD — Fix errors before proceeding"
        vfg, vbg = "#B71C1C", "#FFEBEE"
    elif n_warn:
        verdict = "⚠️  READY WITH WARNINGS — Review below then proceed with caution"
        vfg, vbg = "#E65100", "#FFF3E0"
    else:
        verdict = "✅  ALL CLEAR — Ready to run full evaluation (Step 4)"
        vfg, vbg = "#1B5E20", "#E8F5E9"

    # ── Header + verdict banner ────────────────────────────────────────────────
    html = f"""
<div style="font-family:'Segoe UI',Arial,sans-serif;max-width:820px;
  border:1px solid #CFD8DC;border-radius:8px;overflow:hidden;
  box-shadow:0 1px 4px rgba(0,0,0,.10);margin:8px 0;">

  <div style="background:#263238;color:white;padding:13px 20px 11px;">
    <span style="font-size:13px;font-weight:700;letter-spacing:.3px;">
      Step 3 · Sanity Check
    </span>
    <span style="font-size:11px;opacity:.65;margin-left:12px;">
      {n_ok} ok &nbsp;·&nbsp; {n_warn} warn &nbsp;·&nbsp; {n_fail} fail
    </span>
  </div>

  <div style="background:{vbg};border-left:5px solid {vfg};
    padding:11px 18px;font-size:14px;font-weight:700;color:{vfg};">
    {verdict}
  </div>

  <div style="padding:14px 18px 6px;">
    <table style="width:100%;border-collapse:collapse;font-size:12.5px;">
      <tr style="background:#F5F5F5;border-bottom:2px solid #E0E0E0;">
        <th style="padding:6px 10px;text-align:left;width:34%;">Component</th>
        <th style="padding:6px 10px;text-align:center;width:8%;">Status</th>
        <th style="padding:6px 10px;text-align:left;">Detail</th>
      </tr>
"""

    # ── Status rows ────────────────────────────────────────────────────────────
    for i, (label, v) in enumerate(checks.items()):
        icon, _bg, _fg = _ICONS[v["status"]]
        row_bg = "#FFFFFF" if i % 2 == 0 else "#FAFAFA"
        html += f"""
      <tr style="background:{row_bg};border-bottom:1px solid #EEEEEE;">
        <td style="padding:6px 10px;font-weight:600;color:#37474F;">{label}</td>
        <td style="padding:6px 10px;text-align:center;font-size:14px;">{icon}</td>
        <td style="padding:6px 10px;color:#546E7A;">{v['detail']}</td>
      </tr>"""

    html += """
    </table>
  </div>

  <div style="padding:6px 18px 14px;">
    <div style="font-size:12px;font-weight:700;color:#37474F;
      text-transform:uppercase;letter-spacing:.6px;
      border-bottom:2px solid #ECEFF1;padding-bottom:4px;margin-bottom:10px;">
      Sample Transformations
    </div>"""

    # ── Transformation rows per attack ─────────────────────────────────────────
    for name, pairs in atk_rows:
        level = ATTACK_LEVELS.get(name, "unknown")
        clr   = _LEVEL_CLR.get(level, "#78909C")
        html += f"""
    <div style="margin-bottom:10px;">
      <div style="font-size:11.5px;font-weight:700;color:{clr};
        text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">
        {name} <span style="opacity:.6;font-weight:400;">({level})</span>
      </div>"""

        for orig, attacked, changed in pairs:
            orig_w = orig.split()
            atk_w  = attacked.split()
            orig_hl = _highlight_diff(orig_w, atk_w)
            atk_hl  = _highlight_diff(atk_w, orig_w)
            marker  = "⚠️" if changed else "✅"
            border  = "#7CB342" if changed else "#BDBDBD"
            bg      = "#F1F8E9" if changed else "#FAFAFA"

            html += f"""
      <div style="background:{bg};border-left:3px solid {border};
        border-radius:0 4px 4px 0;padding:5px 10px;margin-bottom:3px;
        font-size:11.5px;font-family:monospace;">
        <span style="color:#9E9E9E;font-size:10px;">orig&nbsp;&nbsp;&nbsp;</span>
        {orig_hl}<br>
        <span style="color:#9E9E9E;font-size:10px;">attacked</span>
        {marker} {atk_hl}
      </div>"""

        html += "\n    </div>"

    html += "\n  </div>\n</div>"
    return html


# ── Public entry point ─────────────────────────────────────────────────────────

def sanity_check(
    dev_df: pd.DataFrame,
    target,
    encoder,
    attacks: dict,
    n_sample: int = 2,
    random_state: int = 7,
) -> tuple[dict, list]:
    """
    Run all pre-flight readiness checks and display an HTML dashboard.

    Parameters
    ----------
    dev_df       : evaluation dataset (columns: sentence, label)
    target       : model connector with ``get_sentiment(text) -> str``
    encoder      : SentenceTransformer instance (or None)
    attacks      : dict[name, attack_instance]
    n_sample     : number of random sentences to use for attack probes (default 2)
    random_state : reproducible sampling seed (default 7)

    Returns
    -------
    (checks, atk_rows)
        checks   : dict[label, {"status": "ok"|"warn"|"fail", "detail": str}]
        atk_rows : list[(attack_name, [(orig, attacked, changed), ...])]

    Side-effect
    -----------
    Renders an HTML dashboard via ``IPython.display`` when inside a Jupyter
    kernel.  Falls back to a plain-text summary in non-interactive environments.
    """
    sample = dev_df.sample(n_sample, random_state=random_state)["sentence"].tolist()

    checks: dict[str, dict] = {}
    checks["Dataset"]              = _check_dataset(dev_df)
    checks["Target model"]         = _check_target(target)
    checks["Encoder (semantic sim)"] = _check_encoder(encoder)

    attack_checks, atk_rows = _check_attacks(attacks, sample)
    checks.update(attack_checks)

    html = render_sanity_html(checks, atk_rows)

    # Display in Jupyter if available; otherwise print a plain summary
    try:
        from IPython.display import display, HTML
        display(HTML(html))
    except Exception:
        n_ok   = sum(1 for v in checks.values() if v["status"] == "ok")
        n_warn = sum(1 for v in checks.values() if v["status"] == "warn")
        n_fail = sum(1 for v in checks.values() if v["status"] == "fail")
        print(f"Sanity check: {n_ok} ok, {n_warn} warn, {n_fail} fail")
        for label, v in checks.items():
            icon = {"ok": "✅", "warn": "⚠️", "fail": "❌"}[v["status"]]
            print(f"  {icon} {label}: {v['detail']}")

    return checks, atk_rows
