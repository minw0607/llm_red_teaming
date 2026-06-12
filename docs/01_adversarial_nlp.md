# Adversarial NLP Attacks — Full Results (Notebook 01)

[← Back to README](../README.md) · [Open notebook](../notebooks/01_adversarial_nlp_demo.ipynb)

**10 adversarial attacks** evaluated on **872 SST-2 samples** against an Azure-hosted GPT-5-4 model. At this scale each percentage point represents ~8–9 real prediction flips, making the rankings stable and operationally meaningful.

The attacks span five perturbation levels — character, word, sentence, semantic, and structural — and are all black-box, requiring only API access to the target.

---

## Attacks Evaluated

These attacks perturb input text at increasing levels of abstraction and measure prediction flip rate on a classification task (SST-2 sentiment). All map to three industry frameworks: [MITRE ATLAS](https://atlas.mitre.org/), [NIST AI 600-1](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf), and [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/).

| Attack | Level | Description | MITRE ATLAS | NIST AI 600-1 | OWASP LLM Top 10 |
|---|---|---|---|---|---|
| **TextBugger** | Character | Substitutes one character near the start of a high-impact word | AML.T0043 · AML.T0015 | Information Security | LLM09 Misinformation |
| **DeepWordBug** | Character | Inserts, deletes, or swaps one character to break tokenisation | AML.T0043 · AML.T0015 | Information Security | LLM09 Misinformation |
| **TextFooler** | Word | Replaces one word with a top-k WordNet synonym that preserves part-of-speech | AML.T0043 · AML.T0015 | Information Security · Information Integrity | LLM09 Misinformation |
| **BERTAttack** | Word | Replaces one word using BERT fill-mask, filtered to cosine similarity ≥ 0.85 | AML.T0043 · AML.T0015 | Information Security · Information Integrity | LLM09 Misinformation |
| **CheckList** | Sentence | Appends a random alphanumeric token to test sensitivity to irrelevant context | AML.T0043 | Information Integrity | LLM09 Misinformation |
| **StressTest** | Sentence | Appends a logically vacuous tautology to probe attention drift | AML.T0043 | Information Integrity · Confabulation | LLM09 Misinformation |
| **SemanticAttack** | Semantic | POS-aware WordNet synonym swap — meaning-preserving, highest stealth | AML.T0043 · AML.T0015 | Information Integrity · Information Security | LLM09 Misinformation |
| **BackTranslation** | Structural | Round-trip EN→DE→EN via MarianMT — fluent paraphrases with high semantic fidelity | AML.T0043 · AML.T0015 | Information Integrity | LLM09 Misinformation |
| **Homoglyph** | Structural | Replaces ASCII characters with visually identical Unicode lookalikes (Cyrillic, Greek) | AML.T0043 | Information Security | LLM01 Prompt Injection · LLM09 Misinformation |
| **NegationInjection** | Structural | Inserts or removes negation words (`not`, `never`, `hardly`) to flip logical meaning | AML.T0043 · AML.T0015 | Information Integrity | LLM09 Misinformation |

**Evaluation metrics:** Accuracy Drop · Attack Success Rate (ASR) · Stealth Score (composite) · Risk Score (Impact × Stealth) · Human Review Queue (HIGH / MEDIUM / LOW)

---

## Risk Matrix & Accuracy Comparison

![Risk Matrix](images/nb01_risk_matrix.png)

*Bubble size = Attack Success Rate (ASR). Shaded region = stealth ≥ 0.80 (danger zone).*

---

## Attack Summary

*n = 872 samples, SST-2 sentiment, sorted by risk score.*

| Attack | Level | Orig Acc | Attacked Acc | Acc Drop | ASR | Stealth | Risk Score |
|---|---|---|---|---|---|---|---|
| 🔴 **NegationInjection** | structural | 95.5% | 78.1% | **+17.5%** | 19.6% | 0.941 | **0.1647** |
| 🟠 **StressTest** | sentence | 95.5% | 92.2% | +3.3% | 4.5% | 0.888 | 0.0293 |
| 🟡 **BackTranslation** | structural | 95.2% | 93.5% | +1.7% | 2.8% | 0.897 | 0.0153 |
| 🟡 **TextFooler** | word | 95.7% | 94.1% | +1.6% | 2.7% | 0.872 | 0.0140 |
| 🟡 **SemanticAttack** | semantic | 95.5% | 94.6% | +0.9% | 1.8% | 0.901 | 0.0081 |
| **TextBugger** | character | 95.2% | 94.7% | +0.5% | 1.6% | 0.816 | 0.0041 |
| **BERTAttack** | word | 95.5% | 95.2% | +0.4% | 1.0% | — | — |
| **DeepWordBug** | character | 95.0% | 95.2% | −0.1% | 0.9% | 0.833 | 0.0000 |
| **CheckList** | sentence | 95.5% | 95.7% | −0.2% | 0.7% | 0.818 | 0.0000 |
| **Homoglyph** | structural | 95.3% | 95.9% | −0.6% | 1.0% | 0.824 | 0.0000 |

*Stealth = composite score (0.5 × semantic_sim + 0.3 × naturalness + 0.2 × edit_sim). Risk Score = Acc Drop × Stealth. Negative acc drop = model improved (data cleaning / noise reduction effect).*

**Key findings at n = 872:**
- 🔴 **NegationInjection dominates**: 17.5% accuracy drop at 0.941 stealth — 5× larger than the second-ranked attack. Logically inverted text remains fluent (ppl_ratio 1.097) and is undetectable by perplexity monitors
- 🟠 **StressTest is the only other HIGH-risk attack**: 3.3% drop — vacuous tautology appended to input causes attention drift, not semantic manipulation
- 🟡 **Word/structural attacks cluster at 1–2%**: BackTranslation, TextFooler, SemanticAttack all stealth ≥ 0.87 — meaningful but individually manageable
- ✅ **Character-level attacks ineffective at frontier scale**: TextBugger (+0.5%), DeepWordBug (−0.1%) — GPT-5-class models are robust to single-character typo perturbations
- ℹ️ **Three attacks improve accuracy**: Homoglyph (−0.6%), CheckList (−0.2%), DeepWordBug (−0.1%) — perturbations land on harder sample variants, confirming these attacks have near-zero adversarial signal at this model scale

---

## Notable Findings & Interpretations

| Finding | What happened | Why it matters |
|---|---|---|
| **NegationInjection: 17.5% at n=872** | Gap to second-place (3.3%) is 5×, stable across 868 samples — not sampling noise | Structural logical attacks are the dominant threat for this model class. Character and word substitutions are largely solved at frontier scale; negation robustness is not. |
| **BackTranslation: +1.7% acc drop** | EN→DE→EN normalises SST-2's noisy pre-tokenised text (removes spaces, fixes capitalisation); ppl_ratio 0.906 < 1 confirms attacked text is *more fluent* than original | Confirms the data-cleaning effect holds at scale. The 2.8% ASR reflects genuine adversarial signal on a small hard subset; the net effect is mild because most samples improve after round-trip translation. |
| **BERTAttack: missing stealth data** | Stealth checkpoint did not complete for BERTAttack in this run — composite stealth columns show `—` | Risk score cannot be computed; treat as a data gap, not zero risk. Re-run `add_stealth_components()` with `checkpoint_path` to fill. Acc drop of 0.4% / ASR 1.0% visible from raw results. |
| **Character attacks: at or below baseline** | TextBugger +0.5%, DeepWordBug −0.1% at n=872 | Deprioritise in future red-team cycles. Frontier LLMs are effectively immune to surface-level typo attacks; testing budget is better spent on structural and prompt-level attacks. |
| **BERTAttack: HIGH→MEDIUM demotion** | `text_changed=False` flip cases (identical text, different model answer) correctly demoted | Model non-determinism on decision-boundary sentences, not attack success. Documented in `display_human_review()` with explicit flagging. |

---

## Executive Security Report

Notebook 01 auto-generates a **business-level security assessment report** (Step 9) using a judge LLM to interpret findings into plain-English risk assessments, regulatory citations, and prioritised recommendations.

[![Executive Summary Report](images/nb01_executive_summary.png)](https://htmlpreview.github.io/?https://github.com/minw0607/llm_red_teaming/blob/main/docs/samples/executive_summary_n872.html)

<div align="center">

📄 **[Open interactive report →](https://htmlpreview.github.io/?https://github.com/minw0607/llm_red_teaming/blob/main/docs/samples/executive_summary_n872.html)**  ·  [Raw HTML](samples/executive_summary_n872.html) *(n=872 full run · model name redacted)*

</div>

**What the report contains:**
- **Overall risk verdict** and risk level (LOW / MEDIUM / HIGH / CRITICAL) — written in plain English for a non-technical leadership audience
- **Key findings** — each with severity badge, headline, and 2–3 sentence business-impact explanation
- **Attack results table** — all 10 attacks with acc drop, ASR, stealth, and risk score
- **Attack methodology** — what each perturbation level simulates in plain English
- **Regulatory obligations** — NIST AI 600-1 · MITRE ATLAS · OWASP LLM Top 10 · EU AI Act citations grounded in actual findings
- **Prioritised recommendations** — numbered action items with rationale tied to specific data

> Numbers in the report come from deterministic metric computation. The judge LLM writes only the *narrative* — it cannot hallucinate the metrics.

---

## Regulatory Alignment

**Dynamic regulatory impact mapping** (`regulatory_report()` + `render_regulatory_heatmap()`):
- Finding-level citations — only attacks with meaningful impact in *this run* are reported; actual metric values (acc_drop, ASR, stealth) embedded in each citation
- Special-case detection — data-cleaning effects (BackTranslation), model non-determinism (BERTAttack), and high-stealth structural attacks each map to specific obligations
- **Frameworks:** NIST AI 600-1 (§2.3, §2.5, §2.6) · MITRE ATLAS (AML.T0043, T0015, T0016, T0040) · OWASP LLM Top 10 (LLM01, LLM09) · EU AI Act (Art. 9, 13, 15, 17)

```python
from evaluate import regulatory_report, render_regulatory_heatmap, generate_executive_summary

reg_df            = regulatory_report(summary_df, review_df=review_df)
render_regulatory_heatmap(reg_df, save_path='results/regulatory_heatmap.png')
exec_html, data   = generate_executive_summary(summary_df, review_df, reg_df, target, config)
```

**Saved outputs** (all in `results/`, gitignored by default):

```
01_raw_results_n872.csv          per-sample predictions (all 10 attacks × 872 rows)
01_attack_summary_n872.csv       per-attack metrics (acc_drop, ASR, stealth, risk_score)
01_attack_summary_n872.html      styled metrics table — open in browser
01_human_review_n872.csv         flagged adversarial cases (HIGH / MEDIUM / LOW)
01_executive_summary_n872.html   executive report — open in browser or share directly
```
