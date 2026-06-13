# NLI Robustness — Design & Methodology (Notebook 05)

[← Back to README](../README.md) · [Open notebook](../notebooks/05_nli_robustness_demo.ipynb)

Measures whether the model still **reasons correctly** when the input is crafted to fool it. NB01–03 attack the model (evasion, jailbreak, injection) and NB04 measures a harm (bias); this notebook measures a **capability under adversarial pressure** — Natural Language Inference (NLI).

---

## Results — GPT-5-4 via Azure (full run, 13,298 items: 9,815 MultiNLI + 3,200 ANLI + 283 AdvGLUE)

![NLI robustness summary](images/nb05_robustness.png)

| Dataset | Accuracy | Gap vs clean |
|---|---|---|
| MultiNLI (clean baseline) | **85.5%** | — |
| AdvGLUE mnli (matched) | 80.2% | +5.3% |
| AdvGLUE mnli-mismatched | 70.4% | +15.1% |
| ANLI R1 | 79.0% | +6.5% |
| ANLI R2 | 68.4% | +17.1% |
| ANLI R3 | 64.7% | **+20.8%** |

The model is strong on ordinary inference (85.5%) and robust to surface perturbations (AdvGLUE matched, ANLI R1), but its reasoning **degrades steadily under adversarial pressure** — a clean monotonic ANLI difficulty curve (**79% → 68% → 65%**) ending in a **+20.8% robustness gap** on ANLI R3.

**Dominant failure mode: hedging toward "neutral".** `entailment → neutral` is the single largest error (**37%** of hard-round misses), and together with `contradiction → neutral` (26%) roughly **63% of errors land on "neutral"** — the model retreats to the safe middle when a verdict requires a world-knowledge or multi-step leap (e.g. *"an equation named after Chandrasekhar"* ⟹ *"he was influential"*). The confusion matrix shows this as a heavy *neutral* column.

**Scale matters:** at 13,298 items no dataset crosses the 25% "material" threshold — the worst gap is 20.8% (moderate). The noisier 100-item pilot had over-stated ANLI R3 at 26%, so the honest full-run read is **graceful degradation, not collapse**.

**Caveats:** 94 of 13,298 replies (0.7%) were unparseable, mostly due to upstream **content filtering** (counted as wrong — a measurement artifact, not a reasoning failure); and a minority of ANLI gold labels are themselves **contestable** (the `reason` field sometimes reveals a debatable judgement), so the true gap may be marginally smaller. Treat flagged ANLI items as candidates for human review.

### Executive report

📄 **[Open the interactive sample report →](https://htmlpreview.github.io/?https://github.com/minw0607/llm_red_teaming/blob/main/docs/samples/nli_executive_summary.html)** · [Raw HTML](samples/nli_executive_summary.html) *(model name redacted)*

[![NLI executive summary](images/nb05_executive_summary.png)](https://htmlpreview.github.io/?https://github.com/minw0607/llm_red_teaming/blob/main/docs/samples/nli_executive_summary.html)

> ⚠️ **Illustrative sample** — accuracy figures come from an automated harness scoring fixed public benchmarks; adversarial sets are deliberately hard, so low scores reflect benchmark difficulty, not a production failure rate.

---

## What NLI tests, and why adversarial NLI

**Natural Language Inference** asks the model to decide the logical relationship between a **premise** and a **hypothesis**:

| Label | Meaning |
|---|---|
| **entailment** | if the premise is true, the hypothesis must be true |
| **neutral** | the hypothesis might be true, but the premise doesn't guarantee it |
| **contradiction** | if the premise is true, the hypothesis must be false |

NLI is the canonical probe of whether a model *reasons* over text rather than pattern-matching surface words. It underpins fact-checking, RAG faithfulness, and contract/policy analysis — anywhere a model must judge whether a claim follows from a source.

**Why adversarial.** Frontier models score very high on ordinary NLI, so clean accuracy alone *overstates* reliability. The headline metric is therefore the **robustness gap**:

```
robustness_gap = clean_accuracy (MultiNLI) − adversarial_accuracy (ANLI / AdvGLUE)
```

A large gap means the model handles ordinary inference but is brittle on adversarially-constructed reasoning — exactly the inputs a production system meets in the wild. This is a **reliability / assurance** finding, not a security breach.

---

## How NB05 differs from NB01

| | NB01 — Adversarial NLP | NB05 — NLI Robustness |
|---|---|---|
| Task | Sentiment (SST-2, binary) | Entailment (3-class reasoning) |
| Adversary | An *algorithm* we run (TextFooler, etc.) | The *dataset* — humans wrote items to fool models |
| Signal | Prediction flip rate under perturbation | Accuracy gap clean → adversarial |
| Risk class | Evasion (MITRE ATLAS) | Information Integrity / accuracy (assurance) |

The two are complementary: NB01 measures sensitivity to *surface* perturbation on a simple task; NB05 measures *reasoning* robustness on a hard one.

---

## Datasets (single 3-way label space: 0 entailment · 1 neutral · 2 contradiction)

| Dataset | Role | Notes |
|---|---|---|
| **MultiNLI** ([`nyu-mll/multi_nli`](https://huggingface.co/datasets/nyu-mll/multi_nli)) | Clean baseline | `validation_matched`, gold-labelled — the reference accuracy |
| **ANLI** ([`facebook/anli`](https://arxiv.org/abs/1910.14599)) | Adversarial (human) | 3 rounds R1→R3 of escalating difficulty (Nie et al., 2020). Each item ships a human **`reason`** annotation explaining the trap |
| **AdvGLUE** ([`adv_glue/adv_mnli*`](https://arxiv.org/abs/2111.02840)) | Adversarial (perturbed) | Adversarially-perturbed MNLI (Wang et al., 2021); two configs kept to preserve the 3-way label space |

All fetched once via the `datasets` library and cached to `eval_datasets/robustness/` (gitignored). A small in-repo fallback keeps the notebook runnable offline.

**ANLI's `reason` field** is the standout asset: where the model errs, the notebook surfaces the human annotator's explanation of *why the item is hard* — turning each failure into an interpretable reasoning weakness rather than an opaque miss.

---

## Methodology

- **Zero-shot classification** (`attacks/robustness/nli.py`). A fixed system prompt defines the three labels and asks for exactly one word. The reply is parsed deterministically to a label id (earliest explicit label wins); unparseable replies are recorded as `pred = -1` and **count as wrong** — never silently dropped.
- **One resume-safe checkpoint** keyed by `(source, idx)`, so all datasets share a single `.jsonl` and an interrupted run resumes for free.
- **Deterministic scoring** — no judge LLM is involved in computing accuracy.

### Metrics (`evaluate/nli_metrics.py`)

| Metric | What it shows |
|---|---|
| Per-dataset accuracy + unparsed rate | Clean vs. adversarial, side by side |
| **Robustness gap** | Clean − adversarial accuracy, per adversarial source (the headline) |
| ANLI difficulty curve | Accuracy by round (R1 → R3) — how steeply reasoning degrades |
| 3×3 confusion matrix | Which way errors go (e.g. *neutral* misread as *contradiction*) |
| Error cases (+ ANLI `reason`) | Human-review queue of misclassifications with the annotator's trap explanation |

### Executive report (`evaluate/nli_executive.py`)

Same pattern as NB02–04: deterministic metrics + a judge-LLM narrative, rendered as a business-level HTML report with the illustrative-sample disclaimer. The judge writes only the *narrative*; every number is computed deterministically.

---

## Regulatory mapping

Reasoning robustness is an **assurance** property, so it maps to integrity/accuracy obligations rather than attack catalogues.

| Framework | Reference | Why it applies |
|---|---|---|
| **NIST AI 600-1** | §2.5 Information Integrity · §2.2 Confabulation | Mis-inferring entailment under adversarial input lets a model assert false conclusions as true |
| **MITRE ATLAS** | AML.T0043 (Craft Adversarial Data) | ANLI/AdvGLUE are adversarial-example evasion of an NLP capability |
| **OWASP LLM Top 10** | LLM09 Misinformation | Brittle reasoning surfaces as confidently wrong outputs |
| **EU AI Act** | Art. 15 (accuracy, robustness & cybersecurity) | High-risk systems must perform consistently and resist adversarial manipulation |

---

## Mitigations

Add adversarial NLI to the evaluation suite (clean accuracy alone is insufficient) · chain-of-thought / self-consistency prompting for inference-heavy tasks · abstention when the premise is under-determined · monitor the robustness gap as a release gate.

---

## Next Steps

SNLI / contract-NLI domain transfer · paraphrase-consistency and negation-robustness metrics (`evaluate/consistency.py`) · calibration/confidence shift under adversarial input · few-shot vs. zero-shot robustness comparison.

See [industry_alignment.md](industry_alignment.md) and [dataset_strategy.md](dataset_strategy.md) for the broader programme.
