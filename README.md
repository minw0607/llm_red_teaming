<div align="center">

# 🔴 LLM Red Teaming

**A modular, extensible toolkit for adversarial testing of large language models and NLP systems.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](CONTRIBUTING.md)

*Adversarial text attacks · prompt injection · jailbreaking · fairness probing · pluggable model targets*

</div>

---

## 📖 Overview

Modern AI systems are increasingly deployed in sensitive contexts — yet their robustness to adversarial inputs remains poorly understood. **LLM Red Teaming** provides a structured, reproducible framework to:

- **Attack** language models at multiple levels: character, word, sentence, semantic, and prompt
- **Jailbreak** instruction-tuned LLMs using standardised benchmarks and custom templates
- **Evaluate** robustness metrics: accuracy drop, attack success rate (ASR), stealth score, risk score
- **Flag** high-risk adversarial examples for human review with priority queuing
- **Align** every evaluation to industry standards: MITRE ATLAS, NIST AI RMF, NIST AI 600-1, OWASP LLM Top 10, EU AI Act

The toolkit is organised into **workstreams**, each delivered as a code-light demo notebook backed by reusable modules. Jump to a workstream:

| Workstream | Status | Front-page section | Full results |
|---|:---:|---|---|
| 🧬 **Adversarial NLP** | ✅ Complete | [↓ jump](#-adversarial-nlp-notebook-01) | [docs/01](docs/01_adversarial_nlp.md) |
| 🔓 **Jailbreaking** | ✅ Complete | [↓ jump](#-jailbreaking-notebook-02) | [docs/02](docs/02_jailbreaking.md) |
| 💉 **Prompt Injection** | 🔜 Next | [↓ jump](#-prompt-injection-notebook-03) | — |
| ⚖️ **Fairness and NLI Robustness** | 📋 Planned | [↓ jump](#-fairness-and-nli-robustness-notebooks-0405) | — |

---

## 🗂️ Repository Structure

```
llm_red_teaming/
│
├── attacks/                    # All attack implementations
│   ├── character/              # TextBugger, DeepWordBug              [✅ implemented]
│   ├── word/                   # TextFooler, BERTAttack                [✅ implemented]
│   ├── sentence/               # CheckList, StressTest                 [✅ implemented]
│   ├── semantic/               # SemanticAttack                        [✅ implemented]
│   ├── structural/             # BackTranslation, Homoglyph, Negation  [✅ implemented]
│   ├── jailbreak/              # JailbreakBench + PAIR artifact runners [✅ implemented]
│   └── prompt/                 # Injection, role-play                  [🔜 next]
│
├── judges/                     # Response evaluation
│   └── classifier_judge.py     # Rule-based + zero-shot BART-MNLI judge
│
├── targets/                    # Model connectors (pluggable)
│   └── azure_openai.py         # Azure OpenAI / APIM gateway
│
├── evaluate/                   # Metrics and reporting
│   ├── metrics.py              # ASR, risk score, human review queue   [✅ implemented]
│   ├── adversarial_eval.py     # run_all_attacks() pipeline            [✅ implemented]
│   ├── stealth.py              # Composite stealth score               [✅ implemented]
│   ├── plots.py                # Risk matrix + accuracy bar charts     [✅ implemented]
│   ├── display.py              # Human review display helper           [✅ implemented]
│   ├── regulatory.py           # Dynamic regulatory impact mapping     [✅ implemented]
│   ├── executive.py            # LLM-interpreted executive report      [✅ implemented]
│   ├── sanity.py               # Pre-run readiness validator           [✅ implemented]
│   ├── consistency.py          # Paraphrase & self-consistency         [📋 planned]
│   └── fairness.py             # Counterfactual demographic scorer     [📋 planned]
│
├── eval_datasets/              # Evaluation datasets
│   ├── sst2/                   # SST-2 sentiment benchmark             [✅ implemented]
│   ├── nli/                    # SNLI, MultiNLI, ANLI, AdvGLUE         [📋 planned]
│   ├── toxicity/               # ToxiGen, HateXplain                   [📋 planned]
│   └── safety/                 # AdvBench, HarmBench jailbreak banks   [📋 planned]
│
├── notebooks/                  # Lightweight demo notebooks
│   ├── 01_adversarial_nlp_demo.ipynb   # 10 attacks × SST-2 [✅]
│   ├── 02_jailbreaking_demo.ipynb      # JailbreakBench    [✅]
│   ├── 03_prompt_injection.ipynb       # Direct + indirect  [🔜]
│   ├── 04_fairness_counterfactual.ipynb # Demographic swap  [📋]
│   └── 05_nli_robustness.ipynb         # AdvGLUE + ANLI    [📋]
│
├── docs/                       # Per-workstream results & deep dives
│   ├── 01_adversarial_nlp.md   # NB01 full results (n=872)            [✅]
│   ├── 02_jailbreaking.md      # NB02 full results (3 modes)          [✅]
│   ├── images/                 # Figures referenced in docs
│   └── samples/                # Sample report outputs
│       └── executive_summary_n872.html   # Redacted executive summary (n=872)
├── configs/                    # Experiment configuration files
├── results/                    # Output files (gitignored)
├── .env.example                # API key template
├── requirements.txt
└── LICENSE
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/minw0607/llm_red_teaming.git
cd llm_red_teaming
pip install -r requirements.txt
cp .env.example .env          # fill in your Azure OpenAI credentials
```

Open any notebook in `notebooks/` and set the parameters in its config cell — everything else runs end-to-end.

```python
# Programmatic usage
from attacks.character import TextBugger
from attacks.word import TextFooler
from targets.azure_openai import AzureOpenAITarget
from evaluate import run_all_attacks, compute_attack_summary

attacks = {"TextBugger": TextBugger(), "TextFooler": TextFooler()}
target  = AzureOpenAITarget()
results = run_all_attacks(attacks, target, eval_df, n_samples=50)
summary = compute_attack_summary(results)
```

---

## 🧬 Adversarial NLP (Notebook 01)

`Status: ✅ Complete`

**What it tests:** how small, often-imperceptible text perturbations degrade a model's accuracy on a classification task (SST-2 sentiment). 10 black-box attacks across 5 perturbation levels — character, word, sentence, semantic, structural.

**Headline result** (GPT-5-4 via Azure, n = 872 samples):

| Attack | Level | Acc Drop | ASR | Stealth | Risk Score |
|---|---|---|---|---|---|
| 🔴 **NegationInjection** | structural | **+17.5%** | 19.6% | 0.941 | **0.1647** |
| 🟠 **StressTest** | sentence | +3.3% | 4.5% | 0.888 | 0.0293 |
| 🟡 **BackTranslation** | structural | +1.7% | 2.8% | 0.897 | 0.0153 |

NegationInjection dominates — a 17.5% accuracy drop at 0.941 stealth, 5× the next attack, and undetectable by perplexity monitors. Character-level attacks are effectively neutralised at frontier scale. [See all 10 attacks, the risk matrix, and per-finding interpretation →](docs/01_adversarial_nlp.md)

Notebook 01 also auto-generates a **business-level executive security report** — a judge LLM interprets the deterministic metrics into a plain-English risk verdict, regulatory citations, and prioritised recommendations (the metrics themselves are never LLM-generated):

[![Executive Summary Report](docs/images/nb01_executive_summary.png)](https://htmlpreview.github.io/?https://github.com/minw0607/llm_red_teaming/blob/main/docs/samples/executive_summary_n872.html)

<div align="center">

📄 **[Open interactive report →](https://htmlpreview.github.io/?https://github.com/minw0607/llm_red_teaming/blob/main/docs/samples/executive_summary_n872.html)**  ·  [Full NB01 results →](docs/01_adversarial_nlp.md)  ·  [Open notebook →](notebooks/01_adversarial_nlp_demo.ipynb)

</div>

---

## 🔓 Jailbreaking (Notebook 02)

`Status: ✅ Complete`

**What it tests:** whether harmful-intent prompts can bypass the model's safety alignment. Uses [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) (100 harmful behaviors) across three escalating attack modes, scored by a BART-MNLI classifier judge.

**Headline result** (GPT-5-4 via Azure, 172 prompts):

| Test | N | ASR | Blocked | Refusal |
|---|---|---|---|---|
| **Direct Goals** | 50 | 0.0% | 46% | 50% |
| **Artifact Templates** | 80 | 0.0% | 88% | 8% |
| **PAIR (Vicuna-13B transfer)** | 42 | 2.4% | 29% | 62% |

The model held firm: 0% ASR on direct and template-wrapped attacks (Azure Prompt Shields blocked most at the platform layer), with a single borderline violation from PAIR transfer. [See the three modes, judge upgrade path, dataset gaps, and full regulatory mapping →](docs/02_jailbreaking.md)

**Regulatory mapping:** MITRE ATLAS (AML.T0054, AML.T0006) · OWASP LLM Top 10 (LLM01, LLM06, LLM07) · NIST AI 600-1 (§2.1, §2.6, §2.8) · EU AI Act (Art. 9, Art. 15)

<div align="center">

[Full NB02 results →](docs/02_jailbreaking.md)  ·  [Open notebook →](notebooks/02_jailbreaking_demo.ipynb)

</div>

---

## 💉 Prompt Injection (Notebook 03)

`Status: 🔜 Next`

**What it will test:** whether adversarial instructions can override the system prompt or hijack model behaviour — both directly (in user input) and indirectly (via retrieved content in RAG pipelines).

| Attack | Vector | MITRE ATLAS | OWASP LLM Top 10 |
|---|---|---|---|
| **Direct Prompt Injection** | Adversarial instructions in user input override the system prompt | AML.T0054 · AML.T0040 | LLM01 Prompt Injection |
| **Indirect Prompt Injection** | Adversarial instructions in external content the LLM retrieves (emails, docs, web) | AML.T0054 · AML.T0040 | LLM01 Prompt Injection · LLM08 Vector & Embedding |

**Primary metric:** override rate — fraction of injected instructions that change model behaviour. See the [Testing Strategy Roadmap](#-testing-strategy-roadmap) for the full plan.

---

## ⚖️ Fairness and NLI Robustness (Notebooks 04/05)

`Status: 📋 Planned`

**What it will test:**
- **Counterfactual fairness (NB04)** — swap protected attributes (name, gender, race, nationality) and measure whether model output changes. Tests fairness, not just robustness.
- **NLI robustness (NB05)** — adversarial robustness on AdvGLUE + ANLI, extending coverage beyond binary sentiment to logical entailment and reasoning.

These workstreams introduce the `evaluate/fairness.py` and `evaluate/consistency.py` modules and the NLI dataset family. See the [Dataset Roadmap](#-dataset-roadmap) and [Testing Strategy Roadmap](#-testing-strategy-roadmap).

---

## 🗺️ Roadmap & Standards

All attacks map to four industry frameworks:
- **[MITRE ATLAS](https://atlas.mitre.org/)** — adversarial ML technique catalogue
- **[NIST AI 600-1](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)** — GenAI risk profile (July 2024)
- **[OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)** — application-layer LLM security risks (2025)
- **[EU AI Act](https://artificialintelligenceact.eu/)** — high-risk AI system obligations (2024/1689)

Status legend: ✅ Implemented · 🔜 Next milestone · 📋 Planned · 🔭 Research horizon

> **Implemented attacks** are documented per-workstream above — see [docs/01](docs/01_adversarial_nlp.md) (10 adversarial NLP attacks) and [docs/02](docs/02_jailbreaking.md) (jailbreak modes), each with full standards mapping. The tables below cover **future** work.

### Future Attack Library — Tier 2 (Medium Priority)

| Status | Attack | Category | Description | MITRE ATLAS | NIST AI 600-1 | OWASP LLM Top 10 |
|:---:|---|---|---|---|---|---|
| 📋 | **Paraphrase Model Attack** | Semantic | Use a paraphrase model (PEGASUS, DIPPER, T5) to generate fluent rewrites — more naturalistic than synonym lookups | AML.T0043 · AML.T0015 | Information Integrity | LLM09 Misinformation |
| 📋 | **Counterfactual / Demographic Swap** | Fairness | Swap protected attributes (name, gender, race, nationality) and measure if model output changes — tests fairness, not just robustness | AML.T0043 | Harmful Bias and Homogenization | LLM09 Misinformation |
| 📋 | **Payload Splitting** | Prompt | Distribute a forbidden phrase across multiple tokens, turns, or encoded segments to evade safety filters | AML.T0054 | Information Security | LLM01 Prompt Injection · LLM07 System Prompt Leakage |
| 📋 | **Many-Shot Jailbreak** | Prompt | Provide many in-context examples of policy-violating exchanges before the target request to shift the model's behaviour (Anthropic, 2024) | AML.T0054 | Information Security (§2.6) | LLM01 Prompt Injection |
| 📋 | **Crescendo / Multi-Turn** | Prompt | Escalate harmful requests across multiple conversational turns, each individually benign (Microsoft, 2024) | AML.T0054 · AML.T0006 | Information Security (§2.6) | LLM01 Prompt Injection · LLM06 Excessive Agency |
| 📋 | **GCG / Adversarial Suffix** | Gradient | Greedy Coordinate Gradient (Zou et al. 2023) — finds a universal adversarial suffix that forces harmful outputs; transferable to black-box targets | AML.T0043 · AML.T0015 | Information Security (§2.6) | LLM01 Prompt Injection |
| 📋 | **Multilingual Bypass** | Structural | Submit harmful requests in low-resource languages where safety fine-tuning is weaker | AML.T0054 | Information Security (§2.6) · Harmful Bias (§2.8) | LLM01 Prompt Injection |
| 📋 | **HarmBench Expansion** | Dataset | Extend jailbreak coverage from 100 (JailbreakBench) to 400 behaviors across 7 harm categories with pre-published ASR baselines for model comparison | AML.T0054 | Information Security (§2.6) · CBRN (§2.1) | LLM01 Prompt Injection |
| 📋 | **StrongREJECT Scoring** | Metric | Replace binary ASR with StrongREJECT score (Souly et al. 2024) — penalises partial compliance; more accurate than BART-MNLI on ambiguous refusals | AML.T0054 | Information Security (§2.6) | LLM01 Prompt Injection |
| 📋 | **Llama Guard Judge** | Judge | Replace BART-MNLI with Meta's Llama Guard 3 (fine-tuned safety classifier) — lower false-positive rate on sensitive-topic academic responses | AML.T0054 | Information Security (§2.6) | LLM01 Prompt Injection |

### Future Attack Library — Tier 3 (Research Horizon)

| Status | Attack | Category | Description | MITRE ATLAS | NIST AI 600-1 | OWASP LLM Top 10 |
|:---:|---|---|---|---|---|---|
| 🔭 | **RAG / Vector Store Poisoning** | Infrastructure | Inject adversarial documents into a retrieval corpus to influence LLM responses via indirect context | AML.T0054 · AML.T0020 | Information Security · Value Chain | LLM08 Vector & Embedding Weaknesses |
| 🔭 | **Multi-Turn Context Manipulation** | Prompt | Build up a false context or persona over many dialogue turns to gradually shift model behaviour | AML.T0054 | Information Security · Human-AI Configuration | LLM01 Prompt Injection |
| 🔭 | **Tool / Function Call Hijacking** | Prompt | Craft adversarial input that redirects an LLM agent's tool calls to unintended targets or actions | AML.T0054 | Information Security | LLM06 Excessive Agency |
| 🔭 | **Backdoor / Trojan Trigger** | Model-level | Insert a hidden trigger phrase during fine-tuning that causes targeted misclassification at inference | AML.T0020 · AML.T0043 | Information Security · Data Provenance | LLM04 Data & Model Poisoning |
| 🔭 | **Membership Inference** | Privacy | Query the model systematically to determine whether specific examples were in its training data | AML.T0040 · AML.T0031 | Data Privacy | LLM02 Sensitive Information Disclosure |
| 🔭 | **Model Extraction / Stealing** | Privacy | Reconstruct a functional surrogate of the target model via repeated black-box queries | AML.T0031 · AML.T0040 | Value Chain and Component Integration | LLM02 Sensitive Information Disclosure |
| 🔭 | **Training Data Extraction** | Privacy | Prompt the model to reproduce memorised training data (PII, copyrighted text) | AML.T0031 | Data Privacy · Intellectual Property | LLM02 Sensitive Information Disclosure |

### 📊 Dataset Roadmap

| Status | Dataset | Task | What it adds | Source |
|:---:|---|---|---|---|
| ✅ | **SST-2** | Binary sentiment | Baseline — fast to score, sensitive to lexical changes | [HuggingFace](https://huggingface.co/datasets/stanfordnlp/sst2) |
| ✅ | **JailbreakBench** | Safety / jailbreak | 100 harmful behaviors + PAIR/GCG artifact library | [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) |
| 📋 | **HarmBench** | Safety / jailbreak | 400 behaviors across 7 harm categories; published ASR baselines | [HarmBench](https://github.com/centerforaisafety/HarmBench) |
| 📋 | **AdvGLUE** | NLI, QA, sentiment | Already adversarially constructed; drop-in replacement for GLUE tasks | [Yang et al. 2021](https://arxiv.org/abs/2106.09680) |
| 📋 | **ANLI** | 3-class entailment | Collected via adversarial human-in-the-loop; harder than SNLI | [Nie et al. 2020](https://arxiv.org/abs/1910.14599) |
| 📋 | **MultiNLI** | 3-class entailment | Tests logical reasoning robustness across 10 genres | [HuggingFace](https://huggingface.co/datasets/nyu-mll/multi_nli) |
| 📋 | **ToxiGen / HateXplain** | Toxicity classification | Safety-critical: does the model correctly flag hate speech after perturbation? | [ToxiGen](https://arxiv.org/abs/2203.09509) · [HateXplain](https://arxiv.org/abs/2012.10289) |
| 📋 | **TriviaQA** | Open-domain QA | Does a factual answer change when the question is rephrased? | [HuggingFace](https://huggingface.co/datasets/trivia_qa) |
| 🔭 | **MMLU** (select subsets) | Multi-domain MCQ | Domain-specific robustness (medical, legal, STEM) under prompt rephrasing | [HuggingFace](https://huggingface.co/datasets/cais/mmlu) |
| 🔭 | **MT-Bench** | Instruction following | Does multi-turn output quality degrade under adversarial system prompts? | [LMSYS](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) |

### 🧪 Testing Strategy Roadmap

| Status | Strategy | What it measures | Primary dataset(s) | Key metric |
|:---:|---|---|---|---|
| ✅ | **Prediction Flip (ASR)** | Fraction of correct predictions overturned by an attack | SST-2 | Attack Success Rate |
| ✅ | **Risk Scoring** | Composite danger = Impact × Stealth; ranks attacks by operational priority | SST-2 | Risk Score |
| ✅ | **Human Review Queue** | Prioritises adversarial examples by flip + stealth for manual inspection | All | HIGH / MEDIUM / LOW |
| ✅ | **Composite Stealth Scoring** | Semantic similarity + perplexity ratio + normalised edit distance — richer imperceptibility signal than cosine similarity alone | All | Weighted composite |
| ✅ | **Jailbreak Success Rate (ASR)** | Fraction of jailbreak prompts that elicit a policy-violating response, judged by a BART-MNLI classifier | JailbreakBench (100 behaviors) | Jailbreak ASR |
| 🔜 | **Prompt Injection Success Rate** | Fraction of injected instructions that override the system prompt or change model behaviour | Custom · indirect RAG | Override rate |
| 📋 | **Paraphrase Consistency** | Does the model give the same answer to semantically equivalent rephrasings? | AdvGLUE · ANLI | Consistency rate |
| 📋 | **Counterfactual Fairness** | Does swapping a protected attribute (name, gender, race) change the model's output? | Custom templates | Flip rate by attribute |
| 📋 | **Factuality Robustness** | Does injecting a false premise into a QA question cause the model to accept it? | TriviaQA · NaturalQuestions | Fact-acceptance rate |
| 📋 | **Logical Negation Robustness** | Does the model correctly track negation under paraphrase? | MultiNLI · custom | Negation flip rate |
| 🔭 | **Multi-Turn Manipulation** | Can a model's behaviour be shifted over successive turns via context accumulation? | MT-Bench · custom | Behaviour drift score |
| 🔭 | **Confidence / Calibration Shift** | Does an adversarial attack inflate the model's confidence in a wrong answer? | SST-2 · AdvGLUE | ECE · confidence delta |
| 🔭 | **Tool Call Hijacking Rate** | Can adversarial input redirect an agent's function calls? | Custom agentic tasks | Hijack rate |
| 🔭 | **Training Data Extraction** | Can repeated prompting extract PII or verbatim training text? | Custom extraction prompts | Extraction rate |

---

## 📓 Demo Notebooks

| Notebook | Status | Description |
|---|:---:|---|
| [`01_adversarial_nlp_demo.ipynb`](notebooks/01_adversarial_nlp_demo.ipynb) | ✅ | 10 adversarial attacks × SST-2 — 5 perturbation levels, accuracy drop, composite stealth scoring, risk matrix, human review queue, executive report. [Results →](docs/01_adversarial_nlp.md) |
| [`02_jailbreaking_demo.ipynb`](notebooks/02_jailbreaking_demo.ipynb) | ✅ | Three-mode jailbreak evaluation — direct goals, artifact templates, PAIR transfer; BART-MNLI judge, incremental checkpointing, regulatory mapping. [Results →](docs/02_jailbreaking.md) |
| [`03_prompt_injection.ipynb`](notebooks/03_prompt_injection.ipynb) | 🔜 | Direct and indirect (RAG) prompt injection — override rate and payload taxonomy |
| [`04_fairness_counterfactual.ipynb`](notebooks/04_fairness_counterfactual.ipynb) | 📋 | Demographic swap testing — attribute-level flip rates and fairness metrics |
| [`05_nli_robustness.ipynb`](notebooks/05_nli_robustness.ipynb) | 📋 | Adversarial robustness on AdvGLUE + ANLI — cross-task coverage beyond sentiment |

Notebooks are intentionally **code-light** — they import from the modules above and focus on results, visualisations, and interpretation.

---

## 📚 References & Standards

### Frameworks & Standards
- [MITRE ATLAS](https://atlas.mitre.org/) — Adversarial Threat Landscape for AI Systems
- [NIST AI RMF (2023)](https://www.nist.gov/system/files/documents/2023/01/26/AI%20RMF%201.0.pdf) — AI Risk Management Framework
- [NIST AI 600-1 (2024)](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) — Generative AI Profile
- [OWASP LLM Top 10 (2025)](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — LLM application security risks
- [EU AI Act (2024/1689)](https://artificialintelligenceact.eu/) — High-risk AI system obligations

### Attacks & Benchmarks
- [TextFooler](https://arxiv.org/abs/1907.11932) — Jin et al., 2019
- [TextBugger](https://arxiv.org/abs/1812.05271) — Li et al., 2019
- [DeepWordBug](https://arxiv.org/abs/1801.04354) — Gao et al., 2018
- [BERT-Attack](https://arxiv.org/abs/2004.09984) — Li et al., 2020
- [PAIR](https://arxiv.org/abs/2310.08419) — Chao et al., 2023
- [GCG Universal Adversarial Attacks](https://arxiv.org/abs/2307.15043) — Zou et al., 2023
- [StrongREJECT](https://arxiv.org/abs/2402.10260) — Souly et al., 2024
- [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) · [AdvBench](https://github.com/llm-attacks/llm-attacks) · [HarmBench](https://github.com/centerforaisafety/HarmBench)

### Datasets
- [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) — Stanford Sentiment Treebank
- [AdvGLUE](https://arxiv.org/abs/2106.09680) — Adversarial GLUE benchmark
- [ANLI](https://arxiv.org/abs/1910.14599) — Adversarial NLI
- [ToxiGen](https://arxiv.org/abs/2203.09509) — Machine-generated toxic text
- [HateXplain](https://arxiv.org/abs/2012.10289) — Hate speech with rationales

### Related Tools
- [Microsoft PyRIT](https://github.com/Azure/PyRIT) — Python Risk Identification Tool for GenAI
- [NVIDIA Garak](https://github.com/NVIDIA/garak) — LLM vulnerability scanner
- [TextAttack](https://github.com/QData/TextAttack) — Adversarial NLP framework

---

## 🤝 Contributing

Contributions are welcome. To add a new attack, dataset, or testing strategy:

1. Fork the repo and create a feature branch
2. Follow the existing module structure — attacks inherit from the base class in `attacks/base.py`
3. Add an entry to the relevant roadmap table (with standards mapping)
4. Open a PR with a short description of the attack and at least one worked example

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## ⚠️ Responsible Use

This toolkit is intended for **security research, model evaluation, and AI safety work**. All jailbreak goals used in testing are sourced from published academic benchmarks. Do not use this toolkit to generate or distribute harmful content.

---

<div align="center">
  <sub>Built for AI safety practitioners, ML engineers, and red team researchers.</sub>
</div>
