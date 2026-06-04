<div align="center">

# 🔴 LLM Red Teaming

**A modular, extensible toolkit for adversarial testing of large language models and NLP systems.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](CONTRIBUTING.md)

*Covering adversarial text attacks · jailbreaking · prompt injection · automated judging · pluggable model targets*

</div>

---

## 📖 Overview

Modern AI systems are increasingly deployed in sensitive contexts — yet their robustness to adversarial inputs remains poorly understood. **LLM Red Teaming** provides a structured, reproducible framework to:

- **Attack** language models at multiple levels: character, word, sentence, and semantic
- **Jailbreak** instruction-tuned LLMs using standardized benchmarks (JailbreakBench)
- **Judge** model responses automatically using rule-based and zero-shot classification
- **Evaluate** robustness metrics: accuracy drop, attack success rate (ASR), refusal rate
- **Report** findings in clean, structured outputs

This repo grows in phases — from adversarial NLP benchmarks through prompt injection, bias testing, and defense mechanisms. Each module is independently usable or combinable into full evaluation pipelines.

---

## 🗂️ Repository Structure

```
llm_red_teaming/
│
├── attacks/                    # All attack implementations
│   ├── character/              # TextBugger, DeepWordBug
│   ├── word/                   # TextFooler, BERTAttack
│   ├── sentence/               # CheckList, StressTest
│   ├── semantic/               # SemanticAttack (WordNet synonym substitution)
│   └── jailbreak/              # JailbreakBench runner + artifact templates
│
├── judges/                     # Response evaluation
│   └── classifier_judge.py     # Rule-based + zero-shot BART-MNLI judge
│
├── targets/                    # Model connectors (pluggable)
│   └── azure_openai.py         # Azure OpenAI / GPT-4o target
│
├── evaluate/                   # Metrics and reporting
│   └── metrics.py              # ASR, accuracy drop, verdict summary
│
├── notebooks/                  # Lightweight demo notebooks
│   ├── 01_adversarial_nlp_demo.ipynb
│   └── 02_jailbreaking_demo.ipynb
│
├── configs/                    # Experiment configuration files
│   └── default_config.yaml
│
├── results/                    # Output files (gitignored)
├── .env.example                # API key template
├── requirements.txt
└── LICENSE
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/minw0607/llm_red_teaming.git
cd llm_red_teaming
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

### 3. Run a Demo Notebook

Open `notebooks/01_adversarial_nlp_demo.ipynb` or `notebooks/02_jailbreaking_demo.ipynb` in Jupyter.

### 4. Run an Attack Programmatically

```python
from attacks.character.textbugger import TextBugger
from attacks.word.textfooler import TextFooler
from evaluate.metrics import summarise_results

texts = ["The movie was absolutely wonderful.", "A complete waste of time."]

bugger  = TextBugger()
fooler  = TextFooler()

for text in texts:
    print(bugger.attack(text))
    print(fooler.attack(text))
```

---

## 🧩 Modules

### ⚔️ Attack Modules

| Module | Attack | Level | Method | Meaning Preserved |
|---|---|---|---|---|
| `attacks/character` | **TextBugger** | Character | Random char substitution | ❌ |
| `attacks/character` | **DeepWordBug** | Character | Insert / delete / swap | ❌ |
| `attacks/word` | **TextFooler** | Word | WordNet synonym swap | ~✅ |
| `attacks/word` | **BERTAttack** | Word | BERT fill-mask + cosine filter | ✅ |
| `attacks/sentence` | **CheckList** | Sentence | Append random noise token | ❌ |
| `attacks/sentence` | **StressTest** | Sentence | Append tautological text | ❌ |
| `attacks/semantic` | **SemanticAttack** | Semantic | POS-aware synonym substitution | ✅ |
| `attacks/jailbreak` | **JBB Runner** | Alignment | JailbreakBench goals + artifacts | N/A |

### 🧑‍⚖️ Judge Module

Automatic response classification using a two-stage pipeline:
1. **Rule-based** — fast regex for explicit refusals and known violation patterns
2. **Zero-shot classifier** — `facebook/bart-large-mnli` for nuanced cases

Verdicts: `violation` · `refusal` · `blocked` · `uncertain` · `benign`

### 🎯 Target Module

Pluggable model connectors. Currently supported:
- **Azure OpenAI** (GPT-4o, GPT-4, GPT-3.5)

Planned: OpenAI, Anthropic Claude, HuggingFace local models

### 📊 Evaluate Module

Standardised metrics computed per attack run:
- **Accuracy Drop** = accuracy(original) − accuracy(attacked)
- **Attack Success Rate (ASR)** = violations / total queries
- **Refusal Rate**, **Blocked Rate**, **Failure Rate**
- Per-category breakdown when using JailbreakBench

---

## 📓 Demo Notebooks

| Notebook | Description |
|---|---|
| [`01_adversarial_nlp_demo.ipynb`](notebooks/01_adversarial_nlp_demo.ipynb) | Run all 7 adversarial attacks on SST-2, compare accuracy drop per attack |
| [`02_jailbreaking_demo.ipynb`](notebooks/02_jailbreaking_demo.ipynb) | Test GPT-4o against JailbreakBench goals and PAIR artifacts, review ASR |

Notebooks are intentionally **code-light** — they import from the modules above and focus on results, visualisations, and observations.

---

## 🗺️ Roadmap

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | Adversarial NLP attacks + JailbreakBench evaluation | ✅ Complete |
| **Phase 2** | Prompt injection (direct + indirect / RAG), multi-model targets, YAML-driven runner | 🔜 Next |
| **Phase 3** | Bias & fairness testing (BBQ, WinoBias), automated HTML reports, config CLI | 📋 Planned |
| **Phase 4** | Defense module (adversarial training, input sanitization), CI regression tests | 📋 Planned |

---

## 📚 References & Datasets

- [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) — standardized LLM jailbreak benchmark
- [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) — Stanford Sentiment Treebank (GLUE)
- [TextFooler](https://arxiv.org/abs/1907.11932) — Jin et al., 2019
- [TextBugger](https://arxiv.org/abs/1812.05271) — Li et al., 2019
- [DeepWordBug](https://arxiv.org/abs/1801.04354) — Gao et al., 2018
- [BERT-Attack](https://arxiv.org/abs/2004.09984) — Li et al., 2020
- [Microsoft PyRIT](https://github.com/Azure/PyRIT) — inspiration for modular design
- [NVIDIA Garak](https://github.com/NVIDIA/garak) — LLM vulnerability scanner

---

## 🤝 Contributing

Contributions are welcome! To add a new attack, target, or judge:

1. Fork the repo and create a feature branch
2. Follow the existing module structure and base classes
3. Add a brief entry to the relevant `__init__.py`
4. Open a PR with a description of what you added and why

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## ⚠️ Responsible Use

This toolkit is intended for **security research, model evaluation, and AI safety work**. All jailbreak goals used in testing are sourced from published academic benchmarks. Do not use this toolkit to generate or distribute harmful content.

---

<div align="center">
  <sub>Built for AI safety practitioners, ML engineers, and red team researchers.</sub>
</div>
