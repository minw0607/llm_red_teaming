# Roadmap — Future Attacks, Datasets & Testing Strategies

[← Back to README](../README.md)

Detailed planning tables for what's next. Everything **implemented** is documented per-workstream — see the [workstream docs](../README.md#-workstreams) (each with full standards mapping). The tables below cover **future** work only.

Status legend: ✅ Implemented · 🛠️ Built (live run pending) · 🔜 Next milestone · 📋 Planned · 🔭 Research horizon

---

## Future Attack Library — Tier 2 (Medium Priority)

| Status | Attack | Category | Description | MITRE ATLAS | NIST AI 600-1 | OWASP LLM Top 10 |
|:---:|---|---|---|---|---|---|
| 📋 | **Paraphrase Model Attack** | Semantic | Use a paraphrase model (PEGASUS, DIPPER, T5) to generate fluent rewrites — more naturalistic than synonym lookups | AML.T0043 · AML.T0015 | Information Integrity | LLM09 Misinformation |
| ✅ | **Counterfactual / Demographic Swap** | Fairness | Swap protected attributes (gender, race, age, nationality, religion) and measure if the decision changes — flip rate + parity gap (NB04) | AML.T0043 *(weak fit — bias is a harm)* | **Harmful Bias and Homogenization (§2.8)** | LLM09 Misinformation |
| ✅ | **BBQ Stereotype Benchmark** | Fairness | 11-category bias benchmark; official ambiguous/disambiguated bias score (NB04) | — | **Harmful Bias and Homogenization (§2.8)** | — |
| 📋 | **Payload Splitting** | Prompt | Distribute a forbidden phrase across multiple tokens, turns, or encoded segments to evade safety filters | AML.T0054 | Information Security | LLM01 Prompt Injection · LLM07 System Prompt Leakage |
| 📋 | **Many-Shot Jailbreak** | Prompt | Provide many in-context examples of policy-violating exchanges before the target request to shift the model's behaviour (Anthropic, 2024) | AML.T0054 | Information Security (§2.6) | LLM01 Prompt Injection |
| 📋 | **Crescendo / Multi-Turn** | Prompt | Escalate harmful requests across multiple conversational turns, each individually benign (Microsoft, 2024) | AML.T0054 · AML.T0006 | Information Security (§2.6) | LLM01 Prompt Injection · LLM06 Excessive Agency |
| 📋 | **GCG / Adversarial Suffix** | Gradient | Greedy Coordinate Gradient (Zou et al. 2023) — finds a universal adversarial suffix that forces harmful outputs; transferable to black-box targets | AML.T0043 · AML.T0015 | Information Security (§2.6) | LLM01 Prompt Injection |
| 📋 | **Multilingual Bypass** | Structural | Submit harmful requests in low-resource languages where safety fine-tuning is weaker | AML.T0054 | Information Security (§2.6) · Harmful Bias (§2.8) | LLM01 Prompt Injection |
| 📋 | **HarmBench Expansion** | Dataset | Extend jailbreak coverage from 100 (JailbreakBench) to 400 behaviors across 7 harm categories with pre-published ASR baselines for model comparison | AML.T0054 | Information Security (§2.6) · CBRN (§2.1) | LLM01 Prompt Injection |
| 📋 | **StrongREJECT Scoring** | Metric | Replace binary ASR with StrongREJECT score (Souly et al. 2024) — penalises partial compliance; more accurate than BART-MNLI on ambiguous refusals | AML.T0054 | Information Security (§2.6) | LLM01 Prompt Injection |
| 📋 | **Llama Guard Judge** | Judge | Replace BART-MNLI with Meta's Llama Guard 3 (fine-tuned safety classifier) — lower false-positive rate on sensitive-topic academic responses | AML.T0054 | Information Security (§2.6) | LLM01 Prompt Injection |

## Future Attack Library — Tier 3 (Research Horizon)

| Status | Attack | Category | Description | MITRE ATLAS | NIST AI 600-1 | OWASP LLM Top 10 |
|:---:|---|---|---|---|---|---|
| 🔭 | **RAG / Vector Store Poisoning** | Infrastructure | Inject adversarial documents into a retrieval corpus to influence LLM responses via indirect context | AML.T0054 · AML.T0020 | Information Security · Value Chain | LLM08 Vector & Embedding Weaknesses |
| 🔭 | **Multi-Turn Context Manipulation** | Prompt | Build up a false context or persona over many dialogue turns to gradually shift model behaviour | AML.T0054 | Information Security · Human-AI Configuration | LLM01 Prompt Injection |
| 📋 | **Tool / Function Call Hijacking** | Prompt | Craft adversarial input that redirects an LLM agent's tool calls to unintended targets or actions *(NB07)* | AML.T0053 · AML.T0054 | Information Security | LLM06 Excessive Agency |
| 🔭 | **Backdoor / Trojan Trigger** | Model-level | Insert a hidden trigger phrase during fine-tuning that causes targeted misclassification at inference | AML.T0020 · AML.T0043 | Information Security · Data Provenance | LLM04 Data & Model Poisoning |
| 🔭 | **Membership Inference** | Privacy | Query the model systematically to determine whether specific examples were in its training data | AML.T0040 · AML.T0031 | Data Privacy | LLM02 Sensitive Information Disclosure |
| 🔭 | **Model Extraction / Stealing** | Privacy | Reconstruct a functional surrogate of the target model via repeated black-box queries | AML.T0031 · AML.T0040 | Value Chain and Component Integration | LLM02 Sensitive Information Disclosure |
| 🔜 | **Training Data Extraction** | Privacy | Prompt the model to reproduce memorised training data (PII, copyrighted text) *(NB06)* | AML.T0031 | Data Privacy · Intellectual Property | LLM02 Sensitive Information Disclosure |

---

## Dataset Roadmap

| Status | Dataset | Task | What it adds | Source |
|:---:|---|---|---|---|
| ✅ | **SST-2** | Binary sentiment | Baseline — fast to score, sensitive to lexical changes | [HuggingFace](https://huggingface.co/datasets/stanfordnlp/sst2) |
| ✅ | **JailbreakBench** | Safety / jailbreak | 100 harmful behaviors + PAIR/GCG artifact library (NB02) | [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) |
| ✅ | **HarmBench** | Safety / jailbreak | 400 behaviors across 7 harm categories; published ASR baselines (NB02) | [HarmBench](https://github.com/centerforaisafety/HarmBench) |
| ✅ | **deepset/prompt-injections** | Prompt injection | 203 real-world injection payloads for the LLM-judged track (NB03) | [HuggingFace](https://huggingface.co/datasets/deepset/prompt-injections) |
| ✅ | **BBQ** | Bias / fairness | Bias Benchmark for QA — 11 social categories, ambiguous/disambiguated bias score (NB04) | [Parrish et al. 2022](https://github.com/nyu-mll/BBQ) |
| ✅ | **MultiNLI** | 3-class entailment | Clean NLI baseline — the robustness-gap reference (NB05) | [HuggingFace](https://huggingface.co/datasets/nyu-mll/multi_nli) |
| ✅ | **ANLI** | 3-class entailment | Human-in-the-loop adversarial NLI, 3 difficulty rounds + `reason` annotations (NB05) | [Nie et al. 2020](https://arxiv.org/abs/1910.14599) |
| ✅ | **AdvGLUE** | 3-class entailment | Adversarially-perturbed MNLI; second adversarial track (NB05) | [Wang et al. 2021](https://arxiv.org/abs/2111.02840) |
| 📋 | **ToxiGen / HateXplain** | Toxicity classification | Safety-critical: does the model correctly flag hate speech after perturbation? | [ToxiGen](https://arxiv.org/abs/2203.09509) · [HateXplain](https://arxiv.org/abs/2012.10289) |
| 📋 | **TriviaQA** | Open-domain QA | Does a factual answer change when the question is rephrased? | [HuggingFace](https://huggingface.co/datasets/trivia_qa) |
| 🔭 | **MMLU** (select subsets) | Multi-domain MCQ | Domain-specific robustness (medical, legal, STEM) under prompt rephrasing | [HuggingFace](https://huggingface.co/datasets/cais/mmlu) |
| 🔭 | **MT-Bench** | Instruction following | Does multi-turn output quality degrade under adversarial system prompts? | [LMSYS](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) |

---

## Testing Strategy Roadmap

| Status | Strategy | What it measures | Primary dataset(s) | Key metric |
|:---:|---|---|---|---|
| ✅ | **Prediction Flip (ASR)** | Fraction of correct predictions overturned by an attack | SST-2 | Attack Success Rate |
| ✅ | **Risk Scoring** | Composite danger = Impact × Stealth; ranks attacks by operational priority | SST-2 | Risk Score |
| ✅ | **Human Review Queue** | Prioritises adversarial examples by flip + stealth for manual inspection | All | HIGH / MEDIUM / LOW |
| ✅ | **Composite Stealth Scoring** | Semantic similarity + perplexity ratio + normalised edit distance — richer imperceptibility signal than cosine similarity alone | All | Weighted composite |
| ✅ | **Jailbreak Success Rate (ASR)** | Fraction of jailbreak prompts that elicit a policy-violating response, judged by a BART-MNLI classifier | JailbreakBench (100 behaviors) | Jailbreak ASR |
| ✅ | **Prompt Injection Success Rate** | Fraction of injected instructions that override the system prompt or change model behaviour | Canary benchmark · `deepset/prompt-injections` | Override rate |
| 📋 | **Paraphrase Consistency** | Does the model give the same answer to semantically equivalent rephrasings? | AdvGLUE · ANLI | Consistency rate |
| ✅ | **Counterfactual Fairness** | Does swapping a protected attribute (gender, race, age, nationality, religion) change the decision? | Custom decision probes | Flip rate · parity gap |
| ✅ | **Stereotype Bias (BBQ)** | Does the model rely on social stereotypes when the answer is underdetermined? | BBQ (11 categories) | Bias score (−1…+1) |
| ✅ | **NLI Reasoning Robustness** | Does the model still infer entailment correctly under adversarial pressure? | MultiNLI · ANLI · AdvGLUE | Robustness gap (clean − adv) |
| 📋 | **Factuality Robustness** | Does injecting a false premise into a QA question cause the model to accept it? | TriviaQA · NaturalQuestions | Fact-acceptance rate |
| 📋 | **Logical Negation Robustness** | Does the model correctly track negation under paraphrase? | MultiNLI · custom | Negation flip rate |
| 🔭 | **Multi-Turn Manipulation** | Can a model's behaviour be shifted over successive turns via context accumulation? | MT-Bench · custom | Behaviour drift score |
| 🔭 | **Confidence / Calibration Shift** | Does an adversarial attack inflate the model's confidence in a wrong answer? | SST-2 · AdvGLUE | ECE · confidence delta |
| 📋 | **Tool Call Hijacking Rate** | Can adversarial input redirect an agent's function calls? *(NB07)* | AgentDojo-style tasks | Unsafe-action rate |
| 🔜 | **Training Data Extraction** | Can repeated prompting extract PII or verbatim training text? *(NB06)* | Canaries · synthetic PII · public-text prefixes | Regurgitation rate |

---

See also: [Industry alignment](industry_alignment.md) · [Dataset strategy](dataset_strategy.md)
