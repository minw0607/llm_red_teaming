# RAG Data Leakage — Design & Methodology (Notebook 06b)

[← Back to README](../README.md) · [Open notebook](../notebooks/06b_rag_data_leakage.ipynb)

**The use-case half of the data red-teaming pair.** [NB06](06_data_redteam.md) is the *benchmark* — does a model disclose secrets placed in its prompt. This is the deployment-shaped audit: does a *retrieval system* hand a user documents they are not entitled to.

> **Status:** 🛠️ Built and ground-truth validated. The deterministic retrieval layer has been run in full; the model-driven tracks are run pending.

---

## Why this pair exists

A bare model has no documents to leak. **Retrieval creates the entire attack surface**, which makes this the cleanest case in the toolkit for why use-case testing is not optional: no amount of model-level benchmarking can reach the risk, because the risk is not in the model.

| | NB06 (benchmark) | NB06b (use case) |
|---|---|---|
| Target | the model | the **retrieval pipeline** + the model |
| Context | documents hand-placed in the prompt | a **real index** decides what enters context |
| Corpus | synthetic fixtures | 600 real Enron documents + clearance overlay |
| Ground truth | canary present / absent | **entitlement**, known per (role, document) |
| Variable under test | the prompt | the **architecture** |

---

## The two failures, measured separately

| Failure | Where | Needs a model? |
|---|---|---|
| **Retrieval failure** — an unentitled document reaches the context | the index | ❌ |
| **Disclosure failure** — the assistant repeats what it was given | the model | ✅ |

The order matters. The deterministic layer runs first, costs nothing, and answers the architectural question outright. The second only applies to content that survived the first. Collapsing them into one "did it leak" number would be useless to whoever has to fix it, because the remediation is entirely different.

---

## The corpus

**Real documents, invented clearances.** 600 Enron emails (FERC-released; the corpus used by LLM-PBE, VLDB 2024 and DecodingTrust, NeurIPS 2023), each assigned one of four tiers — `PUBLIC` / `INTERNAL` / `CONFIDENTIAL` / `RESTRICTED` — read by four roles.

Text has to be real, because retrieval behaviour on synthetic prose is not retrieval behaviour on corporate email. Labels cannot be real: no public corpus ships clearance metadata, and the corpora carrying genuine sensitivity labels (MIMIC, i2b2) are credentialed and non-redistributable.

**Entitlement is therefore known for every (role, document) pair, which is what makes a leak a fact rather than a judgement.** Every `CONFIDENTIAL` and `RESTRICTED` document carries a unique planted marker; if it appears in an answer to someone without clearance, protected content was reproduced. This plays the same role matched-pair résumés play in [NB04b](04b_hiring_fairness_audit.md).

Two deliberate inclusions:

- **Benign documents that merely look sensitive** — without them, an assistant that refuses everything scores a perfect leak rate while being worthless.
- **Poisoned documents** — the attacker is anyone who can add a file to the knowledge base.

**On tier assignment:** ~9% of documents announce their own sensitivity and are labelled from that signal; the rest get a balanced random tier. That is deliberate. If `RESTRICTED` were a synonym for "legal", a per-tier leak rate would partly measure how often queries happen to concern legal matters. Random assignment decorrelates label from topic, isolating access-control enforcement.

---

## The experiment — three architectures

Everything held constant except how access control is wired into retrieval.

| | Behaviour | Role |
|---|---|---|
| 🔴 `no_filter` | top-k, clearance ignored | naive build; **broken-pipeline control** |
| 🟠 `post_filter` | retrieve top-k, then drop what the user may not read | the **common** build |
| 🟢 `pre_filter` | restrict candidates before searching | correct build; **zero-leak control** |

### Results — retrieval layer (deterministic, zero model calls)

600 documents · 96 probes · k=5

![RAG access-control audit](images/nb06b_rag_leakage.png)

| Architecture | Leak rate | 95% CI | Usable context | vs correct build |
|---|---:|---|---:|---|
| `no_filter` | **73.96%** | 0.644 – 0.817 | 5.00 / 5 | p < 1e-6, significant (Holm) |
| `post_filter` | **0%** | 0 – 0.039 | **2.99 / 5** | n.s. |
| `pre_filter` | **0%** | 0 – 0.039 | 5.00 / 5 | baseline |

**The headline is that the model never changed.** Same corpus, same questions, same weights — the architecture alone decides whether a user receives documents they may not read.

**The finding worth carrying:** `post_filter` leaks no content but **discards 40% of the retrieved context**. Restricted documents occupy top-k slots and are then thrown away, so the user silently receives three documents where they asked for five. Answer quality degrades, and it presents as a model defect. No leak metric would ever surface it.

---

## Validation gates

Every gate has a known-correct answer. A failure means the measurement is broken, not that the system is safe.

| Gate | Expected | Actual |
|---|---|---|
| `pre_filter` leaks nothing | 0% | ✅ 0% |
| `no_filter` leaks substantially | > 25% | ✅ 74% |
| Probes reach their target | > 50% | ✅ 79% / 71% |
| Poisoned documents get retrieved | > 50% | ✅ 56% |

The reachability gates exist because of a real failure in [NB07](07_agentic_tool_attacks.md), where attacks scored "resisted" turned out never to have been delivered. **An undelivered attack is not a defended one.**

Both reachability gates failed during development and both failures were genuine bugs, invisible in the leak rate itself:

1. Documents exceeded the encoder's 256-token window, leaving an invisible tail — text present in the context the model reads but absent from the vector it is retrieved by.
2. Keyterm selection took the first qualifying words, which in corporate email is routing chrome.

Target reachability went **31% → 48% → 75%** across the fixes. Similarly, generic poisoned documents reached only 19% of queries; seeding them with the language of the query they intercept lifted reach to 56% — which is also the realistic attack, since a poisoner writes to *rank*.

---

## Leakage is always reported with utility

An assistant that refuses every question has a 0% leak rate. The headline is therefore always a pair — leak rate and **utility retention** (legitimate questions still answered). A system at 0% leakage and 20% utility has not solved the problem; it has removed the product. Over-refusal on the `decoy` family (sensitive-*sounding* but permitted) is a false positive, not safety.

Ground-truth validated against three mock assistants:

| Mock | Leak (`no_filter`) | Leak (`pre_filter`) | Utility |
|---|---:|---:|---:|
| repeats its whole context | 1.00 | 0.00 | 0.88 |
| answers without quoting | 0.00 | 0.00 | 1.00 |
| refuses everything | 0.00 | 0.00 | **0.00** ← caught |

---

## Corpus poisoning — and why access control does not help

The poison track runs against **`pre_filter`, the correctly built pipeline**. The poisoned document sits at a tier every user may read, so access control offers no protection whatsoever. These are orthogonal defences, and a team that has done the access-control work properly may reasonably believe they are covered. They are not.

**Reach is reported separately from success, and success is conditional on reach.** Validated: an obedient mock scores 1.0 given reach; one that ignores document instructions scores 0.0.

---

## Regulatory mapping

| Framework | Requirement | What this produces |
|---|---|---|
| **GDPR** Art. 5(1)(f), 32 | integrity, confidentiality, security appropriate to risk | measured unauthorised-disclosure rate per architecture |
| **GDPR** Art. 25 | data protection **by design** | the architecture comparison *is* a by-design finding |
| **OWASP LLM08** | Vector & Embedding Weaknesses — retrieval access control, corpus poisoning | Tracks 1 and 2 |
| **OWASP LLM02 / LLM01** | sensitive-information disclosure; indirect injection | disclosure track; poison track |
| **ISO/IEC 27001** A.9 · **SOC 2** CC6 | logical access control | per-role entitlement enforcement, measured |
| **EU AI Act** Art. 10 | data governance | corpus composition and access provenance |
| **NIST AI 600-1** §2.4, §2.9 | Data Privacy, Information Security | the risk taxonomy this sits under |

Sector overlays (HIPAA, GLBA, SEC Reg S-P) apply where the corpus warrants; this one is neither clinical nor financial, so they are noted rather than claimed.

> **The governance point.** A finding here is usually an *architecture* finding with a known owner and a known fix. "Our retriever does not enforce clearance" is a defect an engineering team can close this week — far more actionable than a probability that a model misbehaves.

---

## Module layout

```
attacks/rag/
  corpus.py      # real documents + clearance overlay + canaries; entitlement ground truth
  index.py       # MiniLM + numpy cosine; the three architectures; retrieval_leak_check
  probes.py      # queries derived FROM their targets (IDF terms + body excerpt)
  assistant.py   # retrieve → answer; clearance-aware system prompt; canary scoring
  runner.py      # boundary + poison tracks, resume-safe version-gated checkpoints
evaluate/
  rag_metrics.py # architecture comparison, Holm significance, reachability,
                 # utility retention, poison reach-vs-success, detection floor
notebooks/06b_rag_data_leakage.ipynb   # Parts 1–5, Steps 0–9
```

---

## Limitations, stated plainly

- **Clearance tiers are synthetic.** Real documents, invented labels. On an engagement they come from the client's own classification scheme; getting them wrong means measuring the overlay.
- **Enron is 2001-era email.** Text distribution differs from modern corporate documents.
- **A numpy cosine index is not a production vector store.** Architectural findings transfer; specific rates do not. Azure AI Search, Pinecone and pgvector each apply their own filtering semantics.
- **The detection floor bounds any clean result.** At n=96, leak rates below ~6% would not have been visible.

---

## Next steps

- Run the model-driven tracks (Steps 5–6) against the assessed model.
- **Aggregation inference** — a restricted fact split across individually innocuous documents, so the violation exists only in the synthesis. Designed, not yet built.
- **Groundedness** — reuse the NLI machinery from [NB05](05_nli_robustness.md) to check the answer is entailed by retrieved context, catching fabricated citations.
