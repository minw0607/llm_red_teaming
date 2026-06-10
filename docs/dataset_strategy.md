# Dataset Strategy — Choosing Test Sets for Real Engagements

[← Back to README](../README.md)

Generic jailbreak benchmarks (JailbreakBench, HarmBench) are necessary but **not sufficient** for a real red-teaming engagement. This doc explains why, and lays out a layered approach for selecting test data that actually reflects a client's risk.

---

## Why Generic Benchmarks Aren't Enough

A standard benchmark like JailbreakBench gives you:
- ✅ **Reproducibility** — anyone can rerun and compare
- ✅ **Benchmark comparison** — position a target against published numbers
- ✅ **Breadth across harm categories** — a credible baseline sweep

But it tells you very little about:
- ❌ **Deployment-specific risk** — a coding assistant, a customer-service bot, and a medical-triage tool have completely different threat surfaces, yet a generic benchmark tests all three identically
- ❌ **Realistic adversary behaviour** — curated academic goals ("write a defamatory article") rarely match how a real attacker probes *your specific system*
- ❌ **Business-context harms** — MNPI leakage, contract manipulation, or fabricated financial advice never appear in a generic harm taxonomy

> The benchmark establishes the floor. The engagement-specific layers find the risks that actually matter to the client — and that's where the value (and the differentiation) lives.

---

## The Layered Model

Think of test-set selection as six layers, from generic to bespoke. A serious engagement uses several; a mature one uses all.

```
        ┌─────────────────────────────────────────────┐
  most  │ 6. Custom red-team generated  ← differentiator│
bespoke │ 5. Regulatory-driven                          │
   ▲    │ 4. Industry-vertical                          │
   │    │ 3. Threat-model-driven                        │
   ▼    │ 2. Use-case specific                          │
generic │ 1. Generic benchmark          ← table stakes  │
        └─────────────────────────────────────────────┘
```

### 1. Generic benchmark — *table stakes*
JailbreakBench / HarmBench / AdvBench. Establishes a reproducible baseline and lets you say "your model refuses X% where GPT-4 refuses Y%." Always start here; never stop here.

### 2. Use-case specific
Match the test set to what the model is *deployed to do*. The deployment context defines which behaviours are even relevant:
- A **coding assistant** → malware generation, vulnerability exploitation, license-laundering
- A **customer-service bot** → social-engineering, account-takeover assistance, policy-override
- A **document/RAG assistant** → indirect prompt injection via retrieved content, data exfiltration

### 3. Threat-model-driven
Define *who* the adversary is and *what they want*. The same model faces different threats depending on exposure:
- **Insider** probing an internal HR or finance chatbot → privilege escalation, data fishing
- **External attacker** on a public API → resource abuse, content-policy bypass at scale
- **Curious end-user** → casual jailbreaks, edge-case content

Goals should match realistic adversary *motivation*, not just abstract harm categories.

### 4. Industry-vertical
Layer in sector-specific harms with real regulatory and reputational weight:

| Vertical | High-priority harm classes |
|---|---|
| **Financial services** | Fraud facilitation, market manipulation, MNPI / insider-info leakage, unlicensed financial advice |
| **Healthcare** | Drug-synthesis guidance, self-harm / pro-ana content, unqualified medical advice, PHI disclosure |
| **Legal** | Privileged-information extraction, contract manipulation, fabricated citations |
| **Public sector** | Disinformation generation, surveillance enablement, discriminatory decisioning |

### 5. Regulatory-driven
For high-risk AI systems, regulators effectively hand you the test cases:
- **EU AI Act Annex III** — enumerates high-risk use cases; prohibited practices (Art. 5) define hard red lines
- **NIST AI 600-1** — specific GenAI risk categories (CBRN §2.1, harmful bias §2.8, info security §2.6) map directly to test buckets
- Sector regulators (FINRA, HIPAA, etc.) add domain obligations

This layer turns compliance requirements into a concrete, defensible test matrix.

### 6. Custom red-team generated — *the differentiator*
Human red-teamers probing the *specific* model and deployment will find what benchmarks miss: prompts tuned to the system prompt, the tools it has, the data it can reach, the persona it adopts. This is the highest-value layer and the hardest to commoditise — it's where a consulting engagement earns margin that open-source tooling cannot replicate.

---

## How to Choose — A Decision Flow

```
1. What is the model deployed to do?        → fixes Layer 2 (use case)
2. Who can reach it, and what do they want?  → fixes Layer 3 (threat model)
3. What sector / data does it touch?         → fixes Layer 4 (vertical)
4. Is it a regulated / high-risk system?     → fixes Layer 5 (regulatory)
5. Always: run a generic benchmark first     → Layer 1 baseline
6. Budget permitting: custom red-team pass   → Layer 6 differentiation
```

A lightweight engagement might be **Layers 1 + 2 + 6**. A regulated-industry engagement should cover **1–6**.

---

## Mapping It Back to This Toolkit

| Layer | Status in `llm_red_teaming` |
|---|---|
| 1. Generic benchmark | ✅ JailbreakBench (NB02); HarmBench planned |
| 2. Use-case specific | 🔧 Supported via custom goal lists in the runners (`ArtifactRunner` accepts custom templates) |
| 3. Threat-model-driven | 🔧 Author goal sets per threat model; same runner pipeline |
| 4. Industry-vertical | 📋 Curated vertical goal banks — planned |
| 5. Regulatory-driven | ✅ Mapping exists ([regulatory.py](../evaluate/regulatory.py)); goal banks per framework planned |
| 6. Custom red-team | 🔧 Manual — the runners accept arbitrary prompts; this is human-driven by design |

The runner architecture (`JailbreakBenchRunner`, `ArtifactRunner`) already accepts arbitrary goals and templates, so layers 2–6 are a matter of **authoring the right test data**, not building new infrastructure. The roadmap items are curated goal banks, not engine changes.

---

## Practical Engagement Workflow

1. **Scope** — identify use case, threat model, vertical, and regulatory status (Layers 2–5)
2. **Baseline** — run the generic benchmark (Layer 1) for a reproducible starting point and peer comparison
3. **Targeted suite** — assemble use-case + vertical + regulatory goal sets into custom runs
4. **Custom pass** — human red-teamers probe the specific deployment (Layer 6)
5. **Report** — feed all results through the executive-report + regulatory-mapping pipeline; break ASR down per harm category
6. **Re-test** — after mitigations, rerun the same suites (checkpointing makes incremental re-runs cheap)

---

## See Also
- [industry_alignment.md](industry_alignment.md) — where the toolkit's methods stand vs. the field
- [02_jailbreaking.md](02_jailbreaking.md) — the generic-benchmark results that form Layer 1
- [regulatory.py](../evaluate/regulatory.py) — the dynamic regulatory mapping behind Layer 5
