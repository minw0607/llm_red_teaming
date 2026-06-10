# Industry Alignment — Where This Toolkit Stands vs. the Field

[← Back to README](../README.md)

A candid assessment of how `llm_red_teaming` compares to the current state of LLM red teaming practice (as of mid-2026), and the prioritised upgrades that close the gaps. The goal is to be honest about what is solid, what is table-stakes-but-missing, and what is genuinely differentiated.

---

## Summary Scorecard

| Dimension | What we do now | Industry state of the art | Gap |
|---|---|---|---|
| **Adversarial NLP** | 10 attacks × 5 levels, composite stealth, risk scoring | TextAttack-class coverage; perturbation attacks largely "solved" on frontier models | ✅ On par — and correctly de-prioritised |
| **Jailbreak datasets** | JailbreakBench (100 behaviors) | HarmBench (400), AdvBench (520), in-the-wild corpora | 🟡 Add HarmBench for breadth + baselines |
| **Jailbreak attacks** | Direct, template wrap, PAIR transfer | + TAP, AutoDAN, GCG, many-shot, Crescendo (multi-turn) | 🟡 Missing automated & multi-turn methods |
| **Judge / scoring** | BART-MNLI zero-shot classifier | Llama Guard 3, HarmBench classifier, LLM-as-judge | 🔴 Weakest link — false positives |
| **Metric** | Binary ASR | StrongREJECT (graded), per-category ASR | 🟡 Binary ASR undercounts partial compliance |
| **Reporting** | LLM executive report, regulatory mapping | Per-category breakdowns, baseline comparison | ✅ Strong — regulatory mapping is ahead of most |

**One-line read:** the *framework* (pipeline, stealth scoring, regulatory mapping, executive reporting) is ahead of most open tooling. The *jailbreak depth* (judge quality, attack methods, dataset breadth) is where we trail the research frontier — and all three are addressable without architectural change.

---

## 1. Datasets

**Where we are:** [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) — 100 harmful behaviors with a bundled PAIR/GCG artifact library. Reproducible, well-cited, and the artifacts let us test transfer attacks for free.

**Where the field is:**
- **[HarmBench](https://github.com/centerforaisafety/HarmBench)** (Mazeika et al., 2024) — 400 behaviors across 7 semantic categories (cybercrime, bioweapons, harassment, etc.), evaluated against 18 red-teaming methods. This is the de facto comprehensive benchmark; it ships **published ASR baselines** so you can position a target against known models.
- **[AdvBench](https://github.com/llm-attacks/llm-attacks)** (Zou et al., 2023) — 520 harmful behaviors + 1,000 harmful strings; the original GCG evaluation set.
- **In-the-Wild Jailbreak Prompts** (Shen et al., 2023) — 1,405 real jailbreak prompts scraped from online communities; captures what attackers *actually* post, not just curated academic goals.

**Gap & plan:** JailbreakBench is a credible baseline but narrow at 100 behaviors. **Add HarmBench** as the next dataset — the breadth (400 behaviors) and published baselines turn a benchmark run into a comparison ("your model refuses X% vs. GPT-4's Y%"). See [dataset_strategy.md](dataset_strategy.md) for how generic benchmarks fit into a real engagement.

---

## 2. Judge / Scoring Model

**Where we are:** zero-shot **BART-MNLI** (`facebook/bart-large-mnli`) classifying responses into violation / refusal / blocked / uncertain / benign. Free, local, no API cost — but it was never fine-tuned for safety classification, and it produces **false positives** when a model discusses a sensitive topic academically without actually complying (we saw exactly this in NB02's single "violation," a historical account scored at the 0.60 threshold).

**Where the field is:**
- **[Llama Guard 3](https://huggingface.co/meta-llama/Llama-Guard-3-8B)** (Meta) — a classifier *fine-tuned specifically* for input/output safety classification against the MLCommons hazard taxonomy. Free, local, dramatically lower false-positive rate than a generic NLI model.
- **HarmBench classifier** — a fine-tuned judge released with HarmBench, validated against human labels.
- **LLM-as-judge** (GPT-4 / Claude grading responses) — highest accuracy, but adds API cost and introduces circularity if the judge shares a model family with the target.

**Gap & plan:** this is the **highest-leverage upgrade**. Swapping BART-MNLI → Llama Guard 3 is a drop-in judge replacement (`judges/` already abstracts the judge interface) and would immediately reduce the manual-review burden from false positives. LLM-as-judge can be offered as a higher-cost, higher-accuracy tier for engagements that need it.

---

## 3. Metric

**Where we are:** binary **Attack Success Rate** (`violations / total`). Simple, standard, and the right headline number.

**Where the field is:** **[StrongREJECT](https://github.com/alexandrasouly/strongreject)** (Souly et al., 2024) argued that binary ASR is misleading — many "successful" jailbreaks produce vague, hedged, or low-quality outputs that wouldn't actually help an attacker. StrongREJECT provides (a) a curated set of 313 forbidden prompts and (b) a **graded autograder** that scores responses on refusal, specificity, and convincingness — penalising partial compliance like *"I can't help with that, but here's roughly how it works…"* that binary ASR scores as either a clean pass or a clean fail.

**Gap & plan:** keep ASR as the headline, **add StrongREJECT-style graded scoring** as a secondary metric. Also add **per-category ASR breakdown** (cybercrime vs. harassment vs. CBRN) — a single aggregate ASR hides which harm classes a model is actually weak on, and per-category is now expected in serious evaluations.

---

## 4. Attack Methods

**Where we are:** three modes in NB02 — direct goals, hand-crafted template wrapping, and PAIR-artifact transfer.

**Where the field is** (automated and multi-turn methods we don't yet have):

| Method | What it does | Reference |
|---|---|---|
| **GCG** | Gradient-based universal adversarial suffix; transferable to black-box targets | Zou et al., 2023 |
| **PAIR** | Attacker LLM iteratively refines prompts until compliance | Chao et al., 2023 *(transfer-tested in NB02)* |
| **TAP** | Tree of Attacks with Pruning — more query-efficient than PAIR | Mehrotra et al., 2023 |
| **AutoDAN** | Genetic-algorithm-generated stealthy jailbreak prompts | Liu et al., 2023 |
| **Many-shot** | Floods the context with many fake policy-violating exchanges before the real ask | Anil et al. (Anthropic), 2024 |
| **Crescendo** | Multi-turn escalation — each turn individually benign, the trajectory harmful | Russinovich et al. (Microsoft), 2024 |

**Gap & plan:** GCG, many-shot, and Crescendo are already on the [Tier 2 roadmap](../README.md#future-attack-library--tier-2-medium-priority). **Crescendo / multi-turn is the most important strategic gap** — single-turn defences are now strong (NB02 confirms it), so realistic adversaries have moved to multi-turn. A toolkit that only tests single-turn will increasingly over-state a model's safety.

---

## 5. Reporting

**Where we are — and this is a strength:**
- **LLM-interpreted executive report** — deterministic metrics → plain-English risk verdict, severity-badged findings, prioritised recommendations (metrics are never LLM-generated).
- **Dynamic regulatory mapping** — finding-level citations to NIST AI 600-1, MITRE ATLAS, OWASP LLM Top 10, and EU AI Act, with actual metric values embedded.

This regulatory-mapping + executive-narrative layer is **ahead of most open-source red-teaming tools**, which typically stop at a raw ASR table. It's the part most aligned with an enterprise / advisory audience.

**Minor gaps:** add per-category ASR charts and baseline-comparison columns (vs. published HarmBench numbers) so a reader can see not just "X% ASR" but "X% vs. peer models."

---

## Prioritised Upgrade Path

1. **Llama Guard 3 judge** (highest leverage, drop-in) — fixes the false-positive problem that currently inflates manual review.
2. **StrongREJECT graded scoring + per-category ASR** — more honest metric, expected in serious evals.
3. **HarmBench dataset** — breadth (400 behaviors) and published baselines for peer comparison.
4. **Crescendo / multi-turn attack** — closes the single-turn-only blind spot as the field moves multi-turn.
5. **TAP / AutoDAN** — automated attack generation beyond hand-crafted templates.

None of these require architectural change — the `targets/`, `judges/`, and `evaluate/` abstractions already accommodate them. They are additive.

---

## References

- HarmBench — [Mazeika et al., 2024](https://arxiv.org/abs/2402.04249)
- AdvBench / GCG — [Zou et al., 2023](https://arxiv.org/abs/2307.15043)
- PAIR — [Chao et al., 2023](https://arxiv.org/abs/2310.08419)
- TAP — [Mehrotra et al., 2023](https://arxiv.org/abs/2312.02119)
- AutoDAN — [Liu et al., 2023](https://arxiv.org/abs/2310.04451)
- Many-shot Jailbreaking — [Anil et al. (Anthropic), 2024](https://www.anthropic.com/research/many-shot-jailbreaking)
- Crescendo — [Russinovich et al. (Microsoft), 2024](https://arxiv.org/abs/2404.01833)
- StrongREJECT — [Souly et al., 2024](https://arxiv.org/abs/2402.10260)
- In-the-Wild Jailbreak Prompts — [Shen et al., 2023](https://arxiv.org/abs/2308.03825)
- Llama Guard — [Inan et al. (Meta), 2023](https://arxiv.org/abs/2312.06674)
