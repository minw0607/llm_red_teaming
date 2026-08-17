# Agentic Hiring Fairness Audit — Design & Methodology (Notebook 08)

[← Back to README](../README.md) · [Open notebook](../notebooks/08_agentic_hiring_fairness.ipynb)

Use-case-specific fairness testing of an **AI recruiting agent** — the closest thing in this toolkit to a real regulatory bias audit.

> **Status:** harness, corpus, metrics, and executive report are **built and validated end-to-end** (including against synthetic ground-truth data with known bias). Live results are published here after a run against the assessed model.

---

## Why this exists — the gap in NB04

[NB04](04_fairness.md) asks a foundation model about **one candidate at a time** and measures whether the answer flips. That is a legitimate model-level signal, but it is not what a bias audit measures.

A real automated employment decision tool (AEDT) screens a **pool** and advances a shortlist. [NYC Local Law 144](https://www.nycbiasaudit.com/blog/how-to-comply-with-the-nyc-bias-audit-law) requires an annual independent bias audit reporting, per demographic group, the **selection rate** and the **impact ratio** (group rate ÷ highest group's rate), with **< 0.80** indicating adverse impact under the EEOC four-fifths rule.

**NB04's design structurally cannot produce that number** — isolated binary decisions have no pool, so no selection rate. NB08 does.

| | NB04 | NB08 |
|---|---|---|
| Target | foundation LLM | **tool-using agent** in a mock ATS |
| Unit of analysis | one candidate, isolated | **pool of 40 → shortlist top-N** |
| Data | generic templated prompts | **qualification-matched résumé corpus** |
| Headline metric | flip rate · parity gap | **LL144 impact ratio + significance test** |
| Extra surfaces | — | retrieval rank · triage attention · multi-turn drift |

Both are kept: NB04 remains the fast generic model-level baseline; NB08 is the deployment-shaped audit.

---

## The corpus — matched pairs

Every qualification profile is instantiated **once per demographic group with identical credentials**, varying only the name. Defaults: 5 profiles × 4 race groups × 2 genders = **40 candidates** across three tiers (strong / medium / weak).

Because matched candidates are equivalent by construction, any selection disparity is **causal** — it cannot be attributed to merit. This is the LLM analogue of the correspondence-audit method (Bertrand & Mullainathan, 2004), as applied to résumé screening by [Wilson & Caliskan (AIES 2024)](https://arxiv.org/abs/2407.20371) and [FAIRE (2025)](https://arxiv.org/pdf/2504.01420).

**Trade-off, stated plainly:** synthetic résumés buy internal validity (clean causal inference, which real corpora cannot give) at the cost of external validity (they are more uniform than real résumés). Name proxies for race/gender are established but imperfect; multiple names per cell guard against single-name artefacts.

---

## The target — an agent

The agent works inside a mock applicant tracking system (`attacks/hiring/sandbox.py`) with real tool calls, all logged:

```
list_candidates → read_resume → score_candidate → advance_candidate   ← the measured decision
```

Nothing touches the real world; "advancing" appends to an in-memory log. That log exposes surfaces a single-prompt test cannot see:

| Surface | Metric | What it catches |
|---|---|---|
| **Allocation** | selection rate → **impact ratio** | the LL144 core |
| **Triage attention** | `triage_rates` | whose résumé the agent even opened |
| **Retrieval rank** | `rank_disparity` | name-driven ranking *before the LLM reasons* — replicates Wilson & Caliskan on identical résumés |
| **Multi-turn drift** | `drift_by_batch` | bias accumulating across rounds ([FairMT-Bench, ICLR 2025](https://arxiv.org/abs/2410.19317)) |

The agent loop reuses the hardened ReAct parsing from [NB07](07_agentic_tool_attacks.md) (balanced-JSON action parsing, format re-prompting, content-filter detection).

---

## Three controls that make the number trustworthy

**1 · Position control.** The roster is re-shuffled every repeat. Without this, a screener that works down the list and stops at top-N makes list position a perfect confound with demographics — during development this produced a *fully spurious* adverse-impact finding from a deliberately fair screener. `position_check()` reports the residual spread.

**2 · Validity check.** `tier_alignment()` confirms selection tracks qualification tier. If the agent advances weak candidates as often as strong ones it isn't screening, and the fairness metrics are not interpretable — the executive report says so rather than reporting a clean pass.

**3 · Significance testing.** A disparity is reported as **confirmed** only when it fails four-fifths **and** reaches significance on a **Fisher exact test** (implemented directly, no SciPy) with **Holm–Bonferroni** correction. With 8 intersectional groups that is 7 simultaneous tests; uncorrected, roughly 30% of perfectly fair runs would flag something.

---

## Statistical power — the guard that matters most

`audit_confidence()` reports a **minimum detectable ratio**: the subtlest disparity the run could confirm. **If that number is above 0.80, the audit cannot see a four-fifths violation**, and a clean result is a statement about sample size, not fairness.

Measured against synthetic ground-truth data (intersectional grouping, 8 groups — always the noisiest):

| Candidates per group | Can confirm down to | True-positive rate on a real 2× disparity |
|---|---|---|
| ~30 | IR ≤ 0.36 | severe bias only |
| ~100 | IR ≤ 0.59 | 88% |
| ~200 | IR ≤ 0.72 | 100% |
| ~400 | IR ≤ 0.79 | 100% |

Aggregating by **race** (4 groups) or **sex** (2 groups) is materially better powered than intersectional. Raise `REPEATS` to buy power.

### Known limitation, inherent to the regulation

LL144 defines the impact ratio against the **highest-scoring group**. Comparing every group to the observed maximum is a winner's-curse comparison: under a perfectly fair process the top group is high partly by luck, so others look depressed. Against synthetic *fair* data this inflates the false-positive rate to roughly **12–18%** even after Holm correction. That is a property of the mandated metric, not of this implementation — treat a single flagged cell as a prompt to investigate, never as proof of discrimination.

---

## Module layout

```
attacks/hiring/
  corpus.py    # matched-pair résumé corpus, name banks, reshuffle_pool (position control)
  sandbox.py   # mock ATS + tool log → per-candidate audit rows
  rankers.py   # optional embedding ranker for the retrieval-bias track
  runner.py    # HiringAuditRunner — allocation / retrieval / multi-turn tracks
evaluate/
  hiring_metrics.py    # selection & scoring rates, impact ratio, Wilson CI, Fisher+Holm,
                       # power analysis, position/validity checks, drift
  hiring_executive.py  # executive HTML report (leads with power + confirmed-vs-noise)
notebooks/08_agentic_hiring_fairness.ipynb   # Steps 0–9
```

---

## Regulatory mapping

The rules do not all ask for the same thing, so precision matters. See Part 4 of the notebook
for the full treatment.

### Direct obligations

| Rule | In force | Requirement | Coverage here |
|---|---|---|---|
| **NYC Local Law 144** | Jul 2023 | annual independent bias audit; publish selection rates + impact ratios by sex, race, intersectional | ✅ exactly the metric computed. *Gap:* expects real historical applicants; we use synthetic matched pairs |
| **Illinois HB 3773** | 1 Jan 2026 | prohibits AI with a **discriminatory effect** (strict liability, intent irrelevant); bans ZIP as proxy; notice duty | ✅ effect-testing is the notebook's purpose. *Gap:* no ZIP-proxy test; notice is a process control |
| **California FEHA ADS** | 1 Oct 2025 | covers tools that screen, score, **rank** or recommend even with a human in the loop; testing before/after; **4-year retention** | ✅ the "rank" wording is why the retrieval track matters; Step 9 writes the retention artefacts |
| **EU AI Act** | high-risk duties from Aug 2026 | employment AI is Annex III high-risk — bias testing, documentation, logging, oversight | ✅ partially: testing + documentation; logging/oversight are deployment controls |
| **EEOC / Title VII** | long-standing | disparate impact; four-fifths rule | ✅ the 0.80 threshold plus significance testing |

### Relevant but different

| Rule | Status | Why it is not a direct fit |
|---|---|---|
| **Colorado SB 26-189** | eff. 1 Jan 2027 | repealed/replaced the earlier AI Act and **removed the mandatory bias audit**, moving to transparency, notice and human review. Hiring remains a "consequential decision", so testing is good practice but no longer commanded |
| **Texas TRAIGA** | 1 Jan 2026 | **disparate impact alone does not establish a violation** — intent is required. A tool could fail this audit and still comply |

### Voluntary frameworks

**NIST AI RMF / AI 600-1 §2.8** (Harmful Bias and Homogenization) — the risk taxonomy this sits
under. **ISO/IEC 42001** — certification audits look for documented, repeatable bias evaluation,
which the Step 9 artefacts provide.

### Deliberately not cited

**OWASP LLM Top 10** and **MITRE ATLAS** are security frameworks with no fairness or bias
category. OWASP is the right frame for NB02/03/06/07; ATLAS catalogues adversarial attacks, and
bias is a harm the system produces without an attacker. Forcing either mapping would
misrepresent both the finding and the framework.

> Only NYC LL144 prescribes the exact statistic computed here. Illinois and California make the
> underlying question legally consequential without prescribing a method. None of this is legal
> advice; a compliance audit needs the employer's own data and a qualified independent auditor.

---

## Related work leveraged

- **[LangFair](https://github.com/cvs-health/langfair)** (CVS Health) — its core thesis, that fairness must be assessed *per use case* rather than read off generic benchmarks ("bring your own prompts"), is the framing behind this notebook. Metrics are implemented natively here to keep the repo self-contained; LangFair's classification and recommendation fairness metrics are the closest external analogue.
- **Wilson & Caliskan (AIES 2024)** — retrieval-stage bias; the basis for the retrieval track.
- **FairMT-Bench (ICLR 2025)** — multi-turn bias accumulation; the basis for the drift track.
- **FAIRE (2025)** and job-résumé-matching work — résumé-bias evaluation design.

> **A note on expected findings.** Recent work finds the pro-White callback gap reproduces in 2023-vintage models, while models released 2024+ show null or even reversed gaps. This audit is deliberately built so that **"no disparity detected" is a first-class result** — with the power analysis making clear whether that reflects fairness or merely a small sample.

---

## Next Steps

Real-résumé corpus as an external-validity cross-check · intersectional power via targeted oversampling · loan/credit underwriting as a second use case (ECOA/Reg B) · pairing with [`ApplicationTarget`](application_testing.md) to audit a deployed screener with its guardrails in place.
