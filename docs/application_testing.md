# Model-Level vs. Application-Level Testing

[← Back to README](../README.md)

The single most important scoping question for any LLM red-team: **are you testing the bare model, or the deployed application?** They answer different questions, and conflating them is how a report over-claims.

## The two scopes

| | Model-level (default) | Application-level |
|---|---|---|
| **What's in the loop** | the model endpoint + the vendor's platform filter (e.g. Azure Content Safety) | the **app's** system prompt, input/output **guardrails**, retrieval, tools, auth |
| **Target** | `OpenAICompatibleTarget` / `AzureOpenAITarget` | `ApplicationTarget` (points at the app's API) |
| **Question answered** | "Is the model we build on intrinsically sound?" | "Is *this deployment* safe as shipped?" |
| **Properties** | reproducible · vendor-comparable · a conservative **upper bound** on risk | engagement-specific · reflects real defences |
| **Analogy** | a component spec sheet | a penetration test |

Both are standard and legitimate. NB01–07 run **model-level by default** — the runners supply their own minimal system prompt, so they probe intrinsic model behaviour (plus whatever the platform filter blocks). That is the right unit for model assurance, but **it is not a test of your application**, which typically wraps the model in defence-in-depth that would change the results (usually making the app *safer* than the bare model).

## Making it application-level

Point the target at the application instead of the model — the *same* probes then run with the app's guardrails in place:

```python
from targets import OpenAICompatibleTarget, ApplicationTarget

model = OpenAICompatibleTarget()   # bare model
app   = ApplicationTarget()        # deployed app (env: APP_BASE_URL / APP_API_KEY / APP_MODEL)
```

`ApplicationTarget` sends **user input only** — it drops the runner's injected system prompt so the application's own system prompt and guardrails govern the response. Any runner can take it:

```python
results_model = ExfiltrationRunner(model).run()
results_app   = ExfiltrationRunner(app).run()
```

**Scope caveat.** Tracks that *plant a secret in the system prompt* — NB06 Track A (system-prompt disclosure) — are model-level by construction (you control the planted canary). Against a real app you instead attack the app's *own* system prompt, so that track's canary metric does not transfer. Injection, jailbreak, memorization, exfiltration, NLI, and agent probes all deliver via **user input** and transfer directly to `ApplicationTarget`. For RAG and agent workstreams, full fidelity means wiring the runners to the *real* retriever/tools (the NB06 🅲 mock and NB07 sandbox are methodology demonstrators).

## Measuring the guardrails — the delta

The client-relevant number isn't "the model leaks X%" — it's **"your guardrails catch Y% of what the model would leak."** Run the same probes against both scopes and compute the delta:

```python
from evaluate import print_guardrail_report

# success predicate depends on the workstream:
#   data:   lambda r: r['leaked'] and r['leak_type'] != 'verbatim'
#   agent:  'unsafe_action'
#   inject: 'injected'
print_guardrail_report(results_model, results_app,
                       success=lambda r: r['leaked'] and r['leak_type'] != 'verbatim',
                       by='track')
```

Output reports, overall and per category:
- **model-level attack-success rate** (the upper bound),
- **application attack-success rate** (the *residual* — what still gets through),
- **blocked %** = the fraction of model-level successes the guardrails stopped.

A high blocked-% with low residual is a strong, defensible application result; a low residual that is still **non-zero** is exactly the finding a client needs (the guardrail gap).

## How to report it

State the scope explicitly in every deliverable. "0 sensitive leaks (model-level, GPT-5-4 via Azure)" is an honest model-assurance claim; "0 sensitive leaks (application-level, app endpoint with guardrails)" is a much stronger — and different — claim. Never let a model-level result be read as an application guarantee.

See also: [industry_alignment.md](industry_alignment.md) · [dataset_strategy.md](dataset_strategy.md)
