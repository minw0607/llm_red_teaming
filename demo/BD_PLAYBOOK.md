# Selling this capability

How to turn the toolkit into work. The demo earns the meeting; this is the rest of the motion.

```bash
python demo/serve.py
```

Opens a browser at localhost. Takes ~20 seconds to build the index, then runs entirely on
your laptop — the retrieval needs no network at all.

**Lead with "Ask the index".** Invite them to type a question you have never seen. The server
scores every document live and shows the same query answered three ways, with real similarity
numbers and real pool sizes. That is the moment that proves the asset is real; a recorded
answer or an animated reveal proves nothing, because a prospect cannot tell it from typed text.

Then "The documents" if they want to poke at the corpus, "The disclosure" for the finding, and
"Evidence" for the row counts — 14,000+ recorded model responses with dates, read live off the
files rather than asserted on a slide.

`demo/bd_demo.ipynb` remains for a technical follow-up where someone wants to see the code.

---

## 1 · The offer ladder

Never lead with the big engagement. Lead with something small enough to say yes to inside a
discretionary budget, and structured so the next step is obvious.

| | Offer | Why it sells | What it leads to |
|---|---|---|---|
| **Free** | This demo, 8 minutes | Costs them nothing; the finding is memorable | The diagnostic |
| **Entry** | **2-week diagnostic** on one assistant | Small, fixed scope, no procurement drama | The full assessment |
| **Core** | Full assessment across their risk areas | The real engagement | Ongoing |
| **Recurring** | The suite re-runs before every release | **The actual prize** | Renewal |

**The recurring tier is the business.** A one-off assessment is one invoice and a report that ages.
A regression suite runs on every model change, prompt change and new data source — and the risk
genuinely does re-emerge each time, so the pitch is honest rather than a maintenance upsell.

**Anchor on the diagnostic.** It is deliberately narrow: one assistant, their own corpus, one
question — *does this failure exist here?* Two weeks, one clear deliverable. It is the easiest yes
in the ladder and it qualifies the account better than any discovery call.

---

## 2 · Who buys, and what moves them

Their choice in the "pick a disclosure" step tells you who is in the room.

| Picks | Who they are | What they actually care about | Say this |
|---|---|---|---|
| ① termination · ③ medical leave | HR, DPO, employment counsel | personal data, GDPR Art. 9, employee trust | "Your DPIA cannot see this. It reviews documents; this is about answers." |
| ② ethics investigation | Compliance, internal audit, legal | investigation confidentiality, whistleblower protection | "The confidentiality you promised an investigation subject is not enforceable here." |
| ④ acquisition target | Corp dev, general counsel, CISO | MNPI, market abuse, deal leaks | "Your deal-room controls are document-level. The assistant works at the answer level." |

**The economic buyer is usually not in the room.** Whoever you demo to needs to be able to re-tell
this in one sentence to someone with budget. That sentence is: *"our access controls all passed and
it still disclosed an employee's medical leave."* Make sure they leave saying it correctly — say it
twice.

---

## 3 · Qualifying, fast

The three questions in the demo's first cell are the qualifier. Three yeses = a real prospect.

Then one more, quietly: **"Who signed off that assistant going live?"**

- A named person with a documented review → they have a process, and a gap in it. Strong.
- Nobody / "the team just shipped it" → bigger risk, but likely no budget line yet. Nurture.
- "We haven't launched yet" → **best case.** Pre-launch is where testing is cheap and a finding
  changes a design decision rather than triggering an incident.

---

## 4 · Objection handling

**"Our data is properly classified."**
That is exactly the point, and worth agreeing with warmly. Every document in the demo *was*
correctly classified. Neither source was confidential. The disclosure was created by combining them
— classification operates on documents, and this risk lives in answers.

**"Our vendor handles safety."**
Model vendors secure the model. This failure is not in the model — a model with no documents cannot
leak yours. It appears when you connect one to a document store, and that connection is yours.

**"We already do red teaming / pen testing."**
Ask what it covers. Conventional testing looks for a control being bypassed. Here nothing was
bypassed. Ask specifically: *does your current testing produce a number you can compare before and
after a prompt change?* Usually not.

**"Can you prove this happens in our environment?"**
No, and we would not claim to. That is precisely what the two-week diagnostic answers. *(This is
the strongest close in the deck — it converts scepticism directly into scope.)*

**"How do we know your testing is any good?"**
Show the credibility panel. The differentiator is what we report when we find *nothing*: every clean
result carries its detection limit, so we say "no failure above X% was detectable" rather than
"safe". Security teams recognise that immediately, because almost nobody does it.

**"Isn't this just a prompt engineering fix?"**
No — and this matters. There is no instruction that reliably prevents synthesis, because the
assistant is doing its job. The fixes are architectural: limit what can be combined, not what can
be read.

---

## 5 · What to leave behind

The demo is memorable; the artefact is what circulates. Leave exactly one page:

- the two source documents, side by side
- the question and the answer
- the four green checkmarks of what did **not** go wrong
- one line of scope for the diagnostic
- a name and a date

Do not leave the notebook, the repo, or a methodology deck. Those are for the second meeting, once
someone technical asks — and when they do, `docs/06b_rag_data_leakage.md` is the document that wins
that conversation.

---

## 6 · Where this can go wrong

**Over-claiming.** The temptation is to imply this proves their system is unsafe. It does not — it
proves one failure class exists and is measurable. Sophisticated buyers route the demo to a security
team, and that team will find any overreach. The honest version survives review; the inflated one
loses the account.

**Never imply you tested their system.** Always frame as a reference scenario on a public corpus.
Say it out loud once.

**Never name the model vendor.** It reframes an architecture problem as a procurement problem, which
is both wrong and un-sellable — they will simply ask their vendor about it.

**Don't run past eight minutes.** The demo is not the meeting. It buys the diagnostic.

---

## 7 · The one-sentence positioning

> Your access controls are working. That is the problem — they are answering a question about
> documents, and the risk is in the answers.

Everything else is elaboration.
