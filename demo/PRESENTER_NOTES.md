# Leadership demo — presenter notes

**Runtime:** 8–10 minutes of demo, leaving 5 for questions in a 15-minute slot.

```bash
python demo/leadership_demo.py --replay     # recommended for a live audience
python demo/leadership_demo.py              # live API calls
python demo/leadership_demo.py --no-pause   # rehearsal, runs straight through
```

The script pauses at every beat and waits for **Enter**. Talk, then press.

**Use `--replay` unless the room is small and the network is certain.** It plays back
the exact responses from the recorded run — visually identical, and it cannot hang,
rate-limit, or trip a content filter mid-sentence.

---

## The one sentence to open with

> "We can now test an AI system the way an attacker would. I want to show you one
> result, because it is the kind of thing our current reviews cannot find."

---

## Beat by beat

| Screen | What to say | Don't say |
|---|---|---|
| **Setup** | "Ordinary internal assistant. Documents have sensitivity levels, people have clearances. Today's user is a regular employee." | anything about retrieval, embeddings or models |
| **Control — refused** | "First, do the controls work? We ask for something genuinely confidential." *(press)* "It refuses, and it says why." | — |
| **Two documents** | **Ask the room: "Is either of these confidential?"** Let them answer. They'll say no. They're right. | don't answer it yourself |
| **Normal question** | "A perfectly ordinary question." *(press)* "Correct answer. And note — her name is not a secret. It's a seating list." | — |
| **The failure** | "One more question. Nothing in the store says this." *(press, then stay quiet)* | **do not talk over the reveal** |
| **What happened** | "That sentence is confidential, and ten seconds ago it did not exist anywhere in the company." | — |
| **The four checkmarks** | Read them slowly. This is the argument. | — |
| **The close** | "An access review checks that each document has the right label. It cannot tell you what happens when two correctly-labelled documents are read together." | — |

**The single most important instruction: say nothing for three seconds after the
reveal.** The sentence does the work. Filling the silence weakens it.

---

## Questions you will get

**"Is this a real system?"**
The documents are real corporate email (the Enron corpus, public since 2001). The
sensitivity labels and the four people are ours, so we know the right answer and can
score automatically. The assistant and the model are real.

**"Couldn't you just fix it?"**
Not with permissions — every permission here was already correct. It needs a
different control: limiting what can be *combined*, not what can be read. That's a
design change, and it's why finding this before launch matters.

**"How often does it happen?"**
Four out of four in this test. Not a probability — it did it every time.

**"What else did you find?"**
Same technique surfaced an active ethics investigation, an employee's medical leave,
and an unannounced acquisition target.

**"Does this affect systems we already have?"**
Any assistant answering questions over a mixed document store. The more documents it
can reach, the more combinations exist.

**"How much does it cost to run?"**
This scenario is a few hundred model calls — minutes, and pennies. It is repeatable
before and after every change, which is the point: it becomes a regression test.

---

## If someone asks a technical question you'd rather not open

> "Happy to go deeper offline — the full methodology, sample sizes and statistical
> treatment are written up, and I can walk your team through it."

Then move on. The write-up is `docs/06b_rag_data_leakage.md`.

---

## What NOT to do

- Don't show the notebook. Too much on screen, and the code invites code questions.
- Don't lead with the model name or the vendor. The finding is about the *deployment*,
  and naming a vendor makes it sound like a procurement issue.
- Don't claim it proves the system is unsafe overall. It proves one specific class of
  failure exists and is measurable. That's a stronger, more defensible claim.
- Don't skip the control. Showing the refusal first is what stops the room concluding
  the demo was rigged.
