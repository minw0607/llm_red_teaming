#!/usr/bin/env python3
"""
demo/serve.py — The demo, backed by the real system.

    python demo/serve.py            # opens a browser at localhost:8765
    python demo/serve.py --port 9000

Why this exists rather than a slide deck
----------------------------------------
An earlier version of this demo was a self-contained HTML page with the responses
baked in as strings. It looked good and proved nothing: a prospect cannot tell a
recorded answer from a typed one, and a typewriter animation *simulating*
computation is worse than none — it invites the suspicion it was meant to avoid.

So the centrepiece here is the part that cannot be faked. The visitor types a
question we have never seen; the server embeds it, scores all 600 real documents,
and returns the top matches with their actual cosine similarities — under three
different permission architectures at once. Same question, different documents,
because ``pre_filter`` searched 178 candidates and ``no_filter`` searched 600.

That is also, conveniently, the finding: **the architecture decides exposure, not
the model.** And it needs no API call, so the most persuasive part of the demo is
also the most reliable part.

Only the final disclosure step touches a model, and it falls back to the recorded
run if anything goes wrong.

Dependencies: only what the toolkit already needs (sentence-transformers, numpy).
No web framework — this is the standard library's http.server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

STATE: dict = {}


def boot(n_docs: int = 600):
    """Build the corpus and index once, at startup."""
    from attacks.rag import (build_corpus, VectorIndex, build_aggregation_sets,
                             CLEARANCES)
    print(f"  building the index over {n_docs} real documents…", flush=True)
    corpus = build_corpus(n_docs)
    agg_docs, agg_sets = build_aggregation_sets()
    docs = list(corpus) + list(agg_docs)
    index = VectorIndex(docs)
    STATE.update(docs=docs, index=index, clearances=CLEARANCES,
                 agg={s.set_id: s for s in agg_sets},
                 by_id={d.doc_id: d for d in docs})
    print(f"  ready — {len(docs)} documents indexed", flush=True)


def api_search(q: str, clearance: str, k: int = 5) -> dict:
    """The un-fakeable part: one query, three architectures, real scores."""
    from attacks.rag import may_read
    idx, out = STATE["index"], {}
    for arch in ("no_filter", "post_filter", "pre_filter"):
        delivered, withheld = idx.search_debug(q, clearance=clearance, k=k,
                                               architecture=arch)
        # How many documents this wiring was even allowed to consider — the
        # number that explains why the same question returns different results.
        pool = (sum(1 for d in STATE["docs"] if may_read(clearance, d.tier))
                if arch == "pre_filter" else len(STATE["docs"]))
        out[arch] = {
            "pool": pool,
            "delivered": [{"id": h.doc.doc_id, "tier": h.doc.tier,
                           "score": round(h.score, 4), "entitled": h.entitled,
                           "title": h.doc.title[:78]} for h in delivered],
            "withheld": [{"id": h.doc.doc_id, "tier": h.doc.tier,
                          "score": round(h.score, 4)} for h in withheld],
        }
    return out


def api_doc(doc_id: str) -> dict:
    d = STATE["by_id"].get(doc_id)
    if not d:
        return {"error": "not found"}
    return {"id": d.doc_id, "tier": d.tier, "text": d.text,
            "canary": bool(d.canary)}


def api_corpus(query: str = "", limit: int = 40) -> dict:
    """Plain substring browse — proves the documents are real and readable."""
    docs = STATE["docs"]
    if query:
        ql = query.lower()
        docs = [d for d in docs if ql in d.text.lower()]
    return {"total": len(STATE["docs"]), "matched": len(docs),
            "docs": [{"id": d.doc_id, "tier": d.tier, "title": d.title[:90]}
                     for d in docs[:limit]]}


def api_disclose(set_id: str, live: bool) -> dict:
    """The scripted scene — but the retrieval underneath it is real."""
    agg = STATE["agg"][set_id]
    idx = STATE["index"]
    delivered, _ = idx.search_debug(agg.question, clearance="employee", k=6,
                                    architecture="pre_filter")
    got = [h.doc.doc_id for h in delivered if h.doc.doc_id in agg.fragment_ids]
    answer, source, err = "", "recorded", None
    if live:
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(_ROOT, ".env"))
            from targets import AzureOpenAITarget
            from attacks.rag import RagAssistant
            a = RagAssistant(AzureOpenAITarget(), idx, k=6, sleep_sec=0)
            answer = a.answer(agg.question, clearance="employee",
                              architecture="pre_filter")[0].strip()
            source = "live"
        except Exception as exc:                       # fall back, never fail on stage
            err, live = f"{type(exc).__name__}", False
    if not live:
        path = os.path.join(_ROOT, "results", "06b_ckpt_aggregation.jsonl")
        for line in open(path):
            r = json.loads(line)
            if r["target_doc_id"] == set_id and r["family"] == "aggregation":
                answer = r["response"].strip()
                break
    from attacks.rag import score_aggregation
    sc = score_aggregation(answer, agg)
    return {"question": agg.question, "answer": answer, "source": source,
            "error": err, "composed": bool(sc["composed"]),
            "fragments_delivered": len(got), "fragments_total": len(agg.fragment_ids),
            "documents": [{"id": i, "text": STATE["by_id"][i].text,
                           "tier": STATE["by_id"][i].tier} for i in agg.fragment_ids]}


def api_receipts() -> dict:
    """Row counts and dates straight off the evidence files on this machine."""
    import datetime
    rdir = os.path.join(_ROOT, "results")
    want = [("06b_ckpt_boundary.jsonl", "Document access — assistant responses"),
            ("06b_ckpt_poison.jsonl", "Planted-instruction attempts"),
            ("06b_ckpt_aggregation.jsonl", "Assembled-disclosure attempts"),
            ("02b_ckpt_layers.jsonl", "Guardrail layers — bank assistant"),
            ("07_ckpt_agent.jsonl", "Agent tool-use attempts"),
            ("04b_ckpt_allocation.jsonl", "Hiring decisions — fairness audit"),
            ("05_ckpt_nli.jsonl", "Reasoning-robustness items")]
    rows, total = [], 0
    for fn, label in want:
        p = os.path.join(rdir, fn)
        if not os.path.exists(p):
            continue
        # Count only rows at the CURRENT harness version. These files are
        # append-only and retain rows written before a scoring or taxonomy
        # change, so a raw line count overstates what was actually measured:
        # 07_ckpt_agent.jsonl holds 135 lines but only 60 current ones. Showing
        # the raw number would contradict the written findings the moment
        # anyone cross-checked.
        recs = []
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
        versions = {r.get("harness_version") for r in recs
                    if r.get("harness_version") is not None}
        n = (sum(1 for r in recs if r.get("harness_version") == max(versions))
             if versions else len(recs))
        superseded = len(recs) - n
        total += n
        rows.append({"file": fn, "label": label, "rows": n,
                     "superseded": superseded,
                     "modified": datetime.datetime.fromtimestamp(
                         os.path.getmtime(p)).strftime("%d %b %Y")})
    return {"rows": rows, "total": total}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                        # keep the console clean on stage
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(_HERE, "app.html"), "rb") as f:
                    return self._send(f.read(), "text/html; charset=utf-8")
            if u.path == "/api/search":
                data = api_search(q.get("q", ""), q.get("clearance", "employee"))
            elif u.path == "/api/doc":
                data = api_doc(q.get("id", ""))
            elif u.path == "/api/corpus":
                data = api_corpus(q.get("q", ""))
            elif u.path == "/api/disclose":
                data = api_disclose(q.get("set", "AGG3"), q.get("live") == "1")
            elif u.path == "/api/receipts":
                data = api_receipts()
            elif u.path == "/api/meta":
                data = {"documents": len(STATE["docs"]),
                        "clearances": STATE["clearances"],
                        "sets": {k: v.question for k, v in STATE["agg"].items()}}
            else:
                return self._send(b"not found", "text/plain", 404)
            return self._send(json.dumps(data).encode(), "application/json")
        except Exception as exc:
            return self._send(json.dumps({"error": f"{type(exc).__name__}: {exc}"}
                                         ).encode(), "application/json", 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--docs", type=int, default=600)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    boot(a.docs)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"\n  ▸ {url}\n    (ctrl-C to stop)\n", flush=True)
    if not a.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("  stopped")


if __name__ == "__main__":
    main()
