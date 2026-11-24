"""Generate extra training queries for TRAINING-document chunks with the local LLM, then filter.

    python scripts/generate_synthetic_queries.py            # needs llama-server running (tools/start_llm.bat)

Pipeline (InPars/Promptagator-style, fully self-hosted):
 1. For each training chunk, the local LLM writes 6 questions answerable from the chunk:
    2 MSA Arabic, 1 Saudi dialect, 2 English, 1 Arabic/English code-switched.
 2. Filters: valid JSON; script/language matches the requested type; not a near-copy of the
    passage; not a duplicate of an existing query.
 3. Round-trip consistency: the base retriever (BM25 + base e5 -> RRF) must rank the source chunk
    in its top-5 over the whole corpus, otherwise the question is discarded as off-target.
Output: data/synthetic_train.jsonl (+ a rejection log in data/processed/synthetic_rejected.jsonl).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hamsrag.config import load_config, p  # noqa: E402
from hamsrag.fusion import rrf  # noqa: E402
from hamsrag.lexical import LexicalIndex  # noqa: E402
from hamsrag.llm import LLMClient  # noqa: E402
from hamsrag.qa_data import norm_match  # noqa: E402
from hamsrag.retrieval import chunks_from_jsonl  # noqa: E402
from hamsrag.text import detect_language  # noqa: E402

PROMPT = """You write search questions for a company knowledge base (Saraya Living, a Saudi furniture retailer).
Read the passage and write 6 different questions that a customer or employee could ask and that the passage answers.
Rules:
- Every question must be fully answerable from the passage alone. Mention the specific product, policy or service so the question is self-contained.
- Do not copy sentences from the passage; paraphrase naturally. Vary length; make at least one question long and detailed.
- Use exactly these styles, in this order:
  1. "ar": Modern Standard Arabic
  2. "ar": Modern Standard Arabic, different angle
  3. "dia": Saudi colloquial Arabic (e.g. أبي، وش، كم، أقدر، حق)
  4. "en": English
  5. "en": English, different angle
  6. "mx": Arabic sentence with one or two English terms mixed in (e.g. "كم الـdelivery fee ...")
Return ONLY a JSON list: [{"type": "ar", "q": "..."}, ...]

Passage title: {title}
Passage:
{text}
"""


def parse(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [it for it in items if isinstance(it, dict) and isinstance(it.get("q"), str) and it.get("type") in {"ar", "dia", "en", "mx"}]


def lang_ok(qtype: str, q: str) -> bool:
    d = detect_language(q)
    if qtype in ("ar", "dia"):
        return d["lang"] == "ar"
    if qtype == "en":
        return d["lang"] == "en" and not d["mixed"]
    return d["mixed"]  # mx must contain both scripts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--embedder", default="e5-base")
    args = ap.parse_args()
    cfg = load_config("serving_config.yaml")
    llm = LLMClient()
    if not llm.health():
        sys.exit("llama-server is not running on " + llm.base_url + " - start tools/start_llm.bat first")

    chunks = chunks_from_jsonl(p("data/processed/chunks.jsonl"))
    splits = json.loads(p("data/processed/doc_splits.json").read_text(encoding="utf-8"))
    train_chunks = [c for c in chunks if splits[c.doc_id] == "train"][: args.limit]
    existing = {norm_match(json.loads(x)["query"]) for f in ("train", "validation", "test")
                for x in p(f"data/{f}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}

    generated, rejected = [], []
    t0 = time.time()
    for i, c in enumerate(train_chunks):
        g = llm.chat([{"role": "user", "content": PROMPT.replace("{title}", f"{c.title} — {c.section}").replace("{text}", c.text)}],
                     max_tokens=700, temperature=0.7)
        items = parse(g.text)
        if not items:
            rejected.append({"chunk_id": c.chunk_id, "reason": "unparseable", "raw": g.text[:300]})
        for it in items:
            q = " ".join(it["q"].split())
            nq = norm_match(q)
            reason = None
            if not lang_ok(it["type"], q):
                reason = "language"
            elif len(q) < 8 or nq in existing:
                reason = "duplicate/short"
            elif nq in norm_match(c.text) or len(set(nq.split()) & set(norm_match(c.text).split())) / max(1, len(nq.split())) > 0.9:
                reason = "copied"
            if reason:
                rejected.append({"chunk_id": c.chunk_id, "q": q, "type": it["type"], "reason": reason})
                continue
            existing.add(nq)
            generated.append({"query": q, "qtype": it["type"], "chunk_id": c.chunk_id, "doc_id": c.doc_id})
        print(f"[{i + 1}/{len(train_chunks)}] {c.chunk_id}: {len(items)} questions ({g.total_ms/1000:.1f}s)", flush=True)
    print(f"generation: {time.time() - t0:.0f}s, {len(generated)} candidates")

    # round-trip consistency filter
    from hamsrag.dense import DenseIndex
    lex = LexicalIndex(p(cfg["index_dir"]) / "kb.sqlite")
    di = DenseIndex(args.embedder)
    di.build(chunks)
    dense = di.search_vectors(di.encode_queries([g["query"] for g in generated], batch_size=64), 50)
    kept = []
    for g, d in zip(generated, dense):
        fused = [cid for cid, _ in rrf({"bm25": [cid for cid, _ in lex.search(g["query"], 50)], "dense": [cid for cid, _ in d]})][:5]
        if g["chunk_id"] in fused:
            det = detect_language(g["query"])
            kept.append({"qid": f"syn:{len(kept)}", "query": g["query"], "qtype": g["qtype"], "qlang": det["lang"],
                         "answer_lang": det["answer_lang"], "doc_id": g["doc_id"], "gold": [g["chunk_id"]],
                         "split": "train", "source": "synthetic_llm", "group": "synthetic", "dialect": g["qtype"] == "dia"})
        else:
            rejected.append({"chunk_id": g["chunk_id"], "q": g["query"], "reason": "round-trip"})
    with open(p("data/synthetic_train.jsonl"), "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(p("data/processed/synthetic_rejected.jsonl"), "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"kept {len(kept)} / {len(generated)} candidates; rejected: {Counter(r['reason'] for r in rejected)}")
    print("kept by type:", Counter(r["qtype"] for r in kept))


if __name__ == "__main__":
    main()
