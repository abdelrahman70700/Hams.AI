"""Generation quality for Smart AI Search and Interactive AI Search on held-out data.

    python scripts/benchmark_generation.py --tag after                                   # deployed models
    python scripts/benchmark_generation.py --tag before --embedder e5-base --reranker gte-rerank
    python scripts/benchmark_generation.py --tag after --judge                           # + local LLM judge

Needs llama-server running. Evaluates: test.jsonl (answerable), test_unanswerable.jsonl and
test_conversations.jsonl (Interactive, real rewriting). Writes:
  results/generation_<tag>.json         aggregate metrics (overall and per group)
  results/generation_items_<tag>.jsonl  every answer with its metrics (error analysis, human eval)
  results/human_eval_<tag>.csv          30 items (10 AR, 10 EN, 10 mixed/cross) to score 1-5
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from hamsrag.config import load_config, p  # noqa: E402
from hamsrag.gen_metrics import (corpus_bleu_chrf, language_ok, lexical_support,  # noqa: E402
                                 numeric_faithfulness, rouge_l)

JUDGE = """You check whether an ANSWER is fully supported by the SOURCES.
SOURCES:
{src}

QUESTION: {q}
ANSWER: {a}

Is every factual statement in the ANSWER supported by the SOURCES? Reply with exactly one word: YES or NO."""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def safe_write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    """Write a results file; if another program has it locked (Windows), retry, then fall back to <name>.new
    so a long benchmark is never lost because a file is open in an editor or preview pane."""
    for _ in range(10):
        try:
            path.write_text(text, encoding=encoding)
            return path
        except PermissionError:
            time.sleep(2)
    alt = path.with_name(path.name + ".new")
    alt.write_text(text, encoding=encoding)
    print(f"[warn] {path.name} is locked by another program; wrote {alt.name} instead", flush=True)
    return alt


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def item_metrics(out: dict, gold: list[str], ref: str | None, expected_lang: str, by_id: dict) -> dict:
    cited = [c["chunk_id"] for c in out["citations"]]
    ctx = [r["chunk_id"] for r in out.get("retrieved", [])]
    support = [by_id[c].text for c in cited if c in by_id] or [by_id[c].text for c in ctx if c in by_id]
    g = set(gold)
    m = {"abstained": out["not_found"], "lang_ok": language_ok(out["answer"], expected_lang),
         "ttft_ms": out["latency_ms"]["time_to_first_token"], "total_ms": out["latency_ms"]["total"]}
    if g:  # answerable
        m["context_recall"] = 1.0 if set(ctx) & g else 0.0
        m["context_precision"] = len(set(ctx) & g) / len(ctx) if ctx else 0.0
        if not out["not_found"]:
            m["citation_precision"] = len(set(cited) & g) / len(cited) if cited else 0.0
            m["citation_recall"] = len(set(cited) & g) / len(g)
            m["numeric_faithfulness"] = numeric_faithfulness(out["answer"], support)
            m["lexical_support"] = lexical_support(out["answer"], support)
            m["citation_inferred"] = out.get("citation_inferred", False)
            m["rouge_l"] = rouge_l(out["answer"], ref) if ref else None
            # hallucination = a number not present in the sources, or (same-language answers) most content
            # words absent from the cited sources
            unsupported = (m["numeric_faithfulness"] is not None and m["numeric_faithfulness"] < 1.0) or \
                          (m["lexical_support"] is not None and m["lexical_support"] < 0.4)
            m["hallucination"] = 1.0 if unsupported else 0.0
    else:  # unanswerable: any non-abstaining answer is a hallucination
        m["hallucination"] = 0.0 if out["not_found"] else 1.0
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge", action="store_true", help="local LLM YES/NO faithfulness judge (weak, secondary)")
    ap.add_argument("--skip-interactive", action="store_true")
    ap.add_argument("--abstain-threshold", type=float, default=None,
                    help="override the gate threshold (e.g. the one calibrated for the base reranker)")
    ap.add_argument("--gate-file", default=None, help="read the threshold from a calibrate_gate.py result JSON")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute metrics from results/generation_items_<tag>.jsonl (no retrieval/LLM calls); "
                         "re-applies the current abstention detection to the stored answers")
    ap.add_argument("--relevance", action="store_true", help="with --rescore: (re)compute answer relevance")
    ap.add_argument("--no-relevance", action="store_true", help="skip answer relevance (compute later with --rescore --relevance)")
    args = ap.parse_args()
    if args.rescore:
        return rescore(args.tag, relevance=args.relevance)

    from hamsrag.rag import SearchService
    from hamsrag.retrieval import RetrievalEngine
    cfg = load_config("serving_config.yaml")
    if args.gate_file:
        args.abstain_threshold = json.loads(p(args.gate_file).read_text(encoding="utf-8"))["chosen"]["threshold"]
    if args.abstain_threshold is not None:
        cfg["retrieval"]["abstain_threshold"] = args.abstain_threshold
    engine = RetrievalEngine(cfg, embedder=args.embedder, reranker=args.reranker)
    svc = SearchService(cfg, engine=engine)
    if not svc.llm.health():
        sys.exit("llama-server is not running (tools/start_llm.bat)")
    by_id = {c.chunk_id: c for c in engine.lexical_index.all_chunks()}

    test = load_jsonl(p("data/test.jsonl"))[: args.limit]
    unans = load_jsonl(p("data/test_unanswerable.jsonl"))[: args.limit]
    convs = load_jsonl(p("data/test_conversations.jsonl"))[: args.limit]
    items = []
    t0 = time.time()
    for i, r in enumerate(test + unans):
        out = svc.smart_search(r["query"])
        m = item_metrics(out, r.get("gold", []), r.get("answer"), r["answer_lang"], by_id)
        items.append({"set": "unanswerable" if r.get("unanswerable") else "test", "mode": "smart_ai_search",
                      "qid": r["qid"], "group": r.get("group", "unanswerable"), "query": r["query"],
                      "expected_lang": r["answer_lang"], "reference": r.get("answer"), "gold": r.get("gold", []),
                      "answer": out["answer"], "citations": [c["chunk_id"] for c in out["citations"]],
                      "retrieved": [x["chunk_id"] for x in out.get("retrieved", [])],
                      "abstained_by": out.get("abstained_by"), "metrics": m, "latency_ms": out["latency_ms"]})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(test) + len(unans)} ({time.time() - t0:.0f}s)", flush=True)

    if not args.skip_interactive:
        for conv in convs:
            sid = svc.sessions.new()
            for turn in conv["turns"]:
                out = svc.interactive(turn["q"], sid)
                m = item_metrics(out, turn.get("gold", []), None, turn["lang"], by_id)
                m["rewrite_ms"] = out["latency_ms"].get("query_rewrite", 0.0)
                m["has_followups"] = bool(out.get("suggested_followups"))
                items.append({"set": "conversation", "mode": "interactive", "qid": f"{conv['id']}:{out['turn']}",
                              "group": "interactive", "query": turn["q"], "rewritten": out["rewritten_query"],
                              "reference_rewrite": turn.get("rewrite"), "expected_lang": turn["lang"],
                              "gold": turn.get("gold", []), "answer": out["answer"],
                              "citations": [c["chunk_id"] for c in out["citations"]],
                              "retrieved": [x["chunk_id"] for x in out.get("retrieved", [])],
                              "followups": out.get("suggested_followups"), "metrics": m, "latency_ms": out["latency_ms"]})

    if args.judge:
        for it in items:
            if it["metrics"].get("abstained") or not it["citations"]:
                continue
            src = "\n\n".join(f"[{i + 1}] {by_id[c].text}" for i, c in enumerate(it["citations"]) if c in by_id)
            g = svc.llm.chat([{"role": "user", "content": JUDGE.format(src=src, q=it["query"], a=it["answer"])}],
                             max_tokens=3, temperature=0.0)
            it["metrics"]["judge_faithful"] = 1.0 if g.text.strip().upper().startswith("YES") else 0.0

    for x in items:
        x["metrics"].pop("answer_relevance", None)
    # save the raw answers first, so nothing is lost if a later step fails (e.g. low RAM)
    res_dir = p("results"); res_dir.mkdir(exist_ok=True)
    safe_write(res_dir / f"generation_items_{args.tag}.jsonl", "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    meta = {"embedder": engine.embedder_name, "reranker": engine.reranker_name, "llm": cfg["llm"],
            "context_k": cfg["retrieval"]["context_k"], "abstain_threshold": cfg["retrieval"].get("abstain_threshold"),
            "seconds": round(time.time() - t0, 1)}
    # answer relevance needs one more encoder; it is computed by `--rescore --relevance` in a fresh process
    finalize(items, meta, args.tag, recompute_relevance=not args.no_relevance)


def rescore(tag: str, relevance: bool = False) -> None:
    from hamsrag.prompts import NOT_FOUND
    from hamsrag.rag import SearchService
    from hamsrag.retrieval import chunks_from_jsonl
    by_id = {c.chunk_id: c for c in chunks_from_jsonl(p("data/processed/chunks.jsonl"))}
    items = load_jsonl(p(f"results/generation_items_{tag}.jsonl"))
    old = json.loads(p(f"results/generation_{tag}.json").read_text(encoding="utf-8"))
    changed = 0
    for it in items:
        m0 = it["metrics"]
        abst = m0["abstained"] or SearchService.is_not_found(it["answer"], it["expected_lang"])
        if abst and not m0["abstained"]:
            changed += 1
            it["raw_answer"] = it["answer"]
            it["answer"], it["citations"] = NOT_FOUND[it["expected_lang"]], []
            it["abstained_by"] = "llm"
        out = {"not_found": abst, "answer": it["answer"], "citations": [{"chunk_id": c} for c in it["citations"]],
               "retrieved": [{"chunk_id": c} for c in it.get("retrieved", [])], "latency_ms": it["latency_ms"],
               "citation_inferred": m0.get("citation_inferred", False)}
        m = item_metrics(out, it["gold"], it.get("reference"), it["expected_lang"], by_id)
        for k in ("rewrite_ms", "has_followups"):
            if k in m0:
                m[k] = m0[k]
        if not abst:
            for k in ("judge_faithful",) + (() if relevance else ("answer_relevance",)):
                if k in m0:
                    m[k] = m0[k]
        it["metrics"] = m
    meta = {**old.get("_config", {}), "rescored": True, "rescored_new_abstentions": changed}
    print(f"[rescore] {tag}: {changed} stored answers re-classified as abstentions")
    finalize(items, meta, tag, recompute_relevance=relevance)


def finalize(items: list[dict], meta: dict, tag: str, recompute_relevance: bool = True) -> None:
    # ---------------- aggregate
    def agg(sel: list[dict]) -> dict:
        ms = [x["metrics"] for x in sel]
        answerable = [x for x in sel if x["gold"]]
        answered = [x["metrics"] for x in answerable if not x["metrics"]["abstained"]]
        res = {"n": len(sel),
               "language_correctness": mean([1.0 if m["lang_ok"] else 0.0 for m in ms]),
               "abstain_rate": mean([1.0 if m["abstained"] else 0.0 for m in ms]),
               "hallucination_rate": mean([m.get("hallucination") for m in ms]),
               "context_recall": mean([m.get("context_recall") for m in ms]),
               "context_precision": mean([m.get("context_precision") for m in ms]),
               "citation_precision": mean([m.get("citation_precision") for m in answered]),
               "citation_recall": mean([m.get("citation_recall") for m in answered]),
               "numeric_faithfulness": mean([m.get("numeric_faithfulness") for m in answered]),
               "lexical_support": mean([m.get("lexical_support") for m in answered]),
               "citation_inferred_rate": mean([1.0 if m.get("citation_inferred") else 0.0 for m in answered]),
               "rouge_l": mean([m.get("rouge_l") for m in answered]),
               "judge_faithful": mean([m.get("judge_faithful") for m in answered]),
               "ttft_ms_avg": mean([m["ttft_ms"] for m in ms]), "total_ms_avg": mean([m["total_ms"] for m in ms])}
        refs = [(x["answer"], x["reference"]) for x in answerable if x.get("reference") and not x["metrics"]["abstained"]]
        if refs:
            res.update(corpus_bleu_chrf([a for a, _ in refs], [r for _, r in refs]))
        return res

    # answer relevance: embedding similarity answer <-> reference answer (multilingual encoder)
    pairs = [x for x in items if x.get("reference") and not x["metrics"]["abstained"]]
    if pairs and recompute_relevance:
        from hamsrag.models import load_embedder
        enc = load_embedder("e5-base")
        va = enc.encode(["query: " + x["answer"] for x in pairs], normalize_embeddings=True)
        vr = enc.encode(["query: " + x["reference"] for x in pairs], normalize_embeddings=True)
        for x, s in zip(pairs, (va * vr).sum(1)):
            x["metrics"]["answer_relevance"] = float(s)

    groups = defaultdict(list)
    for it in items:
        groups[it["set"]].append(it)
        if it["set"] == "test":
            groups[f"test:{it['group']}"].append(it)
    report = {name: agg(sel) for name, sel in groups.items()}
    for name, sel in groups.items():
        report[name]["answer_relevance"] = mean([x["metrics"].get("answer_relevance") for x in sel])
    conv = [x for x in items if x["set"] == "conversation"]
    if conv:
        report["conversation"]["rewrite_ms_avg"] = mean([x["metrics"]["rewrite_ms"] for x in conv])
        report["conversation"]["followups_rate"] = mean([1.0 if x["metrics"]["has_followups"] else 0.0 for x in conv])
    report["_config"] = meta
    res = p("results")
    res.mkdir(exist_ok=True)
    safe_write(res / f"generation_{tag}.json", json.dumps(report, ensure_ascii=False, indent=1))
    safe_write(res / f"generation_items_{tag}.jsonl", "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))

    # human-evaluation sheet: 10 AR, 10 EN, 10 mixed/cross-lingual (fixed seed)
    rng = random.Random(7)
    pool = [x for x in items if x["set"] == "test"]
    def take(sel, n=10):
        sel = [x for x in pool if sel(x)]
        return rng.sample(sel, min(n, len(sel)))
    pick = take(lambda x: x["group"] == "ar") + take(lambda x: x["group"] == "en") + take(lambda x: x["group"] in ("mixed", "cross"))
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["qid", "group", "query", "answer", "reference", "citations", "accuracy_1to5", "faithfulness_1to5",
                "fluency_1to5", "completeness_1to5", "citation_quality_1to5", "notes"])
    for x in pick:
        w.writerow([x["qid"], x["group"], x["query"], x["answer"], x["reference"], " ".join(x["citations"]), "", "", "", "", "", ""])
    he = res / f"human_eval_{tag}.csv"
    scored = False
    if he.exists():  # never overwrite a sheet a person has started scoring
        try:
            rows = list(csv.DictReader(he.read_text(encoding="utf-8-sig").splitlines()))
            scored = any((r.get(c) or "").strip() for r in rows for c in r if c and c.endswith("_1to5"))
        except Exception:
            scored = True
    if scored:
        he = he.with_name(he.stem + ".new.csv")
        print(f"[info] {tag}: human-eval sheet already has scores; new sheet written to {he.name}")
    safe_write(he, "\ufeff" + buf.getvalue())
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("test:")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
