"""Latency benchmark (in-process, full pipeline incl. pre/post-processing and JSON serialisation).

    python scripts/benchmark_latency.py                 # all experiments (needs llama-server for AI modes)
    python scripts/benchmark_latency.py --no-llm        # retrieval-only experiments
    python scripts/benchmark_latency.py --cpu           # force CPU for encoders (CPU-only report)

Experiments
 1. per-mode latency over test queries: avg / p50 / p95 / max, per stage, TTFT and total
 2. corpus-size scaling: 1k / 10k / 100k chunks (real KB + synthetic distractor chunks) for
    BM25 and FAISS (flat vs HNSW)
 3. top-k scaling: rerank_k in {5,10,20,30,50} -> rerank latency and Recall@5/MRR
 4. context_k in {2,4,6} -> prompt tokens, TTFT
 5. long input: maximum query length tested (approx. 512 tokens) through the whole Smart AI pipeline
Writes results/latency_report.json and appends a Markdown summary to results/latency_report.md.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402


def stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    if not xs:
        return {}
    return {"n": len(xs), "avg": round(statistics.mean(xs), 2), "p50": round(xs[len(xs) // 2], 2),
            "p95": round(xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))], 2), "max": round(xs[-1], 2)}


def hardware() -> dict:
    import psutil
    info = {"platform": platform.platform(), "cpu": platform.processor(), "cpu_threads": psutil.cpu_count(),
            "ram_GiB": round(psutil.virtual_memory().total / 2**30, 1)}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_GiB"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    except Exception:
        pass
    return info


def gpu_used_mib() -> float | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return round(pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20, 0)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--n", type=int, default=60, help="queries per mode")
    ap.add_argument("--sizes", default="1000,10000,100000")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--tag", default="after")
    ap.add_argument("--quick", action="store_true", help="only experiment 1 (per-mode latency)")
    ap.add_argument("--merge", action="store_true",
                    help="with --quick: update per-mode latency in an existing results file, keep its other sections")
    args = ap.parse_args()
    if args.cpu:
        os.environ["HAMS_DEVICE"] = "cpu"

    import psutil
    from hamsrag.config import load_config, p
    from hamsrag.rag import SearchService
    from hamsrag.retrieval import RetrievalEngine

    cfg = load_config("serving_config.yaml")
    rows = [json.loads(x) for x in p("data/test.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    queries = [r["query"] for r in rows][: args.n]
    engine = RetrievalEngine(cfg, embedder=args.embedder, reranker=args.reranker)
    svc = SearchService(cfg, engine=engine)
    llm_up = (not args.no_llm) and svc.llm.health()
    report = {"system_ram_used_GiB_at_start": round(psutil.virtual_memory().used / 2**30, 2), "hardware": hardware(), "device": "cpu" if args.cpu else "auto", "embedder": engine.embedder_name,
              "reranker": engine.reranker_name, "llm": cfg["llm"] if llm_up else None, "n_queries": len(queries)}
    proc = psutil.Process()

    # warm-up (CUDA kernels, SQLite page cache, llama.cpp prompt cache)
    for q in queries[:3]:
        svc.quick_search(q); engine.retrieve(q)
        if llm_up:
            svc.smart_search(q)

    # ---------------- 1. per-mode latency
    per_mode = {}
    qs = {"total": [], "search": [], "preprocessing": []}
    for q in queries:
        t = time.perf_counter()
        out = json.dumps(svc.quick_search(q), ensure_ascii=False)  # include serialisation
        tot = (time.perf_counter() - t) * 1000
        d = json.loads(out)["latency_ms"]
        qs["total"].append(tot); qs["search"].append(d["search"]); qs["preprocessing"].append(d["preprocessing"])
    per_mode["quick_search"] = {k: stats(v) for k, v in qs.items()}
    rt = {"retrieval_stage": [], "rerank": [], "first_result": []}
    for q in queries:
        t = time.perf_counter()
        hits, tm = engine.hybrid(q)
        t_first = (time.perf_counter() - t) * 1000
        _, tr = engine.rerank(q, hits)
        rt["retrieval_stage"].append(t_first); rt["first_result"].append(t_first); rt["rerank"].append(tr["rerank"])
    per_mode["retrieval_only(hybrid+rerank)"] = {k: stats(v) for k, v in rt.items()}
    if llm_up:
        sm = {k: [] for k in ("retrieval", "rerank", "time_to_first_token", "generation_total", "total")}
        toks = []
        for q in queries:
            t = time.perf_counter()
            out = svc.smart_search(q)
            json.dumps(out, ensure_ascii=False)
            for k in ("retrieval", "rerank", "time_to_first_token", "generation_total"):
                sm[k].append(out["latency_ms"][k])
            sm["total"].append((time.perf_counter() - t) * 1000)
            toks.append(out.get("tokens", {}))
        per_mode["smart_ai_search"] = {k: stats(v) for k, v in sm.items()}
        per_mode["smart_ai_search"]["avg_prompt_tokens"] = round(statistics.mean(t.get("prompt", 0) for t in toks), 1)
        per_mode["smart_ai_search"]["avg_completion_tokens"] = round(statistics.mean(t.get("completion", 0) for t in toks), 1)
        convs = [json.loads(x) for x in p("data/test_conversations.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        it = {k: [] for k in ("query_rewrite", "time_to_first_token", "total")}
        for c in convs:
            sid = svc.sessions.new()
            for tturn in c["turns"]:
                out = svc.interactive(tturn["q"], sid)
                for k in it:
                    it[k].append(out["latency_ms"][k])
        per_mode["interactive"] = {k: stats(v) for k, v in it.items()}
        per_mode["interactive"]["query_rewrite_when_triggered"] = stats([x for x in it["query_rewrite"] if x > 0])
    report["per_mode"] = per_mode
    report["memory"] = {"process_rss_GiB": round(proc.memory_info().rss / 2**30, 2), "gpu_used_MiB(all processes)": gpu_used_mib()}
    try:
        import torch
        report["memory"]["torch_peak_allocated_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0
    except Exception:
        pass
    print(json.dumps(per_mode, indent=1))
    if args.quick:
        res = p("results"); res.mkdir(exist_ok=True)
        old_file = res / f"latency_{args.tag}.json"
        if args.merge and old_file.exists():
            merged = json.loads(old_file.read_text(encoding="utf-8"))
            merged.update(report)  # new per-mode latency/memory/hardware; scaling, top-k, context sections kept
            report = merged
        (res / f"latency_{args.tag}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[saved] results/latency_{args.tag}.json")
        return

    # ---------------- 2. corpus-size scaling (BM25 + FAISS), synthetic distractors
    from hamsrag.ingest import Chunk
    from hamsrag.lexical import LexicalIndex
    import faiss
    base_chunks = engine.lexical_index.all_chunks()
    sentences = [s for c in base_chunks for s in c.text.replace("\n", ". ").split(". ") if len(s.split()) > 3]
    rng = random.Random(0)
    dim = engine.dense_index.index.d
    scaling = []
    qv = engine.dense_index.encode_queries(queries[:30])
    for size in [int(x) for x in args.sizes.split(",")]:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            lex = LexicalIndex(Path(td) / "s.sqlite")
            extra = size - len(base_chunks)
            synth = [Chunk(f"syn#{i}", f"syn{i // 5}", "synthetic", "", None, "mixed", "synthetic",
                           ". ".join(rng.sample(sentences, 4)) + f" {rng.randint(1, 99999)}", i % 5) for i in range(max(0, extra))]
            t = time.perf_counter(); lex.add(base_chunks + synth); lex.optimize(); t_idx = time.perf_counter() - t
            b = []
            for q in queries[:30]:
                t = time.perf_counter(); lex.search(q, 50); b.append((time.perf_counter() - t) * 1000)
            # FAISS: real vectors for the KB + random unit vectors for distractors (search cost depends only on N, dim)
            real = engine.dense_index.index.reconstruct_n(0, engine.dense_index.index.ntotal)
            rnd = np.random.default_rng(0).standard_normal((max(0, extra), dim)).astype("float32")
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            allv = np.vstack([real, rnd])
            res = {"chunks": size, "bm25_index_build_s": round(t_idx, 1), "bm25_ms": stats(b)}
            for kind in ("flat", "hnsw"):
                if kind == "flat":
                    idx = faiss.IndexFlatIP(dim)
                else:
                    idx = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT); idx.hnsw.efConstruction = 80; idx.hnsw.efSearch = 64
                t = time.perf_counter(); idx.add(allv); t_build = time.perf_counter() - t
                d = []
                for v in qv:
                    t = time.perf_counter(); idx.search(v[None, :], 50); d.append((time.perf_counter() - t) * 1000)
                res[f"faiss_{kind}_ms"] = stats(d)
                res[f"faiss_{kind}_build_s"] = round(t_build, 1)
            lex.close()  # Windows cannot delete an open SQLite file
            scaling.append(res)
            print("[scaling]", json.dumps(res))
    report["corpus_scaling"] = scaling

    # ---------------- 3. rerank top-k scaling (latency + quality)
    from hamsrag.metrics import score_query
    topk = []
    for k in (5, 10, 20, 30, 50):
        lat, mets = [], []
        for r in rows[: args.n]:
            hits, _ = engine.hybrid(r["query"], top_n=k)
            t = time.perf_counter()
            rr, _ = engine.rerank(r["query"], hits)
            lat.append((time.perf_counter() - t) * 1000)
            mets.append(score_query([h.chunk.chunk_id for h in rr], r["gold"]))
        topk.append({"rerank_k": k, "rerank_ms": stats(lat), "recall@5": round(statistics.mean(m["recall@5"] for m in mets), 4),
                     "mrr": round(statistics.mean(m["mrr"] for m in mets), 4)})
        print("[top-k]", topk[-1])
    report["rerank_topk"] = topk

    # ---------------- 4/5. context size and long input (LLM)
    if llm_up:
        ctx_res = []
        orig = cfg["retrieval"]["context_k"]
        for ck in (2, 4, 6):
            svc.rc["context_k"] = ck
            tt, pt = [], []
            for q in queries[:20]:
                out = svc.smart_search(q)
                tt.append(out["latency_ms"]["time_to_first_token"]); pt.append(out.get("tokens", {}).get("prompt", 0))
            ctx_res.append({"context_k": ck, "ttft_ms": stats(tt), "avg_prompt_tokens": round(statistics.mean(pt), 1)})
            print("[context]", ctx_res[-1])
        svc.rc["context_k"] = orig
        report["context_k"] = ctx_res
        long_q = ("I am the facilities manager of a company with offices in Riyadh, Jeddah and Dammam and we are comparing options "
                  "for furnishing our new headquarters. " * 6) + "What is the monthly price of the Enterprise package and what does it include?"
        out = svc.smart_search(long_q)
        report["long_input"] = {"query_words": len(long_q.split()), "approx_tokens": out.get("tokens", {}).get("prompt"),
                                "latency_ms": out["latency_ms"], "answered": not out["not_found"]}
    res = p("results")
    res.mkdir(exist_ok=True)
    (res / f"latency_{args.tag}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    if args.tag == "after":
        (res / "latency_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] results/latency_{args.tag}.json")


if __name__ == "__main__":
    main()
