"""Retrieval quality benchmark: BM25 vs dense vs hybrid vs hybrid+rerank, per language group.

    python scripts/benchmark_retrieval.py --embedder e5-base --reranker gte-rerank --tag before
    python scripts/benchmark_retrieval.py --embedder e5-base-ft --reranker gte-rerank-ft --tag after
    python scripts/benchmark_retrieval.py --embedders e5-small e5-base gte-base bge-m3 --tag zeroshot  # dense bake-off
    python scripts/benchmark_retrieval.py --tune --split validation   # grid-search RRF k / weights

Dense indexes are built in memory for the requested embedder, so the benchmark never depends on
which index is currently deployed. Reports Recall@1/5/10/20, Hit@k, MRR, nDCG@10 for: all, ar, en,
mixed, cross (cross-lingual), dialect; plus the Interactive mode (gold rewrites of conversation
turns) and the cross-lingual direction split (AR->EN docs, EN->AR docs).
Results: results/retrieval_<tag>.json and a Markdown table in results/retrieval_report.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hamsrag.config import load_config, p  # noqa: E402
from hamsrag.fusion import rrf, union_candidates  # noqa: E402
from hamsrag.lexical import LexicalIndex  # noqa: E402
from hamsrag.metrics import aggregate, score_query  # noqa: E402

GROUPS = ["all", "ar", "en", "mixed", "cross", "dialect"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def xling_direction(r: dict) -> str | None:
    if r["group"] != "cross":
        return None
    return "ar->en" if r["qlang"] == "ar" else "en->ar"


def evaluate(rows: list[dict], ranked_lists: list[list[str]]) -> dict:
    per = []
    for r, ranked in zip(rows, ranked_lists):
        per.append({"group": r["group"], "dialect": r.get("dialect", False), "metrics": score_query(ranked, r["gold"])})
    agg = aggregate(per)
    for direction in ("ar->en", "en->ar"):
        sel = [x for x, r in zip(per, rows) if xling_direction(r) == direction]
        if sel:
            agg[f"cross {direction}"] = {k: round(sum(s["metrics"][k] for s in sel) / len(sel), 4) for k in sel[0]["metrics"]}
            agg[f"cross {direction}"]["n"] = len(sel)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument("--embedder", default=None, help="embedder for dense/hybrid modes")
    ap.add_argument("--embedders", nargs="*", default=None, help="dense-only bake-off over several embedders")
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--rerank-k", type=int, default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--tune", action="store_true", help="grid search RRF k and weights (use --split validation)")
    ap.add_argument("--fusion", choices=["rrf", "union"], default=None, help="override configs/serving_config.yaml")
    args = ap.parse_args()

    cfg = load_config("serving_config.yaml")
    rc = cfg["retrieval"]
    if args.fusion:
        rc["fusion"] = args.fusion
    lex = LexicalIndex(p(cfg["index_dir"]) / "kb.sqlite")
    chunks = lex.all_chunks()
    by_id = {c.chunk_id: c for c in chunks}
    rows = load_jsonl(p(f"data/{args.split}.jsonl"))
    queries = [r["query"] for r in rows]
    convs = load_jsonl(p("data/test_conversations.jsonl")) if args.split == "test" else []
    conv_rows = [{"query": t.get("rewrite") or t["q"], "gold": t["gold"], "group": "interactive",
                  "qlang": t["lang"]} for c in convs for t in c["turns"] if t["gold"]]
    out: dict = {"split": args.split, "n_queries": len(rows), "n_chunks": len(chunks), "modes": {}, "timing_ms": {}}

    # ---- BM25 (Quick Search)
    t = time.perf_counter()
    bm25 = [[cid for cid, _ in lex.search(q, 50)] for q in queries]
    out["timing_ms"]["bm25_per_query"] = round((time.perf_counter() - t) * 1000 / len(queries), 2)
    out["modes"]["bm25"] = evaluate(rows, bm25)

    from hamsrag.dense import DenseIndex

    if args.embedders:  # zero-shot dense bake-off
        for name in args.embedders:
            di = DenseIndex(name)
            t = time.perf_counter(); di.build(chunks); t_build = time.perf_counter() - t
            t = time.perf_counter(); qv = di.encode_queries(queries); t_q = time.perf_counter() - t
            dense = [[cid for cid, _ in r] for r in di.search_vectors(qv, 50)]
            out["modes"][f"dense[{name}]"] = evaluate(rows, dense)
            out["timing_ms"][f"dense[{name}]"] = {"index_build_s": round(t_build, 2), "query_encode_per_query": round(t_q * 1000 / len(queries), 2)}
            print(f"[bake-off] {name}: {json.dumps(out['modes'][f'dense[{name}]']['all'])}")
    else:
        name = args.embedder or cfg["embedder"]
        di = DenseIndex(name)
        di.build(chunks)
        t = time.perf_counter()
        qv = di.encode_queries(queries)
        out["timing_ms"]["dense_query_encode_per_query"] = round((time.perf_counter() - t) * 1000 / len(queries), 2)
        dense = [[cid for cid, _ in r] for r in di.search_vectors(qv, 50)]
        out["modes"]["dense"] = evaluate(rows, dense)

        if args.tune:
            grid = []
            for k_rrf, w_d in itertools.product([5, 10, 30, 60], [1.0, 1.5, 2.0, 3.0, 5.0]):
                fused = [[c for c, _ in rrf({"bm25": b, "dense": d}, k=k_rrf, weights={"bm25": 1.0, "dense": w_d})]
                         for b, d in zip(bm25, dense)]
                ev = evaluate(rows, fused)
                m = ev["all"]
                grid.append({"rrf_k": k_rrf, "w_dense": w_d, "ndcg@10": m["ndcg@10"], "recall@5": m["recall@5"],
                             "recall@20": m["recall@20"], "mrr": m["mrr"],
                             "cross_recall@20": ev.get("cross", {}).get("recall@20")})
            dense_all = out["modes"]["dense"]["all"]
            grid.append({"rrf_k": None, "w_dense": "dense-only", "ndcg@10": dense_all["ndcg@10"], "recall@5": dense_all["recall@5"],
                         "recall@20": dense_all["recall@20"], "mrr": dense_all["mrr"],
                         "cross_recall@20": out["modes"]["dense"].get("cross", {}).get("recall@20")})
            # candidates feed the reranker, so candidate recall@20 is the primary objective, nDCG@10 the tie-breaker
            grid.sort(key=lambda g: (-g["recall@20"], -g["ndcg@10"]))
            out["rrf_grid"] = grid
            print("best RRF settings:", grid[:3])
        weights = rc.get("rrf_weights") or {}
        hybrid_rrf = [[c for c, _ in rrf({"bm25": b, "dense": d}, k=rc["rrf_k"], weights=weights)] for b, d in zip(bm25, dense)]
        out["modes"]["hybrid_rrf"] = evaluate(rows, hybrid_rrf)
        kk = args.rerank_k or rc["rerank_k"]
        nb = rc.get("union_bm25", 5)
        hybrid_union = []
        for b, d in zip(bm25, dense):
            u = [c for c, _ in union_candidates({"bm25": b, "dense": d}, {"dense": kk - nb, "bm25": nb}, k=rc["rrf_k"], weights=weights)]
            rest = [c for c, _ in rrf({"bm25": b, "dense": d}, k=rc["rrf_k"], weights=weights) if c not in set(u)]
            hybrid_union.append(u + rest)
        out["modes"]["hybrid_union"] = evaluate(rows, hybrid_union)
        hybrid = hybrid_union if rc.get("fusion", "rrf") == "union" else hybrid_rrf
        out["fusion"] = rc.get("fusion", "rrf")

        if args.reranker:
            from hamsrag.rerank import Reranker
            rr = Reranker(args.reranker)
            k = args.rerank_k or rc["rerank_k"]
            reranked, t_total = [], 0.0
            for q, cand in zip(queries, hybrid):
                cs = [by_id[c] for c in cand[:k]]
                t = time.perf_counter()
                scored = rr.rerank(q, cs)
                t_total += time.perf_counter() - t
                reranked.append([c.chunk_id for c, _ in scored] + cand[k:])
            out["modes"]["hybrid+rerank"] = evaluate(rows, reranked)
            out["timing_ms"]["rerank_per_query"] = round(t_total * 1000 / len(queries), 2)
            out["rerank_k"] = k

        if conv_rows:  # Interactive mode: retrieval quality with the reference standalone rewrites
            cq = [r["query"] for r in conv_rows]
            cb = [[c for c, _ in lex.search(q, 50)] for q in cq]
            cd = [[c for c, _ in r] for r in di.search_vectors(di.encode_queries(cq), 50)]
            if rc.get("fusion", "rrf") == "union":
                ch = []
                for b, d in zip(cb, cd):
                    u = [c for c, _ in union_candidates({"bm25": b, "dense": d}, {"dense": kk - nb, "bm25": nb}, k=rc["rrf_k"], weights=weights)]
                    ch.append(u + [c for c, _ in rrf({"bm25": b, "dense": d}, k=rc["rrf_k"], weights=weights) if c not in set(u)])
            else:
                ch = [[c for c, _ in rrf({"bm25": b, "dense": d}, k=rc["rrf_k"], weights=weights)] for b, d in zip(cb, cd)]
            if args.reranker:
                ch = [[c.chunk_id for c, _ in rr.rerank(q, [by_id[x] for x in cand[:k]])] + cand[k:] for q, cand in zip(cq, ch)]
            out["modes"]["interactive(gold-rewrite)"] = {"all": evaluate(conv_rows, ch)["all"]}

    out["embedder"] = args.embedder or (args.embedders and ",".join(args.embedders)) or cfg["embedder"]
    out["reranker"] = args.reranker
    res_dir = p("results")
    res_dir.mkdir(exist_ok=True)
    (res_dir / f"retrieval_{args.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # Markdown table
    lines = [f"## Retrieval — {args.tag} ({args.split}, {len(rows)} queries, {len(chunks)} chunks)",
             f"embedder: `{out['embedder']}`, reranker: `{args.reranker}`", "",
             "| mode | group | n | R@1 | R@5 | R@10 | R@20 | Hit@5 | MRR | nDCG@10 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for mode, groups in out["modes"].items():
        for g, m in groups.items():
            lines.append(f"| {mode} | {g} | {m['n']} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | "
                         f"{m['recall@20']:.3f} | {m['hit@5']:.3f} | {m['mrr']:.3f} | {m['ndcg@10']:.3f} |")
    md = "\n".join(lines) + "\n"
    with open(res_dir / "retrieval_report.md", "a", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
