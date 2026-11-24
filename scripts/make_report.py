"""Compile results/*.json into Markdown tables (results/summary.md) and the README results section.
Every number comes from a results file produced by the benchmark scripts; missing files are skipped.

    python scripts/make_report.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hamsrag.config import p  # noqa: E402

R = p("results")


def load(name: str) -> dict | None:
    f = R / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def f3(x) -> str:
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def ms(x) -> str:
    return "—" if x is None else f"{x:,.0f} ms"


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def human_scores(tag: str) -> dict | None:
    f = R / f"human_eval_{tag}.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(f.open(encoding="utf-8-sig")))
    def avg(col, sel=None):
        vals = [float(r[col]) for r in rows if r.get(col, "").strip() and (sel is None or sel(r))]
        return round(sum(vals) / len(vals), 2) if vals else None
    is_ar = lambda r: r["group"] == "ar" or bool(re.search(r"[؀-ۿ]", r["answer"]))  # noqa: E731
    return {"n_scored": sum(1 for r in rows if r.get("accuracy_1to5", "").strip()), "accuracy": avg("accuracy_1to5"),
            "faithfulness": avg("faithfulness_1to5"), "completeness": avg("completeness_1to5"),
            "citation_quality": avg("citation_quality_1to5"),
            "fluency_ar": avg("fluency_1to5", is_ar), "fluency_en": avg("fluency_1to5", lambda r: not is_ar(r))}


def main() -> None:
    parts = []
    rb, ra = load("retrieval_before_finetuning.json"), load("retrieval_after_finetuning.json")
    gb, ga = load("generation_before.json"), load("generation_after.json")
    lb, la = load("latency_before.json"), load("latency_after.json") or load("latency_report.json")
    hb, ha = human_scores("before"), human_scores("after")

    # ------------------------------------------------ before vs after (assessment table)
    def rv(res, mode, group, metric):
        try:
            return res["modes"][mode][group][metric]
        except (TypeError, KeyError):
            return None

    def gv(res, key, metric):
        try:
            return res[key][metric]
        except (TypeError, KeyError):
            return None

    def lv(res, mode, stage, stat="avg"):
        try:
            return res["per_mode"][mode][stage][stat]
        except (TypeError, KeyError):
            return None

    M = "hybrid+rerank"
    rows = [
        ["Recall@5 (AR)", f3(rv(rb, M, "ar", "recall@5")), f3(rv(ra, M, "ar", "recall@5"))],
        ["Recall@5 (EN)", f3(rv(rb, M, "en", "recall@5")), f3(rv(ra, M, "en", "recall@5"))],
        ["Recall@5 (mixed)", f3(rv(rb, M, "mixed", "recall@5")), f3(rv(ra, M, "mixed", "recall@5"))],
        ["Recall@5 (cross-lingual)", f3(rv(rb, M, "cross", "recall@5")), f3(rv(ra, M, "cross", "recall@5"))],
        ["MRR (AR)", f3(rv(rb, M, "ar", "mrr")), f3(rv(ra, M, "ar", "mrr"))],
        ["MRR (EN)", f3(rv(rb, M, "en", "mrr")), f3(rv(ra, M, "en", "mrr"))],
        ["nDCG@10 (AR)", f3(rv(rb, M, "ar", "ndcg@10")), f3(rv(ra, M, "ar", "ndcg@10"))],
        ["nDCG@10 (EN)", f3(rv(rb, M, "en", "ndcg@10")), f3(rv(ra, M, "en", "ndcg@10"))],
        ["nDCG@10 (all)", f3(rv(rb, M, "all", "ndcg@10")), f3(rv(ra, M, "all", "ndcg@10"))],
        ["nDCG@10 (cross-lingual)", f3(rv(rb, M, "cross", "ndcg@10")), f3(rv(ra, M, "cross", "ndcg@10"))],
        ["Faithfulness (numeric)", f3(gv(gb, "test", "numeric_faithfulness")), f3(gv(ga, "test", "numeric_faithfulness"))],
        ["Faithfulness (lexical support)", f3(gv(gb, "test", "lexical_support")), f3(gv(ga, "test", "lexical_support"))],
        ["Answer relevance (sim. to reference)", f3(gv(gb, "test", "answer_relevance")), f3(gv(ga, "test", "answer_relevance"))],
        ["Hallucination rate (answerable)", f3(gv(gb, "test", "hallucination_rate")), f3(gv(ga, "test", "hallucination_rate"))],
        ["Hallucination rate (unanswerable)", f3(gv(gb, "unanswerable", "hallucination_rate")), f3(gv(ga, "unanswerable", "hallucination_rate"))],
        ["Correct \"not found\" on unanswerable", f3(gv(gb, "unanswerable", "abstain_rate")), f3(gv(ga, "unanswerable", "abstain_rate"))],
        ["False \"not found\" on answerable", f3(gv(gb, "test", "abstain_rate")), f3(gv(ga, "test", "abstain_rate"))],
        ["Citation precision", f3(gv(gb, "test", "citation_precision")), f3(gv(ga, "test", "citation_precision"))],
        ["Local-LLM judge: faithful", f3(gv(gb, "test", "judge_faithful")), f3(gv(ga, "test", "judge_faithful"))],
        ["Answer language correct", f3(gv(gb, "test", "language_correctness")), f3(gv(ga, "test", "language_correctness"))],
        ["Human meaning score (1–5)", f3(hb and hb["accuracy"]), f3(ha and ha["accuracy"])],
        ["Human fluency AR (1–5)", f3(hb and hb["fluency_ar"]), f3(ha and ha["fluency_ar"])],
        ["Human fluency EN (1–5)", f3(hb and hb["fluency_en"]), f3(ha and ha["fluency_en"])],
        ["Time to first token (Smart AI, avg)", ms(lv(lb, "smart_ai_search", "time_to_first_token")), ms(lv(la, "smart_ai_search", "time_to_first_token"))],
        ["Total latency (Smart AI, avg)", ms(lv(lb, "smart_ai_search", "total")), ms(lv(la, "smart_ai_search", "total"))],
        ["p95 latency (Smart AI)", ms(lv(lb, "smart_ai_search", "total", "p95")), ms(lv(la, "smart_ai_search", "total", "p95"))],
        ["Quick Search p95", ms(lv(lb, "quick_search", "total", "p95")), ms(lv(la, "quick_search", "total", "p95"))],
        ["GPU memory (all processes, MiB)", f3(lb and lb["memory"].get("gpu_used_MiB(all processes)") and int(lb["memory"]["gpu_used_MiB(all processes)"])),
         f3(la and la["memory"].get("gpu_used_MiB(all processes)") and int(la["memory"]["gpu_used_MiB(all processes)"]))],
        ["System RAM in use when measured (GiB of 16)", f3(lb and lb.get("system_ram_used_GiB_at_start")), f3(la and la.get("system_ram_used_GiB_at_start"))],
        ["CPU memory (API process RSS, GiB)", f3(lb and lb["memory"].get("process_rss_GiB")), f3(la and la["memory"].get("process_rss_GiB"))],
    ]
    parts.append("### Before vs after fine-tuning (held-out test set, hybrid + rerank)\n\n"
                 "Before = base e5-base + base gte reranker; after = fine-tuned e5-base + fine-tuned reranker. "
                 "Same LLM, prompts, index and data in both columns.\n\n" + table(["Metric", "Before", "After"], rows))

    # ------------------------------------------------ retrieval comparison
    for name, res in (("after fine-tuning", ra), ("before fine-tuning", rb)):
        if not res:
            continue
        rows = []
        for mode, groups in res["modes"].items():
            for g in ("all", "ar", "en", "mixed", "cross", "cross ar->en", "cross en->ar", "dialect"):
                if g in groups:
                    m = groups[g]
                    rows.append([mode, g, m["n"], f3(m["recall@1"]), f3(m["recall@5"]), f3(m["recall@10"]), f3(m["recall@20"]),
                                 f3(m["hit@5"]), f3(m["mrr"]), f3(m["ndcg@10"])])
        parts.append(f"### Retrieval — {name} (`{res['embedder']}` + `{res['reranker']}`)\n\n" +
                     table(["Mode", "Group", "n", "R@1", "R@5", "R@10", "R@20", "Hit@5", "MRR", "nDCG@10"], rows))
    bo = load("retrieval_zeroshot_bakeoff.json")
    if bo:
        rows = [[k, f3(v["all"]["recall@5"]), f3(v["all"]["mrr"]), f3(v["all"]["ndcg@10"]), f3(v.get("cross", {}).get("recall@5"))]
                for k, v in bo["modes"].items()]
        parts.append("### Zero-shot model bake-off (dense only, test set)\n\n" + table(["Model", "R@5", "MRR", "nDCG@10", "R@5 cross-lingual"], rows))

    # ------------------------------------------------ generation
    for name, res in (("after fine-tuning", ga), ("before fine-tuning", gb)):
        if not res:
            continue
        keys = [k for k in res if not k.startswith("_")]
        cols = ["n", "language_correctness", "abstain_rate", "hallucination_rate", "context_recall", "context_precision",
                "citation_precision", "citation_recall", "numeric_faithfulness", "lexical_support", "answer_relevance",
                "judge_faithful", "bleu", "chrf", "rouge_l"]
        rows = [[k] + [f3(res[k].get(c)) for c in cols] for k in keys]
        parts.append(f"### Generation — {name}\n\n" + table(["Set"] + cols, rows))

    # ------------------------------------------------ latency
    if la:
        hw = la["hardware"]
        rows = []
        for mode, stages in la["per_mode"].items():
            for st, s in stages.items():
                if isinstance(s, dict) and "avg" in s:
                    rows.append([mode, st, s["n"], ms(s["avg"]), ms(s["p50"]), ms(s["p95"]), ms(s["max"])])
        parts.append(f"### Latency (sequential, {hw.get('gpu', 'CPU')}, {hw.get('cpu_threads')} CPU threads)\n\n" +
                     table(["Mode", "Stage", "n", "avg", "p50", "p95", "max"], rows))
        if la.get("corpus_scaling"):
            rows = [[r["chunks"], ms(r["bm25_ms"]["p95"]), ms(r["faiss_flat_ms"]["p95"]), ms(r["faiss_hnsw_ms"]["p95"])] for r in la["corpus_scaling"]]
            parts.append("**Latency vs corpus size** (p95 per query)\n\n" + table(["Chunks", "BM25", "FAISS flat", "FAISS HNSW"], rows))
        if la.get("rerank_topk"):
            rows = [[r["rerank_k"], ms(r["rerank_ms"]["avg"]), ms(r["rerank_ms"]["p95"]), f3(r["recall@5"]), f3(r["mrr"])] for r in la["rerank_topk"]]
            parts.append("**Rerank top-k** (latency and quality)\n\n" + table(["rerank_k", "avg", "p95", "R@5", "MRR"], rows))
        if la.get("context_k"):
            rows = [[r["context_k"], r["avg_prompt_tokens"], ms(r["ttft_ms"]["avg"]), ms(r["ttft_ms"]["p95"])] for r in la["context_k"]]
            parts.append("**Context size vs time to first token**\n\n" + table(["context chunks", "prompt tokens", "TTFT avg", "TTFT p95"], rows))
        if la.get("long_input"):
            li = la["long_input"]
            parts.append(f"**Longest input tested:** {li['query_words']} words (≈{li['approx_tokens']} prompt tokens incl. context), "
                         f"total {ms(li['latency_ms']['total'])}, answered: {li['answered']}.")

    cc = load("concurrency_report.json")
    if cc:
        rows = [[mode, r["concurrency"], r["requests"], ms(r["avg_ms"]), ms(r["p95_ms"]), r["throughput_qps"], f3(r["failure_rate"]),
                 f3(r["gpu_mem_MiB_max"]), f3(r["ram_used_GiB_max"])] for mode, rs in cc["levels"].items() for r in rs]
        parts.append("### Concurrency (through the HTTP API)\n\n" +
                     table(["Mode", "Users", "Requests", "avg", "p95", "QPS", "Failures", "GPU MiB (max)", "RAM GiB (max)"], rows))

    md = "\n\n".join(parts) + "\n"
    R.mkdir(exist_ok=True)
    (R / "summary.md").write_text(md, encoding="utf-8")
    readme = p("README.md")
    txt = readme.read_text(encoding="utf-8")
    txt = re.sub(r"<!-- RESULTS:BEGIN -->.*<!-- RESULTS:END -->", lambda _: "<!-- RESULTS:BEGIN -->\n" + md + "<!-- RESULTS:END -->",
                 txt, flags=re.S)
    readme.write_text(txt, encoding="utf-8")
    # convenience copies with the names the assessment lists
    for src, dst in (("retrieval_before_finetuning.json", "before_finetuning_metrics.json"),
                     ("retrieval_after_finetuning.json", "after_finetuning_metrics.json")):
        if (R / src).exists():
            gen = load(src.replace("retrieval_", "generation_").replace("_finetuning", ""))
            (R / dst).write_text(json.dumps({"retrieval": load(src), "generation": gen}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
