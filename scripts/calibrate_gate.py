"""Calibrate the retrieval abstention gate on VALIDATION data only.

The gate answers "not found" without calling the LLM when the best reranker score is below a
threshold. We pick the threshold that maximises balanced accuracy between validation answerable
queries (should pass) and held-out validation unanswerable questions (should be blocked), with a
safety constraint: at most 5% of answerable validation queries may be blocked.

    python scripts/calibrate_gate.py [--reranker gte-rerank-ft] [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from hamsrag.config import load_config, p  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--max-false-block", type=float, default=0.05)
    ap.add_argument("--write", action="store_true", help="store the threshold in configs/serving_config.yaml")
    ap.add_argument("--tag", default=None, help="results/gate_calibration_<tag>.json")
    args = ap.parse_args()
    from hamsrag.retrieval import RetrievalEngine
    cfg = load_config("serving_config.yaml")
    engine = RetrievalEngine(cfg, embedder=args.embedder, reranker=args.reranker)
    load = lambda f: [json.loads(x) for x in p(f).read_text(encoding="utf-8").splitlines() if x.strip()]  # noqa: E731
    pos = load("data/validation.jsonl")
    neg = load("data/validation_unanswerable.jsonl")

    def top(q):
        hits, _ = engine.retrieve(q)
        return hits[0].sources.get("rerank", hits[0].score) if hits else -1e9

    sp = np.array([top(r["query"]) for r in pos])
    sn = np.array([top(r["query"]) for r in neg])
    cands = np.unique(np.concatenate([sp, sn]))
    best = None
    for t in cands:
        tpr = float((sp >= t).mean())          # answerable passing the gate
        tnr = float((sn < t).mean())           # unanswerable blocked
        if 1 - tpr > args.max_false_block:
            continue
        bal = (tpr + tnr) / 2
        if best is None or bal > best["balanced_accuracy"] or (bal == best["balanced_accuracy"] and t < best["threshold"]):
            best = {"threshold": float(t), "answerable_pass_rate": tpr, "unanswerable_block_rate": tnr, "balanced_accuracy": bal}
    # place the threshold midway to the next lower score for robustness
    if best is not None:
        lower = cands[cands < best["threshold"]]
        if len(lower):
            best["threshold"] = float((best["threshold"] + lower.max()) / 2)
    report = {"embedder": engine.embedder_name, "reranker": engine.reranker_name, "n_answerable": len(sp), "n_unanswerable": len(sn),
              "answerable_scores": {"min": float(sp.min()), "p5": float(np.percentile(sp, 5)), "median": float(np.median(sp))},
              "unanswerable_scores": {"max": float(sn.max()), "p95": float(np.percentile(sn, 95)), "median": float(np.median(sn))},
              "chosen": best}
    p("results").mkdir(exist_ok=True)
    name = f"gate_calibration_{args.tag}.json" if args.tag else "gate_calibration.json"
    (p("results") / name).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    if args.write and best is not None:
        path = p("configs/serving_config.yaml")
        txt = path.read_text(encoding="utf-8")
        import re
        txt = re.sub(r"abstain_threshold: [^\n#]*", f"abstain_threshold: {best['threshold']:.4f} ", txt)
        path.write_text(txt, encoding="utf-8")
        print(f"[written] abstain_threshold = {best['threshold']:.4f}")


if __name__ == "__main__":
    main()
