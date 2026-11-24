"""Pick the fusion strategy and RRF parameters on VALIDATION results and write them to
configs/serving_config.yaml. Inputs: results/retrieval_tune_ft.json (RRF grid, no reranker) and
results/retrieval_val_rrf.json / retrieval_val_union.json (validation, with reranker).
Rule: highest validation nDCG@10 after reranking; ties -> higher recall@5, then the cheaper option (rrf)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hamsrag.config import p  # noqa: E402


def main() -> None:
    res = {}
    for f in ("rrf", "union"):
        path = p(f"results/retrieval_val_{f}.json")
        if path.exists():
            m = json.loads(path.read_text(encoding="utf-8"))["modes"]["hybrid+rerank"]["all"]
            res[f] = (m["ndcg@10"], m["recall@5"], 1 if f == "rrf" else 0)
    choice = max(res, key=lambda k: res[k])
    grid = json.loads(p("results/retrieval_tune_ft.json").read_text(encoding="utf-8")).get("rrf_grid", [])
    best = next((g for g in grid if g["rrf_k"] is not None), None)
    cfg = p("configs/serving_config.yaml")
    txt = cfg.read_text(encoding="utf-8")
    txt = re.sub(r"fusion: \w+", f"fusion: {choice}", txt)
    if best:
        txt = re.sub(r"rrf_k: \d+", f"rrf_k: {best['rrf_k']}", txt)
        txt = re.sub(r"rrf_weights: \{[^}]*\}", f"rrf_weights: {{bm25: 1.0, dense: {best['w_dense']}}}", txt)
    cfg.write_text(txt, encoding="utf-8")
    out = {"validation_hybrid+rerank": res, "chosen_fusion": choice, "rrf_params": best}
    p("results/fusion_selection.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
