"""Pick the fine-tuned reranker with the best VALIDATION MRR@10 among runs that improved on their
base model, register it as `reranker-ft` in configs/models.yaml and make it the serving reranker.
Writes results/reranker_selection.json (all candidates and the decision)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from hamsrag.config import p  # noqa: E402


def main() -> None:
    cands = []
    for f in sorted(p("models").glob("*/training_report.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if "validation_after" not in r or "val_mrr@10" not in r.get("validation_after", {}):
            continue  # embedding model reports
        cands.append({"dir": str(f.parent.relative_to(p("."))).replace("\\", "/"), "base_model": r["base_model"],
                      "val_mrr_before": round(r["validation_before"]["val_mrr@10"], 4),
                      "val_mrr_after": round(r["validation_after"]["val_mrr@10"], 4),
                      "learning_rate": r.get("learning_rate"), "denoise_margin": r.get("denoise_margin"),
                      "bf16": r.get("bf16"), "improved": r["validation_after"]["val_mrr@10"] > r["validation_before"]["val_mrr@10"]})
    good = [c for c in cands if c["improved"]]
    best = max(good, key=lambda c: c["val_mrr_after"]) if good else None
    out = {"candidates": cands, "chosen": best}
    if best:
        models = yaml.safe_load(p("configs/models.yaml").read_text(encoding="utf-8"))
        base = next((v for v in models["rerankers"].values() if v.get("path") == best["base_model"]), {})
        txt = p("configs/models.yaml").read_text(encoding="utf-8")
        entry = (f"  reranker-ft:                      # selected by scripts/select_reranker.py (best validation MRR@10)\n"
                 f"    path: {best['dir']}\n    max_length: 384\n"
                 + ("    trust_remote_code: true\n" if base.get("trust_remote_code") else ""))
        txt = re.sub(r"  reranker-ft:.*?(?=\n  \S|\Z)", "", txt, flags=re.S)
        txt = txt.replace("rerankers:\n", "rerankers:\n" + entry, 1)
        p("configs/models.yaml").write_text(txt, encoding="utf-8")
        sc = p("configs/serving_config.yaml").read_text(encoding="utf-8")
        sc = re.sub(r"^reranker: \S+", "reranker: reranker-ft", sc, flags=re.M)
        p("configs/serving_config.yaml").write_text(sc, encoding="utf-8")
    p("results").mkdir(exist_ok=True)
    (p("results") / "reranker_selection.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
