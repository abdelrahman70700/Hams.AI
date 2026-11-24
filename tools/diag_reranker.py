"""Diagnose the GTE reranker collapse: compare logits for identical pairs scored alone vs. in a padded
batch, in eval vs. train mode (dropout disabled), fp32 vs. bf16, sdpa vs. eager attention."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from hamsrag.config import p  # noqa: E402
from hamsrag.retrieval import chunks_from_jsonl  # noqa: E402

chunks = chunks_from_jsonl(p("data/processed/chunks.jsonl"))
val = [json.loads(x) for x in p("data/validation.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()][:6]
by_id = {c.chunk_id: c for c in chunks}
pairs = []
for r in val:
    pairs.append((r["query"], by_id[r["gold"][0]].embed_text()))           # positive
    pairs.append((r["query"], chunks[(hash(r["qid"]) % len(chunks))].embed_text()))  # random negative


def run(attn: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = "Alibaba-NLP/gte-multilingual-reranker-base"
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name, attn_implementation=attn, dtype=torch.float32).cuda()
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0

    def logits(batch, train, bf16):
        m.train(train)
        enc = tok([a for a, _ in batch], [b for _, b in batch], padding=True, truncation=True, max_length=384, return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            return m(**enc).logits.float().view(-1).cpu()

    res = {}
    single = torch.cat([logits([pr], False, False) for pr in pairs])
    for train in (False, True):
        for bf16 in (False, True):
            batched = logits(pairs, train, bf16)
            res[f"train={train},bf16={bf16}"] = {
                "max_abs_diff_vs_single_eval_fp32": round(float((batched - single).abs().max()), 4),
                "pos_minus_neg_mean": round(float((batched[0::2] - batched[1::2]).mean()), 3)}
    res["single_eval_fp32_pos_minus_neg_mean"] = round(float((single[0::2] - single[1::2]).mean()), 3)
    del m
    torch.cuda.empty_cache()
    return res


out = {}
for attn in ("sdpa", "eager"):
    try:
        out[attn] = run(attn)
    except Exception as e:  # noqa: BLE001
        out[attn] = f"{type(e).__name__}: {e}"
print(json.dumps(out, indent=1))
p("results").mkdir(exist_ok=True)
(p("results") / "diag_reranker.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
