"""Fine-tune the bilingual embedding model with contrastive learning.

    python scripts/train_embeddings.py                    # config: configs/training_config.yaml
    python scripts/train_embeddings.py --epochs 1 --dry   # quick smoke test

Method
------
* Base: intfloat/multilingual-e5-base (XLM-R, 278M). The 250k-token word-embedding matrix
  (192M params) is frozen; all 12 transformer layers + layer norms are trained. This cuts the
  trainable state from ~4.5 GB to ~1.4 GB so training fits a 4 GB GPU, and protects the
  multilingual vocabulary from drifting on a small domain dataset.
* Loss: CachedMultipleNegativesRankingLoss (InfoNCE, GradCache). Each (query, positive,
  hard negative) triple is contrasted against every other positive and negative in a batch of
  64, while GPU memory only holds a micro-batch of 16. (Plain gradient accumulation would NOT
  add in-batch negatives.)
* Hard negatives: mined with the base model from TRAINING documents only; candidates scoring
  above 0.95 x the positive's score are discarded as likely false negatives.
* Selection: best checkpoint by validation nDCG@10; seed fixed; config + data hash saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from hamsrag.config import load_config, p  # noqa: E402
from hamsrag.models import embedder_spec  # noqa: E402
from hamsrag.retrieval import chunks_from_jsonl  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def mine_hard_negatives(rows, chunks, miner_name, top_k, margin, per_query, seed):
    """Return one (or more) hard negative chunk per training row."""
    from hamsrag.dense import DenseIndex

    rng = random.Random(seed)
    di = DenseIndex(miner_name)
    di.build(chunks)
    qv = di.encode_queries([r["query"] for r in rows], batch_size=64)
    results = di.search_vectors(qv, top_k)
    pv = {c.chunk_id: v for c, v in zip(chunks, di.encode_passages(chunks))}
    out = []
    dropped, fallback = 0, 0
    for r, res, q in zip(rows, results, qv):
        gold = set(r["gold"])
        pos_score = max(float(q @ pv[g]) for g in gold if g in pv)
        non_gold = [(cid, s) for cid, s in res if cid not in gold]
        cands = []
        for cid, s in non_gold:
            if s > margin * pos_score:  # suspiciously close to the positive: likely unlabeled positive
                dropped += 1
                continue
            cands.append(cid)
        if not cands:  # all filtered: take the least similar of the top-k (still a semi-hard negative)
            fallback += 1
            cands = [non_gold[-1][0]] if non_gold else [c.chunk_id for c in rng.sample(chunks, 3) if c.chunk_id not in gold]
        out.append(cands[:per_query])
    print(f"[mining] candidates dropped as likely false negatives: {dropped}; queries using fallback negative: {fallback}/{len(rows)}")
    release_gpu(di)
    return out, dropped


def release_gpu(*objs) -> None:
    """Free the mining encoder before training (it is held by the load_embedder cache)."""
    import gc

    import torch
    from hamsrag import models
    for o in objs:
        del o
    models.load_embedder.cache_clear()
    models.load_reranker.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--mini-batch", type=int, default=None)
    ap.add_argument("--dry", action="store_true", help="tiny run to check the pipeline end to end")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    tc = load_config("training_config.yaml")
    ec = tc["embedding"]
    seed = tc["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    spec = embedder_spec(ec["base"])
    qp, pp = spec.get("query_prefix", ""), spec.get("passage_prefix", "")
    epochs = args.epochs or ec["epochs"]
    mini = args.mini_batch or ec["mini_batch_size"]

    chunks = chunks_from_jsonl(p("data/processed/chunks.jsonl"))
    by_id = {c.chunk_id: c for c in chunks}
    splits = json.loads(p("data/processed/doc_splits.json").read_text(encoding="utf-8"))
    train_chunks = [c for c in chunks if splits[c.doc_id] == "train"]

    rows = load_jsonl(p("data/train.jsonl"))
    syn_path = p("data/synthetic_train.jsonl")
    if ec.get("use_synthetic") and syn_path.exists():
        syn = load_jsonl(syn_path)
        rows += syn
        print(f"[data] + {len(syn)} synthetic queries")
    if args.dry:
        rows = rows[:96]
    print(f"[data] {len(rows)} training queries, {len(train_chunks)} training chunks")

    t = time.time()
    hn = ec["hard_negatives"]
    negs, dropped = mine_hard_negatives(rows, train_chunks if hn["train_docs_only"] else chunks,
                                        hn["miner"], hn["top_k"], hn["margin"], hn["per_query"], seed)
    print(f"[mining] {time.time() - t:.1f}s, dropped {dropped} likely false negatives")

    anchors, positives, negatives = [], [], []
    for r, ng in zip(rows, negs):
        anchors.append(qp + r["query"])
        positives.append(pp + by_id[r["gold"][0]].embed_text())
        negatives.append(pp + by_id[ng[0]].embed_text())
    train_ds = Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives})

    # validation: IR evaluator over the full corpus (all documents) with validation queries
    val = load_jsonl(p("data/validation.jsonl"))
    evaluator = InformationRetrievalEvaluator(
        queries={r["qid"]: qp + r["query"] for r in val},
        corpus={c.chunk_id: pp + c.embed_text() for c in chunks},
        relevant_docs={r["qid"]: set(r["gold"]) for r in val},
        name="val", ndcg_at_k=[10], accuracy_at_k=[1, 5], precision_recall_at_k=[1, 5, 10],
        mrr_at_k=[10], map_at_k=[10], show_progress_bar=False, batch_size=32)

    model = SentenceTransformer(spec["path"], device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = ec["max_seq_length"]
    emb = model[0].auto_model.embeddings.word_embeddings
    if ec["freeze_word_embeddings"]:
        emb.weight.requires_grad_(False)
        if torch.cuda.is_available() and ec.get("frozen_embeddings_bf16", True):
            emb.weight.data = emb.weight.data.to(torch.bfloat16)  # frozen: no fp32 master copy needed (-0.4 GB)
    n_total = sum(x.numel() for x in model.parameters())
    n_train = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"[model] {spec['path']}: {n_total/1e6:.0f}M params, {n_train/1e6:.0f}M trainable")

    before = evaluator(model)
    print("[val before]", {k: round(v, 4) for k, v in before.items() if "ndcg" in k or "recall@5" in k})

    loss = CachedMultipleNegativesRankingLoss(model, scale=ec["scale"], mini_batch_size=mini)
    out_dir = p(ec["output"])
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir) + "_ckpt",
        num_train_epochs=epochs,
        per_device_train_batch_size=ec["batch_size"],
        learning_rate=ec["learning_rate"],
        warmup_ratio=ec["warmup_ratio"],
        weight_decay=ec["weight_decay"],
        lr_scheduler_type="linear",
        bf16=bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # duplicate positives would be false in-batch negatives
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_val_cosine_ndcg@10", greater_is_better=True,
        logging_steps=5, seed=seed, data_seed=seed, dataloader_num_workers=0, report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=train_ds, loss=loss, evaluator=evaluator)
    t = time.time()
    trainer.train()
    train_s = time.time() - t
    after = evaluator(model)
    print("[val after]", {k: round(v, 4) for k, v in after.items() if "ndcg" in k or "recall@5" in k})
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0

    if not args.dry:
        emb.weight.data = emb.weight.data.float()  # save a standard fp32 checkpoint
        model.save(str(out_dir))
        data_hash = hashlib.sha256("\n".join(anchors + positives + negatives).encode()).hexdigest()[:16]
        report = {"base_model": spec["path"], "epochs": epochs, "train_queries": len(rows), "batch_size": ec["batch_size"],
                  "mini_batch_size": mini, "trainable_params_M": round(n_train / 1e6, 1), "total_params_M": round(n_total / 1e6, 1),
                  "train_seconds": round(train_s, 1), "peak_gpu_mem_GiB": round(peak, 2), "bf16": bf16,
                  "false_negatives_dropped": dropped, "data_sha256_16": data_hash, "config": ec,
                  "validation_before": before, "validation_after": after,
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
        (out_dir / "training_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        p("results").mkdir(exist_ok=True)
        (p("results") / "embedding_training_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[saved] {out_dir}  ({train_s/60:.1f} min, peak GPU {peak:.2f} GiB)")


if __name__ == "__main__":
    main()
