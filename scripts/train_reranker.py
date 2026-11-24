"""Fine-tune the cross-encoder reranker (second stage).

    python scripts/train_reranker.py          # after train_embeddings.py

* Base: Alibaba-NLP/gte-multilingual-reranker-base (306M, Apache-2.0); word embeddings frozen.
* Candidates/negatives: for every training query we run the deployed first stage
  (BM25 + fine-tuned dense -> RRF) restricted to TRAINING documents and keep the top-ranked
  non-gold chunks as hard negatives - the reranker learns to fix exactly the first stage's mistakes.
* Loss: pointwise BinaryCrossEntropy, 1 positive : 7 negatives, positive class up-weighted.
* Validation: CrossEncoderRerankingEvaluator re-ranking the real top-20 hybrid candidates of the
  validation queries; best epoch by MRR@10.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from hamsrag.config import load_config, p  # noqa: E402
from hamsrag.fusion import rrf  # noqa: E402
from hamsrag.lexical import LexicalIndex  # noqa: E402
from hamsrag.models import reranker_spec  # noqa: E402
from hamsrag.retrieval import chunks_from_jsonl  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def release_gpu() -> None:
    import gc

    import torch
    from hamsrag import models
    models.load_embedder.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def first_stage(queries, chunks, lex, embedder, k=30, allowed=None):
    from hamsrag.dense import DenseIndex
    di = DenseIndex(embedder)
    di.build(chunks)
    dense = di.search_vectors(di.encode_queries(queries, batch_size=64), 60)
    out = []
    for q, d in zip(queries, dense):
        b = [c for c, _ in lex.search(q, 60)]
        fused = [c for c, _ in rrf({"bm25": b, "dense": [c for c, _ in d]})]
        if allowed is not None:
            fused = [c for c in fused if c in allowed]
        out.append(fused[:k])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=float, default=None)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--eval-steps", type=int, default=0, help="evaluate every N steps (0 = every epoch)")
    ap.add_argument("--tag", default="", help="suffix for the output directory and report")
    ap.add_argument("--base", default=None, help="reranker key in configs/models.yaml (default: training_config)")
    ap.add_argument("--attn", default=None, help="attention implementation, e.g. eager or sdpa")
    ap.add_argument("--denoise-margin", type=float, default=None,
                    help="drop mined negatives the BASE reranker scores within this many logits of the positive")
    ap.add_argument("--skip-top", type=int, default=0, help="skip the N highest-ranked non-gold candidates")
    ap.add_argument("--negatives", type=int, default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import (CrossEncoder, CrossEncoderTrainer,
                                                     CrossEncoderTrainingArguments)
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    tc = load_config("training_config.yaml")
    rc = tc["reranker"]
    seed = tc["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    spec = reranker_spec(args.base or rc["base"])
    cfg = load_config("serving_config.yaml")
    lex = LexicalIndex(p(cfg["index_dir"]) / "kb.sqlite")

    chunks = chunks_from_jsonl(p("data/processed/chunks.jsonl"))
    by_id = {c.chunk_id: c for c in chunks}
    splits = json.loads(p("data/processed/doc_splits.json").read_text(encoding="utf-8"))
    train_ids = {c.chunk_id for c in chunks if splits[c.doc_id] == "train"}

    rows = load_jsonl(p("data/train.jsonl"))
    syn = p("data/synthetic_train.jsonl")
    if syn.exists():
        rows += load_jsonl(syn)
    rows = [r for r in rows if r.get("source") != "answer_augmentation"]  # rerank real questions only
    if args.dry:
        rows = rows[:40]
    miner = rc["miner_embedder"] if p("models/e5-base-saraya-ft").exists() else "e5-base"
    t = time.time()
    cands = first_stage([r["query"] for r in rows], chunks, lex, miner, k=30, allowed=train_ids)
    print(f"[mining] miner={miner} {time.time() - t:.1f}s")

    n_neg = args.negatives or rc["negatives_per_query"]
    margin = args.denoise_margin if args.denoise_margin is not None else rc.get("denoise_margin")
    base_scores = None
    if margin is not None:  # score candidates with the untouched base reranker (RocketQA-style denoising)
        t = time.time()
        base_ce = CrossEncoder(spec["path"], num_labels=1, max_length=rc["max_length"],
                               trust_remote_code=spec.get("trust_remote_code", False),
                               device="cuda" if torch.cuda.is_available() else "cpu")
        base_scores = []
        for r, cand in zip(rows, cands):
            docs = [r["gold"][0]] + [c for c in cand if c not in set(r["gold"])]
            sc = base_ce.predict([(r["query"], by_id[c].embed_text()) for c in docs], batch_size=16,
                                 activation_fn=torch.nn.Identity(), show_progress_bar=False)
            base_scores.append(dict(zip(docs, [float(x) for x in sc])))
        del base_ce
        release_gpu()
        print(f"[denoise] scored candidates with the base reranker in {time.time() - t:.1f}s")
    dropped_fn = 0
    q_, d_, y_ = [], [], []
    for qi, (r, cand) in enumerate(zip(rows, cands)):
        gold = set(r["gold"])
        negs = [c for c in cand if c not in gold][args.skip_top or rc.get("skip_top", 0):]
        if base_scores is not None:
            pos = base_scores[qi][r["gold"][0]]
            kept = [c for c in negs if base_scores[qi][c] < pos - margin]
            dropped_fn += len(negs) - len(kept)
            if not kept and negs:  # keep at least the clearest negative
                kept = [min(negs, key=lambda c: base_scores[qi][c])]
            negs = kept
        negs = negs[:n_neg]
        for g in r["gold"][:1]:
            q_.append(r["query"]); d_.append(by_id[g].embed_text()); y_.append(1.0)
        for c in negs:
            q_.append(r["query"]); d_.append(by_id[c].embed_text()); y_.append(0.0)
    if base_scores is not None:
        print(f"[denoise] dropped {dropped_fn} candidate negatives as likely false negatives (margin {margin})")
    train_ds = Dataset.from_dict({"query": q_, "document": d_, "label": y_}).shuffle(seed=seed)
    print(f"[data] {len(rows)} queries -> {len(train_ds)} pairs ({sum(y_):.0f} positive)")

    val = load_jsonl(p("data/validation.jsonl"))
    vcands = first_stage([r["query"] for r in val], chunks, lex, miner, k=20)
    samples = [{"query": r["query"], "positive": [by_id[g].embed_text() for g in r["gold"]],
                "documents": [by_id[c].embed_text() for c in cand]} for r, cand in zip(val, vcands)]
    release_gpu()  # the miner encoder is no longer needed on the GPU
    evaluator = CrossEncoderRerankingEvaluator(samples=samples, at_k=10, name="val", always_rerank_positives=False,
                                               batch_size=16, show_progress_bar=False)

    model = CrossEncoder(spec["path"], num_labels=1, max_length=rc["max_length"],
                         trust_remote_code=spec.get("trust_remote_code", False),
                         model_kwargs={"attn_implementation": args.attn} if args.attn else None,
                         device="cuda" if torch.cuda.is_available() else "cpu")
    if rc["freeze_word_embeddings"]:
        for name, prm in model.model.named_parameters():
            if "word_embeddings" in name:
                prm.requires_grad_(False)
    n_total = sum(x.numel() for x in model.model.parameters())
    n_train = sum(x.numel() for x in model.model.parameters() if x.requires_grad)
    print(f"[model] {spec['path']}: {n_total/1e6:.0f}M params, {n_train/1e6:.0f}M trainable")
    before = evaluator(model)
    print("[val before]", before)

    n_pos = sum(y_)
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(float((len(y_) - n_pos) / max(1.0, n_pos))))
    base_out = rc["output"] if not args.base else f"models/{args.base}-saraya-ft"
    out_dir = p(base_out + (f"_{args.tag}" if args.tag else ""))
    bf16 = (not args.no_bf16) and rc.get("bf16", True) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    lr = args.lr or rc["learning_rate"]
    warmup = args.warmup if args.warmup is not None else rc["warmup_ratio"]
    eval_steps = args.eval_steps or rc.get("eval_steps", 0)
    strategy = "steps" if eval_steps else "epoch"
    targs = CrossEncoderTrainingArguments(
        output_dir=str(out_dir) + "_ckpt", num_train_epochs=args.epochs or rc["epochs"],
        per_device_train_batch_size=rc["batch_size"], gradient_accumulation_steps=rc["gradient_accumulation"],
        learning_rate=lr, warmup_ratio=warmup, weight_decay=rc["weight_decay"], max_grad_norm=1.0,
        bf16=bf16, eval_strategy=strategy, save_strategy=strategy, eval_steps=eval_steps or None,
        save_steps=eval_steps or None, save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_val_mrr@10", greater_is_better=True,
        logging_steps=5, seed=seed, dataloader_num_workers=0, report_to="none")
    trainer = CrossEncoderTrainer(model=model, args=targs, train_dataset=train_ds, loss=loss, evaluator=evaluator)
    t = time.time()
    trainer.train()
    train_s = time.time() - t
    after = evaluator(model)
    print("[val after]", after)
    improved = after["val_mrr@10"] > before["val_mrr@10"]
    print(f"[verdict] val MRR@10 {before['val_mrr@10']:.4f} -> {after['val_mrr@10']:.4f} ({'improved' if improved else 'NOT improved'})")
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    if not args.dry:
        model.save(str(out_dir))
        report = {"base_model": spec["path"], "pairs": len(train_ds), "queries": len(rows), "miner": miner,
                  "learning_rate": lr, "warmup_ratio": warmup, "eval_steps": eval_steps, "improved_on_validation": improved,
                  "denoise_margin": margin, "skip_top": args.skip_top or rc.get("skip_top", 0), "negatives_per_query": n_neg, "false_negatives_dropped": dropped_fn,
                  "trainable_params_M": round(n_train / 1e6, 1), "total_params_M": round(n_total / 1e6, 1),
                  "train_seconds": round(train_s, 1), "peak_gpu_mem_GiB": round(peak, 2), "bf16": bf16,
                  "config": rc, "validation_before": before, "validation_after": after,
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
        (out_dir / "training_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        (p("results") / f"reranker_training_report{('_' + args.tag) if args.tag else ''}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"[saved] {out_dir} ({train_s/60:.1f} min, peak GPU {peak:.2f} GiB)")


if __name__ == "__main__":
    main()
