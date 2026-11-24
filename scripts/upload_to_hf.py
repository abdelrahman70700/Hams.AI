"""Upload the two fine-tuned models to the Hugging Face Hub, each with a model card.

Log in first in this terminal (the token is stored by huggingface_hub, never in this repo):
    tools\\hf_login.bat      (asks for the token; stored in .cache\\hf, never in this repo)
Then:
    .env\\python.exe scripts\\upload_to_hf.py --user <your-hf-username> [--private]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hamsrag.config import load_config, p  # noqa: E402

CARD = """---
language: [ar, en]
license: {license}
base_model: {base}
library_name: sentence-transformers
tags: [{tags}, arabic, bilingual, retrieval, rag]
---

# {name}

Fine-tuned from [`{base}`](https://huggingface.co/{base}) for Arabic (MSA) / English / mixed and
cross-lingual retrieval over an enterprise knowledge bank, as part of the Hams.AI self-hosted RAG assessment.

{usage}

## Training
{training}

## Held-out test results (10 unseen documents, 138 questions; RTX 3050 Laptop)
{results}

Full details, including what went wrong and how it was fixed: `model_card.md` in the project repository.
Trained on a small synthetic corpus: expect smaller gains on very different domains.
"""

EMB_USAGE = """```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("{repo}")
q = m.encode(["query: ما هي مدة الضمان؟"], normalize_embeddings=True)
d = m.encode(["passage: The warranty period is 5 years."], normalize_embeddings=True)
print(q @ d.T)
```
Keep the e5 prefixes `query: ` / `passage: `."""

RR_USAGE = """```python
from sentence_transformers import CrossEncoder
m = CrossEncoder("{repo}", trust_remote_code=True)
print(m.predict([("ما هي مدة الضمان؟", "The warranty period is 5 years.")]))  # sigmoid relevance
```"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write the cards to results/hf_cards/ only")
    ap.add_argument("--export", default=None, help="copy each model + its model card into <dir>/<repo-name> (no upload)")
    args = ap.parse_args()
    from huggingface_hub import HfApi
    models = load_config("models.yaml")
    emb, rr = models["embedders"]["e5-base-ft"], models["rerankers"]["reranker-ft"]
    ra = json.loads(p("results/retrieval_after_finetuning.json").read_text(encoding="utf-8"))["modes"]
    rb = json.loads(p("results/retrieval_before_finetuning.json").read_text(encoding="utf-8"))["modes"]

    def row(mode, label, metric, group="all"):
        return f"| {label} | {rb[mode][group][metric]:.3f} | {ra[mode][group][metric]:.3f} |"

    jobs = [
        ("hams-e5-base-ar-en-retrieval", emb, "mit", "intfloat/multilingual-e5-base", "sentence-transformers, feature-extraction",
         EMB_USAGE,
         "Contrastive fine-tuning (CachedMultipleNegativesRankingLoss, batch 64, 4 epochs, lr 2e-5) on 475 "
         "query–passage rows with mined hard negatives; word-embedding matrix frozen.",
         "Dense retrieval only:\n\n| Metric | Base | Fine-tuned |\n|---|---|---|\n" + "\n".join([
             row("dense", "Recall@5", "recall@5"), row("dense", "nDCG@10", "ndcg@10"),
             row("dense", "Recall@5 cross-lingual", "recall@5", "cross")])),
        ("hams-gte-reranker-ar-en", rr, "apache-2.0", "Alibaba-NLP/gte-multilingual-reranker-base", "cross-encoder, text-ranking",
         RR_USAGE,
         "Binary cross-entropy, lr 2e-6, fp32, 2 epochs, 4 denoised hard negatives per query, word embeddings frozen.",
         "Full hybrid pipeline (BM25 + fine-tuned e5 → RRF → this reranker), base vs fine-tuned pipeline:\n\n"
         "| Metric | Base | Fine-tuned |\n|---|---|---|\n" + "\n".join([
             row("hybrid+rerank", "Recall@1", "recall@1"), row("hybrid+rerank", "MRR", "mrr"),
             row("hybrid+rerank", "nDCG@10", "ndcg@10")])),
    ]
    api = HfApi()
    if args.export:
        out = Path(args.export)
        for name, spec, lic, base, tags, usage, training, results in jobs:
            src = p(spec["path"]) if not Path(spec["path"]).is_absolute() else Path(spec["path"])
            dst = out / name
            shutil.copytree(src, dst, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("checkpoint-*", "optimizer.pt", "*.pth", "runs"))
            card = CARD.format(license=lic, base=base, tags=tags, name=name, usage=usage.format(repo=f"{args.user}/{name}"),
                               training=training, results=results)
            (dst / "README.md").write_text(card, encoding="utf-8")
            print("[export]", dst)
        return
    if not args.dry_run:
        print("[hf] logged in as", api.whoami()["name"])
    for name, spec, lic, base, tags, usage, training, results in jobs:
        repo = f"{args.user}/{name}"
        src = p(spec["path"]) if not Path(spec["path"]).is_absolute() else Path(spec["path"])
        card = CARD.format(license=lic, base=base, tags=tags, name=name, usage=usage.format(repo=repo),
                           training=training, results=results)
        if args.dry_run:
            out = p("results/hf_cards"); out.mkdir(parents=True, exist_ok=True)
            (out / f"{name}.md").write_text(card, encoding="utf-8")
            print("[dry-run]", repo, "<-", src)
            continue
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            stage = Path(td) / name
            shutil.copytree(src, stage, ignore=shutil.ignore_patterns("checkpoint-*", "optimizer.pt", "*.pth", "runs"))
            (stage / "README.md").write_text(card, encoding="utf-8")
            api.create_repo(repo, private=args.private, exist_ok=True)
            api.upload_folder(folder_path=str(stage), repo_id=repo, commit_message="Upload fine-tuned model")
        print("[uploaded]", f"https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
