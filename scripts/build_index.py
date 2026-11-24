"""Build the keyword index (Quick Search) and the dense vector index (Smart AI / Interactive).

    python scripts/build_index.py                       # embedder from configs/serving_config.yaml
    python scripts/build_index.py --embedder e5-base    # e.g. the base model for comparisons
    python scripts/build_index.py --lexical-only        # CPU-only Quick Search deployment
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hamsrag.config import load_config, p  # noqa: E402
from hamsrag.lexical import LexicalIndex  # noqa: E402
from hamsrag.retrieval import chunks_from_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--lexical-only", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="delete the existing keyword index first")
    args = ap.parse_args()
    cfg = load_config("serving_config.yaml")
    index_dir = p(cfg["index_dir"])
    chunks = chunks_from_jsonl(p(args.chunks))

    db = index_dir / "kb.sqlite"
    if args.fresh and db.exists():
        db.unlink()
    t = time.perf_counter()
    lex = LexicalIndex(db)
    lex.add(chunks)
    lex.optimize()
    print(f"[bm25] {lex.count()} chunks indexed in {time.perf_counter() - t:.2f}s -> {db}")

    if args.lexical_only:
        return
    from hamsrag.dense import DenseIndex
    name = args.embedder or cfg["embedder"]
    t = time.perf_counter()
    dense = DenseIndex(name)
    dense.build(lex.all_chunks())
    dense.save(index_dir / f"dense_{name}")
    print(f"[dense] {len(dense.ids)} vectors ({name}) in {time.perf_counter() - t:.2f}s -> {index_dir / f'dense_{name}'}")


if __name__ == "__main__":
    main()
