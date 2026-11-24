"""Add new documents (PDF, DOCX, TXT, HTML, MD) to the Knowledge Bank: chunk, embed, index.

    python scripts/ingest.py --path data/corpus/kb          # (re)index a folder
    python scripts/ingest.py --path new_docs/price_list.pdf  # add one file
    python scripts/ingest.py --path data/corpus/kb --api http://127.0.0.1:8010   # into a running server
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--api", default=None)
    ap.add_argument("--lexical-only", action="store_true")
    args = ap.parse_args()
    if args.api:
        import httpx
        r = httpx.post(args.api.rstrip("/") + "/ingest", json={"path": str(Path(args.path).resolve())}, timeout=600)
        r.raise_for_status()
        print(json.dumps(r.json(), ensure_ascii=False, indent=1))
        return
    from hamsrag.config import load_config, p
    from hamsrag.ingest_service import ingest_paths
    from hamsrag.retrieval import RetrievalEngine
    cfg = load_config("serving_config.yaml")
    try:
        engine = RetrievalEngine(cfg, load_dense=not args.lexical_only, load_reranker=False)
    except FileNotFoundError:  # no dense index yet: create an empty one for this embedder
        from hamsrag.dense import DenseIndex
        engine = RetrievalEngine(cfg, load_dense=False, load_reranker=False)
        if not args.lexical_only:
            engine.dense_index = DenseIndex(cfg["embedder"])
    dense_dir = p(cfg["index_dir"]) / f"dense_{cfg['embedder']}"
    res = ingest_paths([Path(args.path)], engine, chunking=load_config("data_config.yaml")["chunking"],
                       save_dense_to=None if args.lexical_only else dense_dir)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
