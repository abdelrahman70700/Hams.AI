"""Add new documents to a live knowledge bank: load -> chunk -> BM25 index -> dense index."""
from __future__ import annotations

import time
from pathlib import Path

from .ingest import LOADERS, chunk_document, load_document


def ingest_paths(paths: list[Path], engine, *, chunking: dict | None = None, save_dense_to: Path | None = None) -> dict:
    t0 = time.perf_counter()
    files = []
    for path in paths:
        path = Path(path)
        files += [f for f in path.rglob("*") if f.suffix.lower() in LOADERS] if path.is_dir() else [path]
    added, docs, errors = [], [], []
    for f in files:
        try:
            doc = load_document(f)
            chunks = chunk_document(doc, **(chunking or {}))
            replaced = engine.lexical_index.delete_doc(doc.doc_id)
            engine.lexical_index.add(chunks)
            added += chunks
            docs.append({"doc_id": doc.doc_id, "title": doc.title, "lang": doc.lang, "chunks": len(chunks),
                         "replaced_chunks": replaced, "source": doc.source})
        except Exception as e:  # keep going; report per-file errors
            errors.append({"file": str(f), "error": f"{type(e).__name__}: {e}"})
    if added and engine.dense_index is not None:
        # simplest correct strategy for replaced docs: rebuild vectors from the keyword store
        if any(d["replaced_chunks"] for d in docs):
            engine.dense_index.build(engine.lexical_index.all_chunks())
        else:
            engine.dense_index.add(added)
        if save_dense_to:
            engine.dense_index.save(save_dense_to)
    return {"documents": docs, "chunks_added": len(added), "errors": errors,
            "total_chunks": engine.lexical_index.count(), "seconds": round(time.perf_counter() - t0, 2)}
