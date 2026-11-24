"""Dense retrieval: fine-tuned bilingual encoder + FAISS inner-product index (cosine on
L2-normalised vectors). Exact search (IndexFlatIP) is used up to `hnsw_threshold` vectors;
above that an HNSW index keeps query latency flat (see benchmark_latency.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ingest import Chunk
from .models import load_embedder


class DenseIndex:
    def __init__(self, embedder_name: str, hnsw_threshold: int = 50_000, device: str | None = None):
        self.embedder_name = embedder_name
        self.model = load_embedder(embedder_name, device)
        self.spec = self.model.spec
        self.hnsw_threshold = hnsw_threshold
        self.index = None
        self.ids: list[str] = []

    # ------------------------------------------------------------------ encoding
    def encode_passages(self, chunks: list[Chunk], batch_size: int = 32) -> np.ndarray:
        texts = [self.spec.get("passage_prefix", "") + c.embed_text() for c in chunks]
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=len(texts) > 500).astype("float32")

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        texts = [self.spec.get("query_prefix", "") + q for q in queries]
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")

    # ------------------------------------------------------------------ index
    def _new_index(self, dim: int, n: int):
        import faiss
        if n >= self.hnsw_threshold:
            idx = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = 80
            idx.hnsw.efSearch = 64
            return idx
        return faiss.IndexFlatIP(dim)

    def build(self, chunks: list[Chunk], vectors: np.ndarray | None = None) -> None:
        vecs = self.encode_passages(chunks) if vectors is None else vectors
        self.index = self._new_index(vecs.shape[1], len(chunks))
        self.index.add(vecs)
        self.ids = [c.chunk_id for c in chunks]

    def add(self, chunks: list[Chunk]) -> None:
        """Incremental ingestion. Replaced chunk ids are rebuilt-out by the caller (rare)."""
        vecs = self.encode_passages(chunks)
        if self.index is None:
            self.index = self._new_index(vecs.shape[1], len(chunks))
        self.index.add(vecs)
        self.ids += [c.chunk_id for c in chunks]

    def search_vectors(self, qv: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        scores, idx = self.index.search(qv, min(k, len(self.ids)))
        return [[(self.ids[i], float(s)) for i, s in zip(row_i, row_s) if i >= 0]
                for row_i, row_s in zip(idx, scores)]

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        return self.search_vectors(self.encode_queries([query]), k)[0]

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        import faiss
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "faiss.index"))
        (d / "ids.json").write_text(json.dumps({"embedder": self.embedder_name, "ids": self.ids}), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, embedder_name: str | None = None, device: str | None = None) -> "DenseIndex":
        import faiss
        d = Path(directory)
        meta = json.loads((d / "ids.json").read_text(encoding="utf-8"))
        name = embedder_name or meta["embedder"]
        if name != meta["embedder"]:
            raise ValueError(f"index built with {meta['embedder']}, requested {name}: rebuild the index")
        obj = cls(name, device=device)
        obj.index = faiss.read_index(str(d / "faiss.index"))
        obj.ids = meta["ids"]
        return obj
