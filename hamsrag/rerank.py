"""Cross-encoder reranking (batched, fp16 on GPU)."""
from __future__ import annotations

from .ingest import Chunk
from .models import load_reranker


class Reranker:
    def __init__(self, name: str, batch_size: int = 16, device: str | None = None):
        self.name = name
        self.model = load_reranker(name, device)
        self.batch_size = batch_size

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        pairs = [(query, c.embed_text()) for c in chunks]
        # default activation of a 1-label CrossEncoder = sigmoid -> scores are probabilities in [0, 1]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
        return [float(s) for s in scores]

    def rerank(self, query: str, chunks: list[Chunk], top_n: int | None = None) -> list[tuple[Chunk, float]]:
        scored = sorted(zip(chunks, self.score(query, chunks)), key=lambda x: -x[1])
        return scored[:top_n] if top_n else scored
