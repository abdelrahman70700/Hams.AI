"""The single retrieval backbone shared by all three search modes.

    lexical()  -> BM25 only                          (Quick Search)
    hybrid()   -> BM25 + dense -> weighted RRF        (Smart AI / Interactive)
    rerank()   -> cross-encoder on fused candidates    (Smart AI / Interactive)

Every call returns results plus a timing breakdown so the API can report per-stage latency.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import load_config, p
from .dense import DenseIndex
from .fusion import rrf, union_candidates
from .ingest import Chunk
from .lexical import LexicalIndex


@dataclass
class Hit:
    chunk: Chunk
    score: float
    sources: dict = field(default_factory=dict)  # e.g. {"bm25": rank, "dense": rank, "rrf": score}


class RetrievalEngine:
    def __init__(self, cfg: dict | None = None, *, embedder: str | None = None, reranker: str | None = None,
                 load_dense: bool = True, load_reranker: bool = True, device: str | None = None):
        self.cfg = cfg or load_config("serving_config.yaml")
        self.rcfg = self.cfg["retrieval"]
        index_dir = p(self.cfg["index_dir"])
        self.lexical_index = LexicalIndex(index_dir / "kb.sqlite")
        self.embedder_name = embedder or self.cfg["embedder"]
        self.reranker_name = reranker or self.cfg["reranker"]
        self.dense_index: DenseIndex | None = None
        self.reranker = None
        if load_dense:
            dense_dir = index_dir / f"dense_{self.embedder_name}"
            self.dense_index = DenseIndex.load(dense_dir, device=device) if (dense_dir / "faiss.index").exists() else None
            if self.dense_index is None:
                raise FileNotFoundError(f"no dense index at {dense_dir}; run scripts/build_index.py")
        if load_reranker and self.reranker_name:
            from .rerank import Reranker
            self.reranker = Reranker(self.reranker_name, device=device)

    # ------------------------------------------------------------------ stages
    def lexical(self, query: str, k: int | None = None) -> tuple[list[Hit], dict]:
        t0 = time.perf_counter()
        res = self.lexical_index.search(query, k or self.cfg["quick_search"]["k"])
        t1 = time.perf_counter()
        chunks = self.lexical_index.get([cid for cid, _ in res])
        hits = [Hit(chunks[cid], s, {"bm25": i + 1}) for i, (cid, s) in enumerate(res) if cid in chunks]
        return hits, {"search": (t1 - t0) * 1000, "fetch": (time.perf_counter() - t1) * 1000}

    def hybrid(self, query: str, *, bm25_k: int | None = None, dense_k: int | None = None,
               top_n: int | None = None, weights: dict | None = None, rrf_k: int | None = None,
               use_bm25: bool = True, use_dense: bool = True) -> tuple[list[Hit], dict]:
        timing: dict[str, float] = {}
        rankings: dict[str, list[str]] = {}
        if use_bm25:
            t = time.perf_counter()
            rankings["bm25"] = [cid for cid, _ in self.lexical_index.search(query, bm25_k or self.rcfg["bm25_k"])]
            timing["bm25"] = (time.perf_counter() - t) * 1000
        if use_dense:
            t = time.perf_counter()
            qv = self.dense_index.encode_queries([query])
            timing["embed"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            rankings["dense"] = [cid for cid, _ in self.dense_index.search_vectors(qv, dense_k or self.rcfg["dense_k"])[0]]
            timing["dense"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        n = top_n or self.rcfg["rerank_k"]
        w = weights or self.rcfg.get("rrf_weights")
        if self.rcfg.get("fusion", "rrf") == "union" and len(rankings) == 2:
            nb = self.rcfg.get("union_bm25", 5)
            fused = union_candidates(rankings, {"dense": n - nb, "bm25": nb}, k=rrf_k or self.rcfg["rrf_k"], weights=w)
        else:
            fused = rrf(rankings, k=rrf_k or self.rcfg["rrf_k"], weights=w, top_n=n)
        chunks = self.lexical_index.get([cid for cid, _ in fused])
        ranks = {name: {cid: i + 1 for i, cid in enumerate(r)} for name, r in rankings.items()}
        hits = [Hit(chunks[cid], s, {**{n: ranks[n].get(cid) for n in ranks}, "rrf": s})
                for cid, s in fused if cid in chunks]
        timing["fusion"] = (time.perf_counter() - t) * 1000
        return hits, timing

    def rerank(self, query: str, hits: list[Hit], top_n: int | None = None) -> tuple[list[Hit], dict]:
        if not self.reranker or not hits:
            return hits[: top_n or len(hits)], {"rerank": 0.0}
        t = time.perf_counter()
        scored = self.reranker.rerank(query, [h.chunk for h in hits])
        by_id = {h.chunk.chunk_id: h for h in hits}
        out = [Hit(c, s, {**by_id[c.chunk_id].sources, "rerank": s}) for c, s in scored]
        return out[: top_n or len(out)], {"rerank": (time.perf_counter() - t) * 1000}

    def retrieve(self, query: str, *, top_n: int | None = None) -> tuple[list[Hit], dict]:
        """Full Smart-AI retrieval: hybrid -> rerank."""
        hits, t1 = self.hybrid(query)
        hits, t2 = self.rerank(query, hits, top_n=top_n)
        return hits, {**t1, **t2}


def chunks_from_jsonl(path: str | Path) -> list[Chunk]:
    import json
    return [Chunk(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
