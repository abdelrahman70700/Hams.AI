"""Reciprocal Rank Fusion (Cormack et al., 2009), optionally weighted per retriever."""
from __future__ import annotations


def rrf(rankings: dict[str, list[str]], k: int = 60, weights: dict[str, float] | None = None,
        top_n: int | None = None) -> list[tuple[str, float]]:
    """rankings: {"bm25": [chunk_id, ...], "dense": [...]} best-first.

    score(d) = sum_r w_r / (k + rank_r(d)), rank starting at 1. A chunk returned by both
    retrievers simply accumulates both contributions, which also de-duplicates the lists.
    """
    weights = weights or {}
    scores: dict[str, float] = {}
    for name, ranked in rankings.items():
        w = weights.get(name, 1.0)
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused[:top_n] if top_n else fused


def union_candidates(rankings: dict[str, list[str]], quotas: dict[str, int], k: int = 60,
                     weights: dict[str, float] | None = None) -> list[tuple[str, float]]:
    """Hybrid candidate generation for a cross-encoder: take the top `quotas[name]` items of each
    retriever (in quota order, so the first retriever's top items are always kept), de-duplicate,
    then order the union by weighted RRF score. Unlike pure RRF truncation, one retriever's
    confident misses can never push the other retriever's top candidates out of the reranked set."""
    chosen: list[str] = []
    for name, n in quotas.items():
        added = 0
        for cid in rankings.get(name, []):
            if added >= n:
                break
            if cid not in chosen:
                chosen.append(cid)
                added += 1
    scores = dict(rrf(rankings, k=k, weights=weights))
    return sorted(((c, scores.get(c, 0.0)) for c in chosen), key=lambda x: -x[1])
