"""Ranking metrics with possibly multiple relevant chunks per query."""
from __future__ import annotations

import math
from collections import defaultdict


def recall_at(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def hit_at(ranked: list[str], gold: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & gold else 0.0


def mrr(ranked: list[str], gold: set[str], k: int = 100) -> float:
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked: list[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(ranked[:k], start=1) if cid in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


KS = (1, 5, 10, 20)


def score_query(ranked: list[str], gold: list[str]) -> dict[str, float]:
    g = set(gold)
    out = {f"recall@{k}": recall_at(ranked, g, k) for k in KS}
    out.update({f"hit@{k}": hit_at(ranked, g, k) for k in KS})
    out["mrr"] = mrr(ranked, g)
    out["ndcg@10"] = ndcg_at(ranked, g, 10)
    return out


def aggregate(per_query: list[dict], group_key: str = "group") -> dict[str, dict[str, float]]:
    """Average metrics overall and per group (ar / en / mixed / cross [/ dialect])."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in per_query:
        buckets["all"].append(r)
        buckets[r[group_key]].append(r)
        if r.get("dialect"):
            buckets["dialect"].append(r)
    out = {}
    for name, rows in buckets.items():
        keys = [k for k in rows[0]["metrics"]]
        out[name] = {k: round(sum(r["metrics"][k] for r in rows) / len(rows), 4) for k in keys}
        out[name]["n"] = len(rows)
    return out
