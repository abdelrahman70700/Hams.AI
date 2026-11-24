import json
from pathlib import Path

from hamsrag.fusion import rrf
from hamsrag.ingest import Chunk
from hamsrag.lexical import LexicalIndex
from hamsrag.metrics import mrr, ndcg_at, score_query


def test_rrf_merges_and_dedups():
    fused = rrf({"bm25": ["a", "b", "c"], "dense": ["b", "d"]}, k=60)
    ids = [c for c, _ in fused]
    assert ids[0] == "b" and len(ids) == len(set(ids)) == 4


def test_metrics():
    assert mrr(["x", "g"], {"g"}) == 0.5
    assert ndcg_at(["g"], {"g"}) == 1.0
    m = score_query(["a", "b", "g"], ["g"])
    assert m["hit@1"] == 0 and m["hit@5"] == 1 and m["recall@5"] == 1


def test_fts5_arabic_english_numbers(tmp_path: Path):
    idx = LexicalIndex(tmp_path / "t.sqlite")
    idx.add([
        Chunk("d1#0", "d1", "سياسة الاسترجاع", "مدة الاسترجاع", 1, "ar", "s", "يحق للعميل استرجاع المنتج خلال ١٤ يوماً", 0),
        Chunk("d2#0", "d2", "SLA", "Targets", None, "en", "s", "Premium tickets: first response 1 hour.", 0),
        Chunk("d3#0", "d3", "التوصيل", "الرسوم", None, "ar", "s", "رسوم التوصيل ٩٩ ريالاً للطلبات الصغيرة", 0),
    ])
    assert idx.search("سياسة الإسترجاع")[0][0] == "d1#0"
    assert idx.search("premium ticket response")[0][0] == "d2#0"
    assert idx.search("14 يوم")[0][0] == "d1#0"           # Western digits find Eastern-digit text
    assert idx.search("الـrefund policy؟ 99")[0][0] == "d3#0"  # numeral match, mixed script
    idx.add([Chunk("d1#0", "d1", "x", "y", 1, "ar", "s", "تم التحديث", 0)])  # re-ingestion replaces
    assert idx.count() == 3 and idx.search("التحديث")[0][0] == "d1#0"


def test_union_keeps_dense_top():
    from hamsrag.fusion import union_candidates
    dense = ["d1", "d2", "d3", "d4"]
    bm25 = ["b1", "b2", "d2", "b3"]
    u = [c for c, _ in union_candidates({"bm25": bm25, "dense": dense}, {"dense": 3, "bm25": 2})]
    assert set(u) == {"d1", "d2", "d3", "b1", "b2"} and len(u) == 5
