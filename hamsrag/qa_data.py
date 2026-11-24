"""Load the authored QA source files and resolve evidence phrases to gold chunk ids."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from .ingest import Chunk
from .text import clean_text, normalize_ar


def norm_match(s: str) -> str:
    s = normalize_ar(clean_text(s)).lower()
    s = re.sub(r"[\"'“”«»()\[\]*]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_facts(qa_dir: Path) -> list[dict]:
    facts: list[dict] = []
    for f in sorted(Path(qa_dir).glob("*.yaml")):
        for i, fact in enumerate(yaml.safe_load(f.read_text(encoding="utf-8")) or []):
            fact["fact_id"] = f"{f.stem}:{i}"
            facts.append(fact)
    return facts


def resolve(facts: list[dict], chunks: list[Chunk]) -> tuple[list[dict], list[str]]:
    """Attach gold chunk ids to each fact. Returns (facts, errors)."""
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    errors: list[str] = []
    for fact in facts:
        if fact.get("unanswerable"):
            fact["gold"] = []
            continue
        ev = norm_match(fact["ev"])
        cands = by_doc.get(fact["doc"], [])
        if not cands:
            errors.append(f"{fact['fact_id']}: unknown doc {fact['doc']}")
            fact["gold"] = []
            continue
        gold = [c.chunk_id for c in cands if ev in norm_match(c.section + " " + c.text)]
        if not gold:  # evidence split across a chunk boundary -> accept chunks holding half of it
            half = ev[: max(12, len(ev) // 2)]
            gold = [c.chunk_id for c in cands if half in norm_match(c.section + " " + c.text)]
        if not gold:
            errors.append(f"{fact['fact_id']}: evidence not found in {fact['doc']}: {fact['ev']}")
        for alt in fact.get("alt", []):  # the same fact stated in another document
            alt_ev = norm_match(alt["ev"])
            hits = [c.chunk_id for c in by_doc.get(alt["doc"], []) if alt_ev in norm_match(c.section + " " + c.text)]
            if not hits:
                errors.append(f"{fact['fact_id']}: alt evidence not found in {alt['doc']}: {alt['ev']}")
            gold += [h for h in hits if h not in gold]
        fact["gold"] = gold
    return facts, errors
