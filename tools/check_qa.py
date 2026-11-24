"""Validate QA source files against the chunked corpus (developer tool)."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hamsrag.ingest import chunk_document, load_document  # noqa: E402
from hamsrag.qa_data import load_facts, resolve  # noqa: E402

root = Path(__file__).resolve().parents[1]
chunks = [c for p in sorted((root / "data/corpus/src").glob("*.md")) for c in chunk_document(load_document(p))]
facts, errors = resolve(load_facts(root / "data/qa_src"), chunks)
print(f"{len(facts)} facts, {sum(len(f.get('qs', [])) for f in facts)} queries, {len(errors)} errors")
print("query types:", Counter(t for f in facts for t, _ in f.get("qs", [])))
print("gold size:", Counter(len(f["gold"]) for f in facts))
docs_with_facts = {f["doc"] for f in facts if not f.get("unanswerable")}
missing = sorted({c.doc_id for c in chunks} - docs_with_facts)
print("docs without facts:", len(missing))
for e in errors:
    print("ERR", e)
