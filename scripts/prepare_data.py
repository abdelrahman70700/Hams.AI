"""Prepare the knowledge bank and all train / validation / test files.

    python scripts/prepare_data.py            # ingest data/corpus/kb (rendered once, shipped with the repo), split
    python scripts/prepare_data.py --render   # re-render data/corpus/kb from data/corpus/src (needs LibreOffice)

Outputs (data/):
    corpus/kb/**                rendered knowledge-bank documents (PDF / DOCX / HTML / MD)
    processed/chunks.jsonl      every chunk with metadata (doc_id, chunk_id, title, section, page, ...)
    processed/doc_splits.json   which documents are train / validation / test
    train.jsonl                 query -> gold chunk pairs (+ cross-lingual answer augmentation)
    validation.jsonl, test.jsonl
    test_unanswerable.jsonl     questions with no answer in the knowledge bank
    test_conversations.jsonl    multi-turn traces for Interactive AI Search
    processed/dataset_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from hamsrag.config import load_config, p  # noqa: E402
from hamsrag.corpus_build import build  # noqa: E402
from hamsrag.ingest import chunk_document, load_document  # noqa: E402
from hamsrag.qa_data import load_facts, norm_match, resolve  # noqa: E402
from hamsrag.text import detect_language  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def eval_group(qtype: str, qlang: str, doc_lang: str) -> str:
    """One evaluation bucket per query: mixed > cross-lingual > monolingual."""
    if qtype == "mx" or doc_lang == "mixed":
        return "mixed"
    if qlang != doc_lang:
        return "cross"
    return qlang


def split_docs(doc_langs: dict[str, str], cfg: dict, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    by_lang: dict[str, list[str]] = defaultdict(list)
    for d, lang in sorted(doc_langs.items()):
        by_lang[lang].append(d)
    split = {}
    for lang, docs in by_lang.items():
        rng.shuffle(docs)
        n_test = cfg["test_docs"].get(lang, 0)
        n_val = cfg["validation_docs"].get(lang, 0)
        for i, d in enumerate(docs):
            split[d] = "test" if i < n_test else "validation" if i < n_test + n_val else "train"
    return split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="re-render data/corpus/kb from the Markdown sources")
    args = ap.parse_args()
    cfg = load_config("data_config.yaml")
    seed = cfg["seed"]
    proc = p(cfg["processed"])
    proc.mkdir(parents=True, exist_ok=True)

    # 1. Render the knowledge bank from the authored sources
    kb = p(cfg["corpus_kb"])
    if args.render or not kb.exists():
        files = build(p(cfg["corpus_src"]), kb)
        print(f"[render] {len(files)} documents -> {kb}")

    # 2. Ingest the rendered files exactly as the production ingestion would
    files = sorted(f for f in kb.rglob("*") if f.is_file())
    chunks, doc_langs = [], {}
    for f in files:
        doc = load_document(f)
        doc.source = str(f.relative_to(p("data/corpus"))).replace("\\", "/")
        doc_langs[doc.doc_id] = doc.lang
        chunks += chunk_document(doc, **cfg["chunking"])
    write_jsonl(proc / "chunks.jsonl", [c.to_dict() for c in chunks])
    print(f"[ingest] {len(doc_langs)} docs, {len(chunks)} chunks")

    # 3. Resolve evidence phrases against the ingested chunks (also a PDF/DOCX extraction check)
    facts, errors = resolve(load_facts(p(cfg["qa_src"])), chunks)
    for e in errors:
        print("[resolve][ERROR]", e)
    if errors:
        sys.exit(f"{len(errors)} evidence phrases could not be resolved - fix extraction or the QA files")

    # 4. Document-level split
    split = split_docs(doc_langs, cfg["split"], seed)
    (proc / "doc_splits.json").write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")

    rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for fact in facts:
        doc_lang = doc_langs[fact["doc"]]
        s = split[fact["doc"]]
        for i, (qtype, q) in enumerate(fact["qs"]):
            det = detect_language(q)
            qlang = det["lang"]
            answer_lang = det["answer_lang"]
            rows[s].append({
                "qid": f"{fact['fact_id']}:{i}", "query": q, "qtype": qtype, "qlang": qlang,
                "answer_lang": answer_lang, "doc_id": fact["doc"], "doc_lang": doc_lang,
                "group": eval_group(qtype, qlang, doc_lang), "dialect": qtype == "dia",
                "gold": fact["gold"], "answer": fact["a_ar"] if answer_lang == "ar" else fact["a_en"],
                "fact_id": fact["fact_id"], "split": s, "source": "authored",
            })
        if s == "train":  # cross-lingual augmentation: reference answers used as extra queries
            for lang in ("ar", "en"):
                rows["train"].append({
                    "qid": f"{fact['fact_id']}:ans_{lang}", "query": fact[f"a_{lang}"], "qtype": f"answer_{lang}",
                    "qlang": lang, "answer_lang": lang, "doc_id": fact["doc"], "doc_lang": doc_lang,
                    "group": eval_group("x", lang, doc_lang), "dialect": False, "gold": fact["gold"],
                    "answer": fact[f"a_{lang}"], "fact_id": fact["fact_id"], "split": "train",
                    "source": "answer_augmentation",
                })

    # Leakage guard: no test/validation query text may appear in train
    train_q = {norm_match(r["query"]) for r in rows["train"]}
    for s in ("validation", "test"):
        dup = [r["qid"] for r in rows[s] if norm_match(r["query"]) in train_q]
        assert not dup, f"leakage: {s} queries also in train: {dup[:5]}"
        assert not {r["doc_id"] for r in rows[s]} & {r["doc_id"] for r in rows["train"]}, "doc leakage"

    write_jsonl(p("data/train.jsonl"), rows["train"])
    write_jsonl(p("data/validation.jsonl"), rows["validation"])
    write_jsonl(p("data/test.jsonl"), rows["test"])

    # 5. Unanswerable questions + conversations (evaluation only)
    extra = p(cfg["qa_extra"])
    unans = yaml.safe_load((extra / "unanswerable.yaml").read_text(encoding="utf-8"))
    # every 3rd unanswerable question is held out for calibrating the abstention gate (never tested on)
    unans_rows = [{"qid": f"unans:{i}", "query": u["q"], "qtype": u["lang"], "answer_lang": detect_language(u["q"])["answer_lang"],
                   "near_doc": u["near"], "gold": [], "unanswerable": True} for i, u in enumerate(unans)]
    write_jsonl(p("data/validation_unanswerable.jsonl"), [r for i, r in enumerate(unans_rows) if i % 3 == 0])
    write_jsonl(p("data/test_unanswerable.jsonl"), [r for i, r in enumerate(unans_rows) if i % 3 != 0])
    convs = yaml.safe_load((extra / "conversations.yaml").read_text(encoding="utf-8"))
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)
    conv_rows = []
    for conv in convs:
        turns = []
        for t in conv["turns"]:
            gold = [] if t.get("unanswerable") else [
                c.chunk_id for c in by_doc[t["doc"]] if norm_match(t["ev"]) in norm_match(c.section + " " + c.text)]
            assert t.get("unanswerable") or gold, f"conversation evidence not found: {conv['id']} {t['ev']}"
            turns.append({**t, "gold": gold})
        conv_rows.append({"id": conv["id"], "turns": turns})
    write_jsonl(p("data/test_conversations.jsonl"), conv_rows)

    # 6. Statistics for the dataset card
    stats = {
        "documents": dict(Counter(doc_langs.values())), "chunks": len(chunks),
        "docs_per_split": dict(Counter(split.values())),
        "queries_per_split": {s: len(r) for s, r in rows.items()},
        "authored_queries_per_split": {s: sum(r["source"] == "authored" for r in rr) for s, rr in rows.items()},
        "test_groups": dict(Counter(r["group"] for r in rows["test"])),
        "test_dialect_queries": sum(r["dialect"] for r in rows["test"]),
        "unanswerable_test": sum(1 for i in range(len(unans)) if i % 3 != 0),
        "unanswerable_validation": sum(1 for i in range(len(unans)) if i % 3 == 0), "conversations": len(convs),
        "conversation_turns": sum(len(c["turns"]) for c in convs),
    }
    (proc / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
