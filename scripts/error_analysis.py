"""Build results/error_analysis.md from the generation benchmark items (real examples only).

    python scripts/error_analysis.py --tag after
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hamsrag.config import p  # noqa: E402
from hamsrag.gen_metrics import numbers  # noqa: E402
from hamsrag.qa_data import norm_match  # noqa: E402
from hamsrag.retrieval import chunks_from_jsonl  # noqa: E402
from hamsrag.text import detect_language, script_stats  # noqa: E402

_LAT_WORD = re.compile(r"[A-Za-z]{4,}")


def fmt(it: dict, extra: str = "") -> str:
    lines = [f"* **Q** ({it.get('group')}, `{it['qid']}`): {it['query']}"]
    if it.get("rewritten") and it["rewritten"] != it["query"]:
        lines.append(f"  * rewritten: {it['rewritten']}  (reference: {it.get('reference_rewrite')})")
    lines.append(f"  * **A**: {it['answer'][:400]}")
    if it.get("reference"):
        lines.append(f"  * reference: {it['reference'][:300]}")
    lines.append(f"  * gold: {', '.join(it['gold']) or '—'} · retrieved: {', '.join(it.get('retrieved', [])) or '—'} · cited: {', '.join(it['citations']) or '—'}")
    if extra:
        lines.append(f"  * note: {extra}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="after")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    items = [json.loads(x) for x in p(f"results/generation_items_{args.tag}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    by_id = {c.chunk_id: c for c in chunks_from_jsonl(p("data/processed/chunks.jsonl"))}
    cats: dict[str, list[tuple[dict, str]]] = {k: [] for k in [
        "Wrong document retrieved", "Correct retrieval but unsupported (hallucinated) answer", "Missing or incorrect citations",
        "Language drift (answered in the wrong language)", "Mixed Arabic/English query handled incorrectly",
        "Conversation memory failure (lost reference / wrong rewrite)", "Wrong numbers, dates, prices, names or negation",
        "Untranslated text in the wrong-language output", "'Not found' handled incorrectly (fabricated answer)",
        "False 'not found' (answer was in the retrieved context)"]}
    totals = {k: 0 for k in cats}

    def add(cat, it, note=""):
        totals[cat] += 1
        cats[cat].append((it, note))

    for it in items:
        m = it["metrics"]
        gold = set(it["gold"])
        retrieved = set(it.get("retrieved", []))
        if it["set"] == "unanswerable":
            if not m["abstained"]:
                add("'Not found' handled incorrectly (fabricated answer)", it)
            continue
        if not gold:
            continue
        if not retrieved & gold:
            add("Wrong document retrieved", it, "no gold chunk among the chunks given to the LLM")
        if m["abstained"]:
            if retrieved & gold:
                add("False 'not found' (answer was in the retrieved context)", it)
            continue
        if not m.get("lang_ok", True):
            add("Language drift (answered in the wrong language)", it, f"expected {it['expected_lang']}, got {detect_language(it['answer'])['lang']}")
        if retrieved & gold and m.get("hallucination"):
            add("Correct retrieval but unsupported (hallucinated) answer", it,
                f"numeric faithfulness {m.get('numeric_faithfulness')}, lexical support {m.get('lexical_support')}")
        if m.get("citation_inferred") or (it["citations"] and not set(it["citations"]) & gold and retrieved & gold):
            add("Missing or incorrect citations", it, "no [n] markers in the answer" if m.get("citation_inferred") else "cited chunk is not the gold chunk")
        src_nums = set()
        for c in it["citations"] or it.get("retrieved", []):
            if c in by_id:
                src_nums |= numbers(by_id[c].text)
        bad = numbers(it["answer"]) - src_nums
        if bad:
            add("Wrong numbers, dates, prices, names or negation", it, f"numbers not in the sources: {sorted(bad)}")
        if it.get("group") == "mixed" and (not retrieved & gold or not m.get("lang_ok", True) or m.get("hallucination")):
            add("Mixed Arabic/English query handled incorrectly", it)
        if it["expected_lang"] == "ar":
            ar, en = script_stats(it["answer"])
            latin = [w for w in _LAT_WORD.findall(it["answer"]) if w.lower() not in norm_match(it["query"]).lower()]
            if en >= 4 and len(latin) >= 4:
                add("Untranslated text in the wrong-language output", it, f"{len(latin)} Latin-script words not present in the question: {latin[:6]}")
        if it["set"] == "conversation" and it.get("reference_rewrite") and not retrieved & gold:
            add("Conversation memory failure (lost reference / wrong rewrite)", it)

    n_test = sum(1 for i in items if i["set"] == "test")
    n_conv = sum(1 for i in items if i["set"] == "conversation")
    n_un = sum(1 for i in items if i["set"] == "unanswerable")
    out = [f"# Error analysis — `{args.tag}` configuration", "",
           f"Source: `results/generation_items_{args.tag}.jsonl` ({n_test} test questions, {n_un} unanswerable, {n_conv} conversation turns). "
           "Every example below is copied verbatim from the benchmark output; categories are assigned by the rules in "
           "`scripts/error_analysis.py` (a question can appear in several categories).", "",
           "| Category | Count |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in totals.items()]
    for k, exs in cats.items():
        out += ["", f"## {k} ({totals[k]})", ""]
        out += [fmt(it, note) for it, note in exs[: args.k]] if exs else ["_No cases found in this run._"]
    path = p("results/error_analysis.md")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:20]))


if __name__ == "__main__":
    main()
