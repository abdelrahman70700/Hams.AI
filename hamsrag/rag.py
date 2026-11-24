"""The three search modes on top of one RetrievalEngine and one LLM client.

quick_search()  BM25 only, no LLM, CPU-only capable.
smart_search()  hybrid retrieval -> rerank -> abstain gate -> grounded generation -> citations.
interactive()   session memory -> (gated) query rewrite -> smart_search backbone -> follow-ups.

Every response matches the payloads in the assessment and carries a per-stage latency breakdown.
`*_stream` variants yield ("token", text) events and finish with ("final", payload).
"""
from __future__ import annotations

import math
import re
import threading
import time
from typing import Iterator

from .config import load_config, p
from .llm import Generation, LLMClient
from .prompts import (NOT_FOUND, REWRITE_SYSTEM, SYSTEM, build_rewrite_message, build_user_message)
from .retrieval import Hit, RetrievalEngine
from .sessions import SessionStore
from .text import _DIGITS, clean_text, detect_language

_CITE = re.compile(r"\[(\d+(?:\s*[,،]\s*\d+)*)\]")
_FOLLOW = re.compile(r"\n?\s*FOLLOWUPS?\s*:?\s*", re.I)
# follow-up needs rewriting if short or contains anaphora / continuation markers
_ANAPHORA = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|one|ones|same|also|and|what about|how about)\b|"
    r"^(و|طيب|طب|وهل|وكم|ومتى|وإذا|واذا|وش|وما)|"
    r"(ها|هم|هذا|هذه|ذلك|تلك|نفس|كذلك|أيضا|أيضاً)(\s|$|؟)", re.I)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _relevance(score: float) -> float:
    """Reranker scores are sigmoid probabilities (sentence-transformers default activation);
    raw logits from other rerankers are squashed into [0, 1]."""
    return score if 0.0 <= score <= 1.0 else _sigmoid(score)


def snippet(text: str, query: str, n: int = 220) -> str:
    """Window around the first query-term hit (so Quick Search shows *why* a chunk matched)."""
    from .text import analyze, normalize_ar
    t = text.replace("\n", " ")
    if len(t) <= n:
        return t
    low = normalize_ar(t).lower()
    terms = sorted({w for w in analyze(query) if len(w) > 2}, key=len, reverse=True)
    pos = min((low.find(w) for w in terms if low.find(w) >= 0), default=0)
    start = max(0, pos - n // 3)
    s = t[start:start + n]
    return ("…" if start else "") + s + ("…" if start + n < len(t) else "")


_SENT_END = re.compile(r"[.!?\u061F\n]+")


def _norm_sentence(s: str) -> str:
    return re.sub(r"[\W_]+", " ", re.sub(r"\[\d+\]", "", s)).strip().lower()


def _repeat_cut(text: str) -> int | None:
    """Small models sometimes loop, writing the same sentence again and again.
    Return the index where the first repeated *complete* sentence starts, or None."""
    seen, start = set(), 0
    for m in _SENT_END.finditer(text):
        sent = _norm_sentence(text[start:m.start()])
        # a sentence only counts once its terminator has arrived and it is long enough to be a real sentence
        if len(sent) >= 12:
            if sent in seen:
                return start
            seen.add(sent)
        start = m.end()
    return None


def _echoes_instruction(line: str) -> bool:
    """True if a line is (part of) the follow-up instruction rather than content."""
    from .prompts import FOLLOWUP_INSTR
    t = re.sub(r"\s+", " ", line).strip().rstrip(":").strip()
    if "FOLLOWUPS" in t.upper():
        return True
    return len(t) >= 6 and any(t in instr or t.rstrip("ـ") in instr for instr in FOLLOWUP_INSTR.values())


class SearchService:
    def __init__(self, cfg: dict | None = None, *, engine: RetrievalEngine | None = None, llm: LLMClient | None = None):
        self.cfg = cfg or load_config("serving_config.yaml")
        self.rc = self.cfg["retrieval"]
        self.engine = engine or RetrievalEngine(self.cfg)
        self.llm = llm or LLMClient(self.cfg["llm"])
        self.sessions = SessionStore(p(self.cfg["sessions"]["db"]))
        self.gpu_lock = threading.Lock()  # one GPU model call at a time (embedder/reranker share 4 GB)
        self._rewrite_cache: dict[tuple, str] = {}

    # ================================================================== Quick Search
    def quick_search(self, query: str, k: int | None = None) -> dict:
        t0 = time.perf_counter()
        lang = detect_language(query)
        t1 = time.perf_counter()
        hits, tm = self.engine.lexical(query, k or self.cfg["quick_search"]["k"])
        n = self.cfg["quick_search"]["snippet_chars"]
        results = [{"doc_id": h.chunk.doc_id, "chunk_id": h.chunk.chunk_id, "title": h.chunk.title,
                    "section": h.chunk.section, "snippet": snippet(h.chunk.text, query, n), "score": round(h.score, 4),
                    "source": h.chunk.source, "page": h.chunk.page} for h in hits]
        total = (time.perf_counter() - t0) * 1000
        return {"mode": "quick_search", "query": query, "language_detected": lang["lang"], "results": results,
                "latency_ms": {"preprocessing": round((t1 - t0) * 1000, 2), "search": round(tm["search"], 2),
                               "fetch": round(tm["fetch"], 2), "total": round(total, 2)}}

    # ================================================================== Smart AI Search
    def _retrieve(self, query: str) -> tuple[list[Hit], dict]:
        with self.gpu_lock:
            hits, t1 = self.engine.hybrid(query)
            hits, t2 = self.engine.rerank(query, hits)
        return hits, {"retrieval": t1.get("bm25", 0) + t1.get("embed", 0) + t1.get("dense", 0) + t1.get("fusion", 0),
                      "rerank": t2["rerank"], "detail": {**t1, **t2}}

    def _gate(self, hits: list[Hit]) -> bool:
        """True if retrieval found nothing relevant enough to answer from (skip the LLM)."""
        thr = self.rc.get("abstain_threshold")
        if not hits:
            return True
        return thr is not None and hits[0].sources.get("rerank", hits[0].score) < thr

    def _context(self, hits: list[Hit]) -> list[Hit]:
        return hits[: self.rc["context_k"]]

    @staticmethod
    def _source_text(h: Hit, max_words: int = 180) -> str:
        words = h.chunk.text.split()
        body = " ".join(words[:max_words]) + (" …" if len(words) > max_words else "")
        head = h.chunk.title if h.chunk.section in ("", h.chunk.title) else f"{h.chunk.title} — {h.chunk.section}"
        return f"({head})\n{body}"

    def _messages(self, question: str, ctx: list[Hit], answer_lang: str, followups: bool) -> list[dict]:
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user_message(question, [self._source_text(h) for h in ctx], answer_lang, followups)}]

    @staticmethod
    def _split_followups(text: str) -> tuple[str, list[str]]:
        parts = _FOLLOW.split(text, maxsplit=1)
        # small models sometimes copy fragments of the follow-up instruction itself: drop those lines
        answer = "\n".join(ln for ln in parts[0].strip().splitlines() if not _echoes_instruction(ln)).strip()
        fups = []
        if len(parts) > 1:
            for line in parts[1].splitlines():
                line = re.sub(r"^\s*([-*•]|\d+[.)])\s*", "", line)
                line = re.sub(r"\s*\[[\d٠-٩\s,،]+\]", "", line).strip()  # follow-ups carry no citations
                if len(line) > 5 and not _echoes_instruction(line):
                    fups.append(line)
        return answer, fups[:3]

    def _citations(self, answer: str, ctx: list[Hit], answer_lang: str) -> tuple[str, list[dict], dict]:
        """Parse [n] markers (Eastern digits too), keep valid ones, map to chunks."""
        normalized = answer.translate(_DIGITS)
        used: list[int] = []
        for m in _CITE.finditer(normalized):
            for n in re.split(r"\s*[,،]\s*", m.group(1)):
                i = int(n)
                if 1 <= i <= len(ctx) and i not in used:
                    used.append(i)
        abstained = self.is_not_found(answer, answer_lang)
        inferred = False
        if not used and not abstained and ctx:  # uncited answer: attribute to the top source, flagged
            used, inferred = [1], True
        cites = [{"id": i, "doc_id": ctx[i - 1].chunk.doc_id, "chunk_id": ctx[i - 1].chunk.chunk_id,
                  "title": ctx[i - 1].chunk.title, "snippet": snippet(ctx[i - 1].chunk.text, answer, 160),
                  "page": ctx[i - 1].chunk.page, "source": ctx[i - 1].chunk.source,
                  "relevance": round(_relevance(ctx[i - 1].sources.get("rerank", 0.0)), 3)} for i in used]
        return normalized if normalized != answer else answer, ([] if abstained else cites), \
            {"abstained": abstained, "citation_inferred": inferred}

    _NF_PATTERNS = re.compile(
        r"not found in the knowledge bank|not available in the knowledge bank|"
        r"(do|does|did) not (contain|include|mention|provide|specify|state|cover)|no information (about|on|regarding)|"
        r"(is|are) not (mentioned|specified|provided|included|available) in the (provided )?sources|"
        r"لم أجد هذه المعلومة|لم اجد هذه المعلومة|غير موجودة في قاعدة المعرفة|لا توجد معلومات|لا تحتوي المصادر|"
        r"لا تتضمن المصادر|لم يرد في المصادر|لم تذكر المصادر|غير مذكور", re.I)

    @classmethod
    def is_not_found(cls, answer: str, lang: str | None = None) -> bool:
        """Exact not-found sentence or a paraphrase of it (small LLMs often refuse in their own words)."""
        a = re.sub(r"\[\d+\]", "", clean_text(answer)).strip()
        if any(nf in a for nf in NOT_FOUND.values()):
            return True
        # a refusal phrase in the opening sentence ("The sources do not contain X. [1] mentions Y ...")
        first = re.split(r"(?<=[.!؟?])\s|\n", a, maxsplit=1)[0]
        return bool(cls._NF_PATTERNS.search(first)) or (bool(cls._NF_PATTERNS.search(a)) and len(a.split()) <= 40)

    def smart_search_stream(self, query: str, *, answer_lang: str | None = None, followups: bool = False,
                            mode: str = "smart_ai_search", extra: dict | None = None,
                            question_for_llm: str | None = None) -> Iterator[tuple[str, object]]:
        t0 = time.perf_counter()
        lang = detect_language(query)
        answer_lang = answer_lang or lang["answer_lang"]
        hits, rt = self._retrieve(query)
        t_ret = time.perf_counter()
        latency = {"retrieval": round(rt["retrieval"], 2), "rerank": round(rt["rerank"], 2)}
        base = {"mode": mode, **(extra or {}), "query": (extra or {}).get("query", query),
                "language_detected": lang["lang"], "answer_language": answer_lang}
        if self._gate(hits):
            text = NOT_FOUND[answer_lang]
            yield "token", text
            total = (time.perf_counter() - t0) * 1000
            yield "final", {**base, "answer": text, "citations": [], "not_found": True, "abstained_by": "retrieval_gate",
                            "suggested_followups": [] if followups else None,
                            "latency_ms": {**latency, "time_to_first_token": round(total, 2), "generation_total": 0.0,
                                           "total": round(total, 2)}}
            return
        ctx = self._context(hits)
        msgs = self._messages(question_for_llm or query, ctx, answer_lang, followups)
        gen: Generation | None = None
        shown = []
        in_followups = False
        t_first = None
        stream = self.llm.stream(msgs, max_tokens=self.cfg["llm"]["max_tokens"])
        for piece in stream:
            if isinstance(piece, Generation):
                gen = piece
                break
            if t_first is None:
                t_first = time.perf_counter()
            shown.append(piece)
            if not in_followups:
                cut = _repeat_cut("".join(shown))
                if cut is not None:  # the model started repeating itself: stop generating and keep what came before
                    shown = ["".join(shown)[:cut]]
                    gen = None
                    stream.close()  # closes the HTTP stream, so llama.cpp stops generating
                    break
            if not in_followups and "FOLLOWUP" in "".join(shown[-4:]).upper():
                in_followups = True  # don't stream the follow-up block as part of the answer
            if not in_followups:
                yield "token", piece
        t_gen_end = time.perf_counter()
        answer, fups = self._split_followups(gen.text if gen else "".join(shown))
        cut = _repeat_cut(answer)
        if cut is not None:
            answer = answer[:cut].rstrip()
        answer, cites, flags = self._citations(answer, ctx, answer_lang)
        if flags["abstained"]:
            answer = NOT_FOUND[answer_lang]  # one clear, canonical "not found" response
        ttft = (t_ret - t0) * 1000 + gen.ttft_ms if gen else ((t_first or t_gen_end) - t0) * 1000
        payload = {**base, "answer": answer, "citations": cites, "not_found": flags["abstained"],
                   "citation_inferred": flags["citation_inferred"],
                   "retrieved": [{"chunk_id": h.chunk.chunk_id, "rerank": round(h.sources.get("rerank", 0.0), 3)} for h in ctx],
                   "latency_ms": {**latency, "time_to_first_token": round(ttft, 2),
                                  "generation_total": round(gen.total_ms if gen else (t_gen_end - t_ret) * 1000, 2),
                                  "postprocessing": round((time.perf_counter() - t_gen_end) * 1000, 2),
                                  "total": round((time.perf_counter() - t0) * 1000, 2)},
                   "tokens": {"prompt": gen.prompt_tokens if gen else 0, "completion": gen.completion_tokens if gen else 0}}
        if followups:
            payload["suggested_followups"] = fups
        yield "final", payload

    def smart_search(self, query: str, **kw) -> dict:
        for kind, obj in self.smart_search_stream(query, **kw):
            if kind == "final":
                return obj

    # ================================================================== Interactive AI Search
    def needs_rewrite(self, query: str, history: list[dict]) -> bool:
        return bool(history) and (len(query.split()) <= 7 or bool(_ANAPHORA.search(query.strip())))

    def rewrite(self, query: str, history: list[dict]) -> tuple[str, float]:
        if not self.needs_rewrite(query, history):
            return query, 0.0
        key = (tuple((h["rewritten"], h["answer"][:120]) for h in history[-2:]), query)
        if key in self._rewrite_cache:
            return self._rewrite_cache[key], 0.0
        hist = [(h["rewritten"] or h["query"], re.sub(r"\[\d+\]", "", h["answer"])[:200]) for h in history[-2:]]
        msgs = [{"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": build_rewrite_message(hist, query)}]
        t = time.perf_counter()
        try:
            g = self.llm.chat(msgs, max_tokens=self.cfg["llm"]["rewrite_max_tokens"], temperature=0.0)
            out = g.text.strip().strip('"').splitlines()[0].strip() if g.text.strip() else ""
        except Exception:
            out = ""
        ms = (time.perf_counter() - t) * 1000
        if len(out) < 4:  # fallback: concatenate with the previous standalone question
            out = f"{hist[-1][0]} {query}"
        self._rewrite_cache[key] = out
        return out, ms

    def interactive_stream(self, query: str, session_id: str | None) -> Iterator[tuple[str, object]]:
        sid = self.sessions.ensure(session_id)
        history = self.sessions.history(sid, self.cfg["sessions"]["max_turns_in_prompt"])
        turn = self.sessions.next_turn(sid)
        rewritten, rw_ms = self.rewrite(query, history)
        lang = detect_language(query)  # answer in the language of THIS turn (code-switching)
        final = None
        for kind, obj in self.smart_search_stream(rewritten, answer_lang=lang["answer_lang"], followups=True,
                                                  mode="interactive", question_for_llm=rewritten,
                                                  extra={"session_id": sid, "turn": turn, "query": query,
                                                         "rewritten_query": rewritten}):
            if kind == "final":
                final = obj
            else:
                yield kind, obj
        final["language_detected"] = lang["lang"]
        final["latency_ms"] = {"query_rewrite": round(rw_ms, 2), **final["latency_ms"]}
        final["latency_ms"]["time_to_first_token"] = round(final["latency_ms"]["time_to_first_token"] + rw_ms, 2)
        final["latency_ms"]["total"] = round(final["latency_ms"]["total"] + rw_ms, 2)
        self.sessions.add(sid, turn, query, rewritten, lang["answer_lang"], final["answer"], final["citations"])
        yield "final", final

    def interactive(self, query: str, session_id: str | None) -> dict:
        for kind, obj in self.interactive_stream(query, session_id):
            if kind == "final":
                return obj
