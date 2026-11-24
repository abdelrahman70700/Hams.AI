from hamsrag.ingest import Chunk
from hamsrag.llm import Generation
from hamsrag.rag import SearchService
from hamsrag.retrieval import Hit

CHUNKS = [Chunk("sla#1", "sla", "SLA", "Targets", 3, "en", "kb/sla.pdf", "Tier: Premium; First response: 1 hour; Resolution target: 8 business hours", 1),
          Chunk("ret#0", "ret", "سياسة الاسترجاع", "مدة الاسترجاع", 1, "ar", "kb/r.pdf", "يحق للعميل استرجاع المنتج خلال 14 يوما", 0)]


class FakeEngine:
    def __init__(self, score=5.0):
        self.score = score
    def hybrid(self, q):
        return [Hit(c, 0.1, {"rrf": 0.1}) for c in CHUNKS], {"bm25": 1.0, "embed": 2.0, "dense": 0.5, "fusion": 0.1}
    def rerank(self, q, hits, top_n=None):
        return [Hit(h.chunk, self.score - i, {**h.sources, "rerank": self.score - i}) for i, h in enumerate(hits)], {"rerank": 3.0}
    def lexical(self, q, k):
        return [Hit(CHUNKS[1], 4.2, {})], {"search": 1.0, "fetch": 0.2}


class FakeLLM:
    def __init__(self, text):
        self.text = text
        self.calls = []
    def stream(self, msgs, **kw):
        self.calls.append(msgs)
        for w in self.text.split(" "):
            yield w + " "
        yield Generation(self.text, 50.0, 120.0, 20, 300)
    def chat(self, msgs, **kw):
        self.calls.append(msgs)
        return Generation("What is the SLA for Enterprise tickets?", 10, 30)


CFG = {"retrieval": {"context_k": 2, "abstain_threshold": 0.0}, "llm": {"max_tokens": 100, "rewrite_max_tokens": 40},
       "sessions": {"db": ":memory:", "max_turns_in_prompt": 4}, "quick_search": {"k": 5, "snippet_chars": 100}}


def svc(text, score=5.0, tmp=None):
    s = SearchService(CFG, engine=FakeEngine(score), llm=FakeLLM(text))
    return s


def test_smart_search_citations_and_latency():
    out = svc("Premium tickets get a first response within 1 hour [1].").smart_search("What is the SLA for premium support?")
    assert out["mode"] == "smart_ai_search" and out["language_detected"] == "en"
    assert out["citations"][0]["chunk_id"] == "sla#1" and out["citations"][0]["page"] == 3
    assert set(out["latency_ms"]) >= {"retrieval", "rerank", "time_to_first_token", "generation_total", "total"}


def test_eastern_digit_citations_and_followups():
    out = svc("مدة الاسترجاع 14 يوما [٢]. FOLLOWUPS:\n- هل يمكن الاستبدال؟\n- متى يعاد المبلغ؟").smart_search(
        "كم مدة الاسترجاع؟", followups=True)
    assert out["citations"][0]["chunk_id"] == "ret#0"
    assert out["suggested_followups"] == ["هل يمكن الاستبدال؟", "متى يعاد المبلغ؟"]
    assert "FOLLOWUP" not in out["answer"]


def test_retrieval_gate_abstains_without_llm():
    s = svc("should not be called", score=-3.0)
    out = s.smart_search("What is the maternity leave?")
    assert out["not_found"] and out["abstained_by"] == "retrieval_gate" and not s.llm.calls


def test_llm_abstention_detected():
    out = svc("Sorry, this information was not found in the knowledge bank.").smart_search("Who is the CEO?")
    assert out["not_found"] and out["citations"] == []


def test_interactive_rewrite_and_session():
    s = svc("Enterprise P1 first response is 15 minutes [1].")
    a = s.interactive("What is the SLA for premium support tickets?", "new")
    assert a["turn"] == 1 and a["rewritten_query"] == a["query"]
    b = s.interactive("And for Enterprise?", a["session_id"])
    assert b["turn"] == 2 and b["rewritten_query"] == "What is the SLA for Enterprise tickets?"
    assert "query_rewrite" in b["latency_ms"]
    c = s.interactive("وكم التعويض؟", a["session_id"])
    assert c["language_detected"] == "ar" and c["answer_language"] == "ar"


def test_quick_search_payload():
    out = svc("x").quick_search("سياسة الاسترجاع")
    r = out["results"][0]
    assert out["mode"] == "quick_search" and r["chunk_id"] == "ret#0" and r["page"] == 1 and "total" in out["latency_ms"]


def test_followup_instruction_echo_is_removed():
    from hamsrag.rag import SearchService
    raw = ("بعد إكمال 5 سنوات متصلة في الخدمة، تزيد الإجازة السنوية إلى 30 يوماً [1].\n"
           "بعد الإجابة اكتب سطراً يبدأ ب\nFOLLOWUPS:\nثم سؤالين قصيرين للمتابعة باللغة العربية تجيب عنهما المصادر، كل سؤال في سطر.\n"
           "هل يمكن ترحيل الإجازة؟")
    answer, fups = SearchService._split_followups(raw)
    assert answer.endswith("30 يوماً [1].")
    assert fups == ["هل يمكن ترحيل الإجازة؟"]


def test_repeat_cut_stops_a_looping_answer():
    from hamsrag.rag import _repeat_cut
    a = "Employees get 21 days of paid leave [2].\nExhibition staff follow a separate policy [1].\n"
    looped = a + "Exhibition staff follow a separate policy [1].\nExhibition staff"
    assert _repeat_cut(a) is None
    assert looped[:_repeat_cut(looped)].strip() == a.strip()
    ar = "يحق للموظف إجازة سنوية مدتها 21 يوماً [2]. تتبع إجازات موظفي المعارض سياسة خاصة [1]. تتبع إجازات موظفي المعارض سياسة خاصة [1]."
    assert ar[:_repeat_cut(ar)].count("المعارض") == 1
