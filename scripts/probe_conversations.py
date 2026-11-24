"""Run candidate multi-turn conversations through the live API (twice each) and report, per turn,
the rewritten question and whether the system answered or said "not found"."""
import json
import sys
import urllib.request

API = "http://127.0.0.1:8010/search"
CONVS = {
    "conv05": ["السرير الذكي يطلع خطأ E05", "وإذا تكرر وصار E07؟", "How do I file the claim?"],
    "conv03": ["كم سعر غرفة نوم ريم؟", "وكم ضمانها؟", "Can I extend it?"],
    "conv09": ["كم يوم إجازة سنوية للموظف؟", "وبعد خمس سنوات؟", "Is there leave for Hajj?"],
    "conv08": ["Do you ship to Kuwait?", "What about Oman?", "ومين يدفع الجمارك للكويت؟"],
    "conv15": ["Where is your Jeddah showroom?", "Is it open on Friday morning?", "وهل عندهم مواقف؟"],
    "conv13": ["أبي أجرب مرتبة قبل ما أشتري", "وإذا ما عجبتني كم يخصمون؟", "What's the warranty on it?"],
}


def post(body):
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


for name, turns in CONVS.items():
    for attempt in (1, 2):
        sid, res = "new", []
        for q in turns:
            d = post({"mode": "interactive", "query": q, "session_id": sid})
            sid = d.get("session_id", sid)
            res.append({"q": q, "rewritten": d.get("rewritten_query"), "not_found": d.get("not_found"),
                        "answer": d.get("answer", "")[:160], "cites": [c.get("title") for c in d.get("citations", [])]})
        ok = all(not r["not_found"] for r in res)
        print(json.dumps({"conv": name, "attempt": attempt, "all_answered": ok, "turns": res}, ensure_ascii=False), flush=True)
