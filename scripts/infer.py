"""Run any of the three modes from the command line.

    python scripts/infer.py --mode quick_search --query "سياسة الاسترجاع"
    python scripts/infer.py --mode smart_ai_search --query "What is the SLA for premium support?"
    python scripts/infer.py --mode interactive --session new --query "كم رسوم التركيب؟"
    python scripts/infer.py --mode interactive --session sess_8c1a --query "وإذا كان الدور الرابع؟"

Uses the API if it is running (--api http://127.0.0.1:8010), otherwise loads the models in-process.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["quick_search", "smart_ai_search", "interactive"])
    ap.add_argument("--query", required=True)
    ap.add_argument("--session", default=None, help="'new' or an existing session id (interactive mode)")
    ap.add_argument("--api", default=None, help="use a running API instead of loading models")
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    if args.api:
        import httpx
        r = httpx.post(args.api.rstrip("/") + "/search", timeout=120,
                       json={"mode": args.mode, "query": args.query, "session_id": args.session, "top_k": args.k})
        r.raise_for_status()
        out = r.json()
    else:
        from hamsrag.rag import SearchService
        from hamsrag.retrieval import RetrievalEngine
        quick = args.mode == "quick_search"
        svc = SearchService(engine=RetrievalEngine(load_dense=not quick, load_reranker=not quick))
        if quick:
            out = svc.quick_search(args.query, args.k)
        elif args.mode == "smart_ai_search":
            out = svc.smart_search(args.query)
        else:
            out = svc.interactive(args.query, args.session)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
