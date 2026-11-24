"""FastAPI service exposing the three search modes + ingestion + the web demo.

    python demo/app.py                      # http://127.0.0.1:8010  (UI)  /docs (OpenAPI)
    set HAMS_QUICK_ONLY=1 & python demo/app.py   # CPU-only Quick Search deployment (no GPU models)
"""
from __future__ import annotations

import json
import os
import time
import threading
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from hamsrag.config import load_config, p  # noqa: E402

CFG = load_config("serving_config.yaml")
QUICK_ONLY = os.environ.get("HAMS_QUICK_ONLY") == "1"
app = FastAPI(title="Hams.AI — Bilingual Knowledge Bank Search", version="1.0")
_service = None


def service():
    global _service
    if _service is None:
        from hamsrag.rag import SearchService
        from hamsrag.retrieval import RetrievalEngine
        engine = RetrievalEngine(CFG, load_dense=not QUICK_ONLY, load_reranker=not QUICK_ONLY)
        _service = SearchService(CFG, engine=engine)
    return _service


class SearchRequest(BaseModel):
    mode: Literal["quick_search", "smart_ai_search", "interactive"] = "smart_ai_search"
    query: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    stream: bool = False


class IngestRequest(BaseModel):
    path: str


@app.on_event("startup")
def _warmup() -> None:
    s = service()
    s.quick_search("warmup")
    if not QUICK_ONLY:
        s.engine.hybrid("warm up the encoder")  # first CUDA call allocates kernels


@app.get("/health")
def health() -> dict:
    s = service()
    return {"status": "ok", "chunks": s.engine.lexical_index.count(), "quick_only": QUICK_ONLY,
            "embedder": None if QUICK_ONLY else s.engine.embedder_name,
            "reranker": None if QUICK_ONLY else s.engine.reranker_name, "llm_up": s.llm.health()}


def _sse(gen):
    for kind, obj in gen:
        data = json.dumps(obj if kind == "final" else {"delta": obj}, ensure_ascii=False)
        yield f"event: {kind}\ndata: {data}\n\n"


@app.post("/search")
def search(req: SearchRequest):
    s = service()
    if req.mode == "quick_search":
        return JSONResponse(s.quick_search(req.query, req.top_k))
    if QUICK_ONLY:
        raise HTTPException(400, "this deployment serves quick_search only (HAMS_QUICK_ONLY=1)")
    if req.mode == "smart_ai_search":
        if req.stream:
            return StreamingResponse(_sse(s.smart_search_stream(req.query)), media_type="text/event-stream")
        return JSONResponse(s.smart_search(req.query))
    if req.stream:
        return StreamingResponse(_sse(s.interactive_stream(req.query, req.session_id)), media_type="text/event-stream")
    return JSONResponse(s.interactive(req.query, req.session_id))


@app.post("/session/new")
def new_session() -> dict:
    return {"session_id": service().sessions.new()}


@app.get("/session/{sid}")
def get_session(sid: str) -> dict:
    return {"session_id": sid, "turns": service().sessions.history(sid, limit=100)}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    from hamsrag.ingest_service import ingest_paths
    s = service()
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(404, f"{path} not found")
    dense_dir = p(CFG["index_dir"]) / f"dense_{s.engine.embedder_name}"
    with s.gpu_lock:
        return ingest_paths([path], s.engine, save_dense_to=None if QUICK_ONLY else dense_dir)


# ---------------------------------------------------------------- before vs after fine-tuning
_base_engine = None
_base_lock = threading.Lock()
BASE_EMBEDDER, BASE_RERANKER = "e5-base", "gte-rerank"


def base_engine():
    """The original (not fine-tuned) models, on the CPU so they do not compete with the serving stack for the
    4 GB GPU. Same index, fusion settings and chunking as the fine-tuned stack: only the two models differ."""
    global _base_engine
    with _base_lock:
        if _base_engine is None:
            from hamsrag.retrieval import RetrievalEngine
            _base_engine = RetrievalEngine(CFG, embedder=BASE_EMBEDDER, reranker=BASE_RERANKER, device="cpu")
    return _base_engine


def _gate(name: str) -> float | None:
    if name == "after":
        return CFG["retrieval"].get("abstain_threshold")
    f = p("results/gate_calibration_before.json")
    try:
        return json.loads(f.read_text(encoding="utf-8"))["chosen"]["threshold"]
    except Exception:
        return None


class CompareRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=10)


@app.post("/compare")
def compare(req: CompareRequest) -> dict:
    """Run the same question through the original and the fine-tuned retrieval stack and return both rankings."""
    if QUICK_ONLY:
        raise HTTPException(400, "Comparison needs the neural models; the server runs in Quick Search only mode.")
    from hamsrag.rag import snippet
    s = service()
    out = {"query": req.query}
    for name, eng, lock, dev in (("after", s.engine, s.gpu_lock, "GPU"), ("before", base_engine(), _base_lock, "CPU")):
        t = time.perf_counter()
        with lock:
            hits, timing = eng.retrieve(req.query, top_n=req.k)
        thr = _gate(name)
        top = hits[0].score if hits else 0.0
        out[name] = {
            "embedder": eng.embedder_name, "reranker": eng.reranker_name, "device": dev,
            "ms": round((time.perf_counter() - t) * 1000, 1),
            "would_say_not_found": thr is not None and top < thr, "threshold": thr,
            "hits": [{"rank": i + 1, "chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id, "title": h.chunk.title,
                      "section": h.chunk.section, "lang": h.chunk.lang, "page": h.chunk.page,
                      "source": Path(h.chunk.source).name, "relevance": round(float(h.score), 3),
                      "snippet": snippet(h.chunk.text, req.query)} for i, h in enumerate(hits)],
        }
    return out


@app.get("/documents")
def documents() -> dict:
    """Documents currently in the knowledge bank (for the UI's library panel)."""
    docs: dict[str, dict] = {}
    for c in service().engine.lexical_index.all_chunks():
        d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.title, "lang": c.lang,
                                      "source": Path(c.source).name, "chunks": 0})
        d["chunks"] += 1
    return {"documents": sorted(docs.values(), key=lambda d: d["title"].lower())}


UPLOAD_TYPES = {".pdf", ".docx", ".html", ".htm", ".md", ".markdown", ".txt"}
MAX_UPLOAD_MB = 25


@app.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)) -> dict:
    from hamsrag.ingest_service import ingest_paths
    s = service()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in UPLOAD_TYPES:
        raise HTTPException(415, f"{suffix or 'this file type'} is not supported. Upload PDF, DOCX, HTML, Markdown or TXT.")
    if file.size is not None and file.size > MAX_UPLOAD_MB * 2**20:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
    upload_dir = p("data/corpus/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename).name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    dense_dir = p(CFG["index_dir"]) / f"dense_{s.engine.embedder_name}"
    with s.gpu_lock:
        return ingest_paths([dest], s.engine, save_dense_to=None if QUICK_ONLY else dense_dir)


app.mount("/fonts", StaticFiles(directory=Path(__file__).parent / "web" / "fonts"), name="fonts")


@app.get("/")
def ui():
    # no-store: always serve the current page, never a copy cached by the browser
    return FileResponse(Path(__file__).parent / "web" / "index.html", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HAMS_HOST", "127.0.0.1"), port=int(os.environ.get("HAMS_PORT", "8010")),
                workers=1, log_level="info")
