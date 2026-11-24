"""Quick Search: BM25 over an SQLite FTS5 inverted index (CPU-only, persistent, incremental).

Text is analysed by ``hamsrag.text.analyze`` (Arabic normalisation + light stemming, English
Snowball stemming, numeral unification) *before* it reaches SQLite, and stored as space-joined
tokens. FTS5's own tokenizer then only splits on whitespace, so index-time and query-time
analysis are guaranteed identical. Two columns are indexed - the heading (title + section) and
the body - with BM25 column weights so that heading matches count a bit more.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from .ingest import Chunk
from .text import analyze

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    doc_id TEXT NOT NULL,
    title TEXT, section TEXT, page INTEGER, lang TEXT, source TEXT,
    text TEXT NOT NULL, ord INTEGER
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    head, body, tokenize = "unicode61 remove_diacritics 0 tokenchars '.'"
);
"""
HEAD_WEIGHT, BODY_WEIGHT = 1.5, 1.0


class LexicalIndex:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conn().executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        # one connection per thread (FastAPI runs sync endpoints in a thread pool)
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA mmap_size=268435456")
            self._local.conn = c
        return c

    # ------------------------------------------------------------------ write
    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    def add(self, chunks: list[Chunk]) -> int:
        con = self._conn()
        with con:
            for c in chunks:
                old = con.execute("SELECT rowid FROM chunks WHERE chunk_id=?", (c.chunk_id,)).fetchone()
                if old:  # re-ingestion replaces the chunk
                    con.execute("DELETE FROM chunks_fts WHERE rowid=?", old)
                    con.execute("DELETE FROM chunks WHERE rowid=?", old)
                cur = con.execute(
                    "INSERT INTO chunks(chunk_id, doc_id, title, section, page, lang, source, text, ord)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (c.chunk_id, c.doc_id, c.title, c.section, c.page, c.lang, c.source, c.text, c.ord))
                head = " ".join(analyze(f"{c.title} {c.section}"))
                body = " ".join(analyze(c.text))
                con.execute("INSERT INTO chunks_fts(rowid, head, body) VALUES (?,?,?)", (cur.lastrowid, head, body))
        return len(chunks)

    def delete_doc(self, doc_id: str) -> int:
        con = self._conn()
        with con:
            rows = con.execute("SELECT rowid FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()
            con.executemany("DELETE FROM chunks_fts WHERE rowid=?", rows)
            con.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        return len(rows)

    def optimize(self) -> None:
        con = self._conn()
        con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        con.commit()

    # ------------------------------------------------------------------ read
    def count(self) -> int:
        return self._conn().execute("SELECT count(*) FROM chunks").fetchone()[0]

    def all_chunks(self) -> list[Chunk]:
        rows = self._conn().execute(
            "SELECT chunk_id, doc_id, title, section, page, lang, source, text, ord FROM chunks ORDER BY rowid").fetchall()
        return [Chunk(*r) for r in rows]

    def get(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        q = ",".join("?" * len(chunk_ids))
        rows = self._conn().execute(
            f"SELECT chunk_id, doc_id, title, section, page, lang, source, text, ord FROM chunks WHERE chunk_id IN ({q})",
            chunk_ids).fetchall()
        return {r[0]: Chunk(*r) for r in rows}

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        """Return [(chunk_id, bm25_score)] best first; scores are positive (higher = better)."""
        toks = list(dict.fromkeys(analyze(query)))  # unique, order kept
        if not toks:
            return []
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in toks)
        rows = self._conn().execute(
            f"SELECT c.chunk_id, -bm25(chunks_fts, {HEAD_WEIGHT}, {BODY_WEIGHT}) AS s "
            "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts, ?, ?) LIMIT ?",
            (match, HEAD_WEIGHT, BODY_WEIGHT, k)).fetchall()
        return [(cid, float(s)) for cid, s in rows]
