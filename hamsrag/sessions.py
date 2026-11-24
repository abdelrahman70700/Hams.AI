"""Self-hosted conversation memory (SQLite)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, created REAL);
CREATE TABLE IF NOT EXISTS turns (
  session_id TEXT, turn INTEGER, query TEXT, rewritten TEXT, lang TEXT, answer TEXT,
  citations TEXT, ts REAL, PRIMARY KEY (session_id, turn));
"""


class SessionStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def new(self) -> str:
        sid = "sess_" + uuid.uuid4().hex[:8]
        with self._lock, self._conn:
            self._conn.execute("INSERT INTO sessions VALUES (?, ?)", (sid, time.time()))
        return sid

    def ensure(self, sid: str | None) -> str:
        if not sid or sid == "new":
            return self.new()
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row:
            with self._lock, self._conn:
                self._conn.execute("INSERT INTO sessions VALUES (?, ?)", (sid, time.time()))
        return sid

    def history(self, sid: str, limit: int = 4) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT turn, query, rewritten, lang, answer, citations FROM turns WHERE session_id=? ORDER BY turn DESC LIMIT ?",
                (sid, limit)).fetchall()
        return [{"turn": r[0], "query": r[1], "rewritten": r[2], "lang": r[3], "answer": r[4],
                 "citations": json.loads(r[5] or "[]")} for r in reversed(rows)]

    def next_turn(self, sid: str) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(MAX(turn), 0) FROM turns WHERE session_id=?", (sid,)).fetchone()
        return row[0] + 1

    def add(self, sid: str, turn: int, query: str, rewritten: str, lang: str, answer: str, citations: list) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO turns VALUES (?,?,?,?,?,?,?,?)",
                               (sid, turn, query, rewritten, lang, answer, json.dumps(citations, ensure_ascii=False), time.time()))
