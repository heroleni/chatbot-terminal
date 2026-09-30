"""Adapter: SessionRepositoryPort -> SQLite (data/chats.db).

Chats survive application restarts. Session ids are stable, short and readable
(for example 20261002-4f9c2a) so they can be typed after /resume.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from codeagent.domain.models import Message, SessionInfo

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    provider    TEXT NOT NULL DEFAULT '',
    model       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    position    INTEGER NOT NULL,
    role        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS messages_session_idx ON messages (session_id, position);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SqliteSessionRepository:
    def __init__(self, db_path: str):
        directory = os.path.dirname(os.path.abspath(db_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._path = db_path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ---- SessionRepositoryPort --------------------------------------------
    def create(self, provider: str, model: str) -> str:
        session_id = f"{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
        stamp = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, provider, model, created_at, updated_at) "
                "VALUES (?, '', ?, ?, ?, ?)",
                (session_id, provider, model, stamp, stamp),
            )
        return session_id

    def append(self, session_id: str, message: Message) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 AS next FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            conn.execute(
                "INSERT INTO messages (session_id, position, role, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, row["next"], message.role,
                 json.dumps(message.to_dict(), ensure_ascii=False), _now()),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))

    def load(self, session_id: str) -> list[Message] | None:
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                return None
            rows = conn.execute(
                "SELECT payload FROM messages WHERE session_id = ? ORDER BY position",
                (session_id,),
            ).fetchall()
        return [Message.from_dict(json.loads(r["payload"])) for r in rows]

    def list_sessions(self, limit: int = 20) -> list[SessionInfo]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT s.id, s.title, s.provider, s.model, s.updated_at, "
                "       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS count "
                "FROM sessions s ORDER BY s.updated_at DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        return [
            SessionInfo(id=r["id"], title=r["title"] or "(empty session)",
                        provider=r["provider"], model=r["model"],
                        message_count=r["count"], updated_at=r["updated_at"])
            for r in rows
        ]

    def touch(self, session_id: str, provider: str, model: str, title: str = "") -> None:
        with self._connect() as conn:
            if title:
                conn.execute(
                    "UPDATE sessions SET provider = ?, model = ?, title = ?, updated_at = ? "
                    "WHERE id = ?",
                    (provider, model, title, _now(), session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET provider = ?, model = ?, updated_at = ? WHERE id = ?",
                    (provider, model, _now(), session_id),
                )
