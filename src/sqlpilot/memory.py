"""Agent memory: per-session conversation history and a store of past successful queries.

Two backends share one schema: :class:`SQLiteMemoryStore` (default, a local file) and
:class:`PostgresMemoryStore` (``sqlpilot[postgres]``). Every read and write is scoped by
``tenant_id``, so tenants never see each other's history or examples.

Past successful (question, SQL) pairs are ranked with BM25 against a new question and handed to
the SQL writer as few-shot examples.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sqlpilot.retrieval import BM25, tokenize

__all__ = ["MemoryStore", "PastQuery", "PostgresMemoryStore", "SQLiteMemoryStore", "open_memory"]

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS sqlpilot_messages (
        id {serial},
        tenant_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at DOUBLE PRECISION NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS sqlpilot_messages_session ON sqlpilot_messages (tenant_id, session_id, id)",
    """CREATE TABLE IF NOT EXISTS sqlpilot_queries (
        id {serial},
        tenant_id TEXT NOT NULL,
        question TEXT NOT NULL,
        sql TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        created_at DOUBLE PRECISION NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS sqlpilot_queries_tenant ON sqlpilot_queries (tenant_id, id)",
]


@dataclass(frozen=True)
class PastQuery:
    question: str
    sql: str
    score: float = 0.0


class MemoryStore(Protocol):
    def add_message(self, tenant_id: str, session_id: str, role: str, content: str) -> None: ...

    def history(self, tenant_id: str, session_id: str, limit: int = 10) -> list[dict[str, str]]: ...

    def record_query(self, tenant_id: str, question: str, sql: str, row_count: int) -> None: ...

    def similar_queries(self, tenant_id: str, question: str, k: int = 2) -> list[PastQuery]: ...

    def close(self) -> None: ...


class _SQLMemory:
    """Shared logic over a DB-API connection. Subclasses set the placeholder and connect."""

    placeholder = "?"
    serial = "INTEGER PRIMARY KEY AUTOINCREMENT"
    scan_limit = 500

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def _conn(self) -> Any:
        raise NotImplementedError

    def _exec(self, sql: str, params: tuple[Any, ...] = (), fetch: bool = False) -> list[tuple[Any, ...]]:
        sql = sql.replace("?", self.placeholder)
        with self._lock:
            conn = self._conn()
            cur = conn.cursor()
            try:
                cur.execute(sql, params)
                rows = list(cur.fetchall()) if fetch else []
                conn.commit()
                return rows
            finally:
                cur.close()

    def _init_schema(self) -> None:
        for stmt in _SCHEMA:
            self._exec(stmt.format(serial=self.serial))

    def add_message(self, tenant_id: str, session_id: str, role: str, content: str) -> None:
        self._exec(
            "INSERT INTO sqlpilot_messages (tenant_id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (tenant_id, session_id, role, content, time.time()),
        )

    def history(self, tenant_id: str, session_id: str, limit: int = 10) -> list[dict[str, str]]:
        rows = self._exec(
            "SELECT role, content FROM sqlpilot_messages WHERE tenant_id = ? AND session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (tenant_id, session_id, limit),
            fetch=True,
        )
        return [{"role": str(r[0]), "content": str(r[1])} for r in reversed(rows)]

    def record_query(self, tenant_id: str, question: str, sql: str, row_count: int) -> None:
        self._exec(
            "INSERT INTO sqlpilot_queries (tenant_id, question, sql, row_count, created_at) VALUES (?, ?, ?, ?, ?)",
            (tenant_id, question, sql, row_count, time.time()),
        )

    def similar_queries(self, tenant_id: str, question: str, k: int = 2) -> list[PastQuery]:
        rows = self._exec(
            "SELECT question, sql FROM sqlpilot_queries WHERE tenant_id = ? ORDER BY id DESC LIMIT ?",
            (tenant_id, self.scan_limit),
            fetch=True,
        )
        if not rows or k <= 0:
            return []
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for q, s in rows:
            key = str(q).strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append((str(q), str(s)))
        scores = BM25([tokenize(q) for q, _ in unique]).scores(tokenize(question))
        ranked = sorted(zip(unique, scores), key=lambda item: -item[1])
        return [PastQuery(q, s, score) for (q, s), score in ranked[:k] if score > 0]


class SQLiteMemoryStore(_SQLMemory):
    def __init__(self, path: str | Path = ":memory:") -> None:
        super().__init__()
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._init_schema()

    def _conn(self) -> Any:
        return self._connection

    def close(self) -> None:
        self._connection.close()


class PostgresMemoryStore(_SQLMemory):
    placeholder = "%s"
    serial = "BIGSERIAL PRIMARY KEY"

    def __init__(self, dsn: str) -> None:
        super().__init__()
        try:
            import psycopg
        except ImportError as err:  # pragma: no cover
            raise ImportError("Postgres memory needs: pip install 'sqlpilot[postgres]'") from err
        self._connection = psycopg.connect(dsn)
        self._init_schema()

    def _conn(self) -> Any:
        return self._connection

    def close(self) -> None:
        self._connection.close()


def open_memory(url: str) -> MemoryStore:
    """``sqlite:///path``, a bare path, ``:memory:``, or ``postgresql://...``."""
    if url.startswith(("postgres://", "postgresql://")):
        return PostgresMemoryStore(url)
    if url.startswith("sqlite:///"):
        return SQLiteMemoryStore(url[len("sqlite:///") :])
    return SQLiteMemoryStore(url)
