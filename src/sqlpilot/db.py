"""Database access: schema introspection and the read-only executor (second safety layer).

Every query that reaches :meth:`Database.execute_readonly` has already passed
:class:`sqlpilot.guard.SQLGuard`. The executor adds engine-level enforcement on top:

* **SQLite** is opened with ``mode=ro`` (URI), ``PRAGMA query_only=ON`` and an authorizer that
  only permits ``SELECT``/``READ``/``FUNCTION`` actions, so even a parser blind spot cannot write,
  attach a file or load an extension. A progress handler aborts queries over the time limit.
* **Postgres** runs every query in a ``READ ONLY`` transaction with ``SET LOCAL statement_timeout``
  and always rolls back. Point the tenant at a role that only has ``SELECT`` grants as well.

Both fetch at most ``max_rows + 1`` rows, so a result reports ``truncated`` instead of streaming
an unbounded table into memory.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

__all__ = [
    "Column",
    "Database",
    "ExecutionError",
    "PostgresDatabase",
    "QueryResult",
    "QueryTimeout",
    "SQLiteDatabase",
    "TableInfo",
    "connect",
]

MAX_CELL_CHARS = 2000


class ExecutionError(RuntimeError):
    """The database rejected or failed the query. The message is safe to show to the LLM."""


class QueryTimeout(ExecutionError):
    """The query ran longer than the configured limit and was cancelled."""


@dataclass(frozen=True)
class Column:
    name: str
    type: str = ""
    primary_key: bool = False
    references: str | None = None  # "other_table.column" for a foreign key


@dataclass(frozen=True)
class TableInfo:
    name: str
    columns: tuple[Column, ...]
    row_count: int | None = None

    def ddl(self) -> str:
        """A compact CREATE TABLE rendering used in prompts."""
        parts = []
        for col in self.columns:
            piece = f"  {col.name} {col.type}".rstrip()
            if col.primary_key:
                piece += " PRIMARY KEY"
            if col.references:
                piece += f" REFERENCES {col.references}"
            parts.append(piece)
        return f"CREATE TABLE {self.name} (\n" + ",\n".join(parts) + "\n);"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "row_count": self.row_count,
            "columns": [
                {"name": c.name, "type": c.type, "primary_key": c.primary_key, "references": c.references}
                for c in self.columns
            ],
        }


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool = False
    elapsed_ms: float = 0.0
    sql: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "sql": self.sql,
        }

    def preview(self, max_rows: int = 10) -> str:
        """Plain-text table preview for prompts."""
        lines = [" | ".join(self.columns)]
        for row in self.rows[:max_rows]:
            lines.append(" | ".join("NULL" if v is None else str(v) for v in row))
        if self.row_count > max_rows or self.truncated:
            lines.append(f"... ({self.row_count}{'+' if self.truncated else ''} rows total)")
        return "\n".join(lines)


def _clean_cell(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(value))} bytes>"
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[:MAX_CELL_CHARS] + "..."
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    return str(value)


class Database(Protocol):
    """What the agent, the API and the MCP server need from a database."""

    dialect: str
    name: str

    def list_tables(self) -> list[str]: ...

    def describe_table(self, table: str) -> TableInfo: ...

    def schema(self) -> list[TableInfo]: ...

    def execute_readonly(self, sql: str, max_rows: int = 200, timeout_s: float = 10.0) -> QueryResult: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------------------------
# SQLite
# --------------------------------------------------------------------------------------------

_SQLITE_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}
_SQLITE_DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


def _sqlite_authorizer(action: int, arg1: str | None, arg2: str | None, db: str | None, src: str | None) -> int:
    if action not in _SQLITE_ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _SQLITE_DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


class SQLiteDatabase:
    """A SQLite file opened read-only. One connection per thread (sqlite3 objects are per-thread)."""

    dialect = "sqlite"

    def __init__(self, path: str | Path, name: str | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"SQLite database not found: {self.path}")
        self.name = name or self.path.stem
        self._local = threading.local()
        self._schema_cache: list[TableInfo] | None = None
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            uri = f"file:{self.path.resolve()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            conn.execute("PRAGMA query_only = ON")
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            self._local.conn = conn
        return conn

    # ---- introspection (runs without the authorizer: PRAGMAs are needed here) --------------
    def list_tables(self) -> list[str]:
        rows = self._connect().execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [str(r[0]) for r in rows]

    def _resolve(self, table: str) -> str:
        for name in self.list_tables():
            if name.lower() == table.lower():
                return name
        raise KeyError(f"Unknown table: {table}")

    def describe_table(self, table: str) -> TableInfo:
        name = self._resolve(table)
        conn = self._connect()
        quoted = '"' + name.replace('"', '""') + '"'
        fks = {
            str(r[3]): f"{r[2]}.{r[4]}" if r[4] else str(r[2])
            for r in conn.execute(f"PRAGMA foreign_key_list({quoted})")
        }
        columns = tuple(
            Column(
                name=str(r[1]),
                type=str(r[2] or ""),
                primary_key=bool(r[5]),
                references=fks.get(str(r[1])),
            )
            for r in conn.execute(f"PRAGMA table_info({quoted})")
        )
        try:
            count = int(conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
        except sqlite3.Error:
            count = None
        return TableInfo(name=name, columns=columns, row_count=count)

    def schema(self) -> list[TableInfo]:
        with self._lock:
            if self._schema_cache is None:
                self._schema_cache = [self.describe_table(t) for t in self.list_tables()]
            return list(self._schema_cache)

    # ---- execution ---------------------------------------------------------------------
    def execute_readonly(self, sql: str, max_rows: int = 200, timeout_s: float = 10.0) -> QueryResult:
        conn = self._connect()
        deadline = time.monotonic() + timeout_s
        timed_out = False

        def _progress() -> int:
            nonlocal timed_out
            if time.monotonic() > deadline:
                timed_out = True
                return 1
            return 0

        start = time.perf_counter()
        conn.set_authorizer(_sqlite_authorizer)
        conn.set_progress_handler(_progress, 1000)
        try:
            cur = conn.execute(sql)
            rows = cur.fetchmany(max_rows + 1)
            columns = [d[0] for d in cur.description or []]
            cur.close()
        except sqlite3.Error as err:
            if timed_out:
                raise QueryTimeout(f"Query exceeded the {timeout_s:g}s time limit.") from err
            raise ExecutionError(str(err)) from err
        finally:
            conn.set_progress_handler(None, 0)
            conn.set_authorizer(None)
        elapsed = (time.perf_counter() - start) * 1000
        truncated = len(rows) > max_rows
        cleaned = [[_clean_cell(v) for v in row] for row in rows[:max_rows]]
        return QueryResult(columns=columns, rows=cleaned, truncated=truncated, elapsed_ms=elapsed, sql=sql)

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


# --------------------------------------------------------------------------------------------
# Postgres
# --------------------------------------------------------------------------------------------


class PostgresDatabase:
    """Postgres via psycopg 3. Queries run in a READ ONLY transaction that is always rolled back."""

    dialect = "postgres"

    def __init__(self, dsn: str, name: str | None = None, schema_name: str = "public") -> None:
        try:
            import psycopg
        except ImportError as err:  # pragma: no cover - exercised only without the extra
            raise ImportError("Postgres support needs: pip install 'sqlpilot[postgres]'") from err
        self._psycopg = psycopg
        self.dsn = dsn
        self.schema_name = schema_name
        self.name = name or (urlparse(dsn).path.lstrip("/") or "postgres")
        self._local = threading.local()
        self._schema_cache: list[TableInfo] | None = None
        self._lock = threading.Lock()

    def _connect(self) -> Any:
        conn = getattr(self._local, "conn", None)
        if conn is None or conn.closed:
            conn = self._psycopg.connect(self.dsn, autocommit=False)
            self._local.conn = conn
        return conn

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())
        finally:
            conn.rollback()

    def list_tables(self) -> list[str]:
        rows = self._query(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s ORDER BY table_name",
            (self.schema_name,),
        )
        return [str(r[0]) for r in rows]

    def describe_table(self, table: str) -> TableInfo:
        match = [t for t in self.list_tables() if t.lower() == table.lower()]
        if not match:
            raise KeyError(f"Unknown table: {table}")
        name = match[0]
        cols = self._query(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (self.schema_name, name),
        )
        pks = {
            str(r[0])
            for r in self._query(
                "SELECT kcu.column_name FROM information_schema.table_constraints tc "
                "JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name "
                "AND tc.table_schema = kcu.table_schema "
                "WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_schema = %s AND tc.table_name = %s",
                (self.schema_name, name),
            )
        }
        fks = {
            str(r[0]): f"{r[1]}.{r[2]}"
            for r in self._query(
                "SELECT kcu.column_name, ccu.table_name, ccu.column_name "
                "FROM information_schema.table_constraints tc "
                "JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name "
                "AND tc.table_schema = kcu.table_schema "
                "JOIN information_schema.constraint_column_usage ccu "
                "ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema "
                "WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = %s AND tc.table_name = %s",
                (self.schema_name, name),
            )
        }
        count_rows = self._query(
            "SELECT reltuples::bigint FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = %s AND c.relname = %s",
            (self.schema_name, name),
        )
        count = int(count_rows[0][0]) if count_rows and count_rows[0][0] is not None else None
        columns = tuple(
            Column(name=str(c[0]), type=str(c[1]), primary_key=str(c[0]) in pks, references=fks.get(str(c[0])))
            for c in cols
        )
        return TableInfo(name=name, columns=columns, row_count=count if count and count >= 0 else None)

    def schema(self) -> list[TableInfo]:
        with self._lock:
            if self._schema_cache is None:
                self._schema_cache = [self.describe_table(t) for t in self.list_tables()]
            return list(self._schema_cache)

    def execute_readonly(self, sql: str, max_rows: int = 200, timeout_s: float = 10.0) -> QueryResult:
        conn = self._connect()
        start = time.perf_counter()
        try:
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute("SELECT set_config('statement_timeout', %s, true)", (str(int(timeout_s * 1000)),))
                cur.execute(sql)
                rows = cur.fetchmany(max_rows + 1)
                columns = [d.name for d in cur.description or []]
        except self._psycopg.errors.QueryCanceled as err:
            raise QueryTimeout(f"Query exceeded the {timeout_s:g}s time limit.") from err
        except self._psycopg.Error as err:
            raise ExecutionError(str(err).strip()) from err
        finally:
            conn.rollback()
        elapsed = (time.perf_counter() - start) * 1000
        truncated = len(rows) > max_rows
        cleaned = [[_clean_cell(v) for v in row] for row in rows[:max_rows]]
        return QueryResult(columns=columns, rows=cleaned, truncated=truncated, elapsed_ms=elapsed, sql=sql)

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


def connect(url: str, name: str | None = None) -> Database:
    """Open a database from a URL: ``sqlite:///path/to.db``, a bare ``.db``/``.sqlite`` path,
    or ``postgresql://user:pass@host/db``."""
    if url.startswith(("postgres://", "postgresql://")):
        return PostgresDatabase(url, name=name)
    if url.startswith("sqlite:///"):
        return SQLiteDatabase(unquote(url[len("sqlite:///") :]) or ":memory:", name=name)
    if url.startswith("sqlite://"):
        raise ValueError("Use sqlite:///relative/path or sqlite:////absolute/path")
    if "://" in url:
        raise ValueError(f"Unsupported database URL: {url.split('://')[0]}://")
    return SQLiteDatabase(url, name=name)
