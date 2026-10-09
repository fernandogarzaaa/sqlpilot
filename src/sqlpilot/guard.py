"""The SQL safety guard: the first of two layers that keep every query read-only.

The guard parses a candidate statement with sqlglot and only lets through exactly one
read-only query (``SELECT``, ``WITH ... SELECT`` and set operations of selects). Everything
else is rejected with a machine-readable reason code:

* more than one statement (``SELECT 1; DROP TABLE t``), empty input, NUL bytes, oversize input;
* any write or DDL node anywhere in the tree, including data-modifying CTEs
  (``WITH x AS (DELETE ... RETURNING *) SELECT ...``) and ``SELECT ... INTO``;
* engine commands (``PRAGMA``, ``ATTACH``, ``COPY``, ``SET``, transactions, ``GRANT``, ``VACUUM``...);
* row locks (``FOR UPDATE``);
* dangerous functions (``load_extension``, ``pg_sleep``, ``pg_read_file``, ``dblink``,
  ``query_to_xml``, ``nextval``...);
* tables outside an optional per-tenant allow-list, or on the system deny-list.

The second layer is the executor (:mod:`sqlpilot.db`), which opens SQLite read-only with an
authorizer and runs Postgres inside a ``READ ONLY`` transaction with a statement timeout, so a
parser blind spot still cannot write.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

__all__ = [
    "DENIED_FUNCTIONS",
    "DENIED_FUNCTION_PREFIXES",
    "DENIED_TABLES",
    "GuardConfig",
    "GuardRejected",
    "GuardResult",
    "SQLGuard",
    "check_sql",
]

# sqlglot logs a warning when it falls back to a generic Command; we reject those anyway.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

_FORBIDDEN_NODE_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Merge",
    "Create",
    "Drop",
    "Alter",
    "TruncateTable",
    "Command",
    "Pragma",
    "Attach",
    "Detach",
    "Copy",
    "Set",
    "Transaction",
    "Commit",
    "Rollback",
    "Grant",
    "Revoke",
    "Into",
    "Lock",
    "LoadData",
    "Use",
    "Analyze",
)
# Resolved once against the installed sqlglot; names a given version lacks are skipped.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name) for name in _FORBIDDEN_NODE_NAMES if hasattr(exp, name)
)

_ALLOWED_ROOT_NAMES = ("Select", "Union", "Intersect", "Except")
ALLOWED_ROOTS: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name) for name in _ALLOWED_ROOT_NAMES if hasattr(exp, name)
)

DENIED_FUNCTIONS: frozenset[str] = frozenset(
    {
        # SQLite
        "load_extension",
        "readfile",
        "writefile",
        "edit",
        "fts3_tokenizer",
        "sqlite_compileoption_get",
        # Postgres: side effects, file / network access, sleeping, executing strings
        "dblink",
        "dblink_exec",
        "query_to_xml",
        "query_to_xml_and_xmlschema",
        "query_to_xmlschema",
        "cursor_to_xml",
        "table_to_xml",
        "database_to_xml",
        "schema_to_xml",
        "nextval",
        "setval",
        "currval",
        "set_config",
        "current_setting",
        "txid_current",
        "lo_import",
        "lo_export",
        "lo_get",
        "lo_put",
        "lo_from_bytea",
        # MySQL style
        "sleep",
        "benchmark",
        "load_file",
    }
)
DENIED_FUNCTION_PREFIXES: tuple[str, ...] = ("pg_", "lo_", "dblink")

DENIED_TABLES: frozenset[str] = frozenset(
    {"pg_authid", "pg_shadow", "pg_user_mappings", "pg_hba_file_rules", "pg_file_settings", "pg_largeobject"}
)


@dataclass(frozen=True)
class GuardConfig:
    """Knobs for :class:`SQLGuard`. ``allowed_tables`` (lower-case names) is an optional allow-list."""

    dialect: str = "sqlite"
    max_chars: int = 20_000
    allowed_tables: frozenset[str] | None = None
    denied_tables: frozenset[str] = DENIED_TABLES
    denied_functions: frozenset[str] = DENIED_FUNCTIONS
    denied_function_prefixes: tuple[str, ...] = DENIED_FUNCTION_PREFIXES


@dataclass(frozen=True)
class GuardResult:
    """Outcome of a guard check. ``reason`` is a stable code; ``message`` is for humans and LLMs."""

    ok: bool
    sql: str
    reason: str = ""
    message: str = ""
    tables: tuple[str, ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return self.ok


class GuardRejected(ValueError):
    """Raised by callers that want an exception instead of a :class:`GuardResult`."""

    def __init__(self, result: GuardResult) -> None:
        super().__init__(result.message)
        self.result = result


def _reject(sql: str, reason: str, message: str) -> GuardResult:
    return GuardResult(ok=False, sql=sql, reason=reason, message=message)


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return str(node.sql_name()).lower()


def _cte_names(tree: exp.Expression) -> set[str]:
    return {str(cte.alias_or_name).lower() for cte in tree.find_all(exp.CTE)}


def _referenced_tables(tree: exp.Expression) -> list[str]:
    ctes = _cte_names(tree)
    names: list[str] = []
    for table in tree.find_all(exp.Table):
        name = str(table.name).lower()
        if not name or name in ctes or name in names:
            continue
        names.append(name)
    return names


class SQLGuard:
    """Validates that a SQL string is exactly one read-only query. Stateless and thread-safe."""

    def __init__(self, config: GuardConfig | None = None) -> None:
        self.config = config or GuardConfig()

    def check(self, sql: object) -> GuardResult:
        if not isinstance(sql, str):
            return _reject("", "invalid_type", "SQL must be a string.")
        text = sql.strip()
        if not text:
            return _reject(text, "empty", "The SQL statement is empty.")
        if len(text) > self.config.max_chars:
            return _reject(text[:200], "too_long", f"The SQL is longer than {self.config.max_chars} characters.")
        if "\x00" in text:
            return _reject(text, "nul_byte", "The SQL contains a NUL byte.")

        try:
            statements = [s for s in sqlglot.parse(text, read=self.config.dialect) if s is not None]
        except (ParseError, TokenError) as err:
            return _reject(text, "parse_error", f"Could not parse the SQL: {str(err).splitlines()[0]}")
        if not statements:
            return _reject(text, "empty", "The SQL statement is empty.")
        if len(statements) > 1:
            return _reject(text, "multiple_statements", f"Only one statement is allowed, got {len(statements)}.")

        tree = statements[0]
        while isinstance(tree, exp.Subquery) and isinstance(tree.this, exp.Expression):
            tree = tree.this
        if not isinstance(tree, ALLOWED_ROOTS):
            return _reject(
                text,
                "not_select",
                f"Only read-only SELECT queries are allowed (got {type(tree).__name__.upper()}).",
            )

        for node in tree.walk():
            if isinstance(node, FORBIDDEN_NODES):
                return _reject(
                    text,
                    "forbidden_statement",
                    f"{type(node).__name__.upper()} is not allowed in a read-only query.",
                )
            if isinstance(node, exp.Func):
                name = _function_name(node)
                if name in self.config.denied_functions or name.startswith(self.config.denied_function_prefixes):
                    return _reject(text, "forbidden_function", f"The function {name}() is not allowed.")

        tables = _referenced_tables(tree)
        for name in tables:
            if name in self.config.denied_tables:
                return _reject(text, "forbidden_table", f"The table {name} is not allowed.")
            if self.config.allowed_tables is not None and name not in self.config.allowed_tables:
                return _reject(text, "table_not_allowed", f"The table {name} is not in this tenant's allow-list.")

        return GuardResult(ok=True, sql=text.rstrip(";").rstrip(), tables=tuple(tables))


def check_sql(sql: object, dialect: str = "sqlite", allowed_tables: Iterable[str] | None = None) -> GuardResult:
    """Convenience wrapper: ``check_sql("SELECT 1").ok``."""
    allowed = frozenset(t.lower() for t in allowed_tables) if allowed_tables is not None else None
    return SQLGuard(GuardConfig(dialect=dialect, allowed_tables=allowed)).check(sql)
