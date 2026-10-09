"""The executor is the second safety layer: it must refuse writes even when the guard is bypassed."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sqlpilot.db import (
    MAX_CELL_CHARS,
    Column,
    ExecutionError,
    PostgresDatabase,
    QueryResult,
    QueryTimeout,
    SQLiteDatabase,
    TableInfo,
    connect,
)


def test_list_and_describe(shop_db: SQLiteDatabase) -> None:
    assert shop_db.list_tables() == ["customers", "orders", "products", "secrets"]
    orders = shop_db.describe_table("ORDERS")
    assert orders.name == "orders"
    assert orders.row_count == 4
    cols = {c.name: c for c in orders.columns}
    assert cols["id"].primary_key is True
    assert cols["quantity"].primary_key is False
    assert cols["customer_id"].references == "customers.id"
    assert cols["product_id"].references == "products.id"
    assert cols["quantity"].references is None
    assert cols["quantity"].type == "INTEGER"


def test_describe_unknown_table(shop_db: SQLiteDatabase) -> None:
    with pytest.raises(KeyError):
        shop_db.describe_table("nope")


def test_schema_is_cached(shop_db: SQLiteDatabase) -> None:
    first = shop_db.schema()
    assert [t.name for t in first] == ["customers", "orders", "products", "secrets"]
    first.clear()
    assert len(shop_db.schema()) == 4


def test_ddl_rendering() -> None:
    table = TableInfo(
        "t",
        (
            Column("id", "INTEGER", primary_key=True),
            Column("u", "INT", references="users.id"),
            Column("note"),
        ),
    )
    assert table.ddl() == "CREATE TABLE t (\n  id INTEGER PRIMARY KEY,\n  u INT REFERENCES users.id,\n  note\n);"
    assert table.to_dict()["columns"][1]["references"] == "users.id"


def test_execute_select(shop_db: SQLiteDatabase) -> None:
    result = shop_db.execute_readonly("SELECT name FROM customers ORDER BY id")
    assert result.columns == ["name"]
    assert result.rows == [["Ada"], ["Linus"], ["Grace"]]
    assert result.row_count == 3
    assert result.truncated is False
    assert result.elapsed_ms >= 0
    assert result.to_dict()["row_count"] == 3


def test_row_limit_boundary(shop_db: SQLiteDatabase) -> None:
    exact = shop_db.execute_readonly("SELECT * FROM orders", max_rows=4)
    assert exact.row_count == 4
    assert exact.truncated is False
    cut = shop_db.execute_readonly("SELECT * FROM orders", max_rows=3)
    assert cut.row_count == 3
    assert cut.truncated is True


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "UPDATE customers SET name = 'x'",
        "INSERT INTO customers VALUES (10, 'x', 'y')",
        "DROP TABLE customers",
        "CREATE TABLE evil (x)",
        "PRAGMA query_only = OFF",
        "ATTACH DATABASE ':memory:' AS evil",
        "SELECT load_extension('evil')",
    ],
)
def test_executor_refuses_writes_even_without_the_guard(shop_db: SQLiteDatabase, shop_path: Path, sql: str) -> None:
    with pytest.raises(ExecutionError):
        shop_db.execute_readonly(sql)
    conn = sqlite3.connect(shop_path)
    assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 3
    conn.close()


def test_executor_resets_authorizer_after_query(shop_db: SQLiteDatabase) -> None:
    with pytest.raises(ExecutionError):
        shop_db.execute_readonly("DELETE FROM customers")
    # introspection uses PRAGMAs, which the authorizer would block if it leaked
    assert shop_db.describe_table("customers").row_count == 3


def test_timeout(shop_db: SQLiteDatabase) -> None:
    sql = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n"
    with pytest.raises(QueryTimeout) as info:
        shop_db.execute_readonly(sql, timeout_s=0.2)
    assert "0.2s" in str(info.value)
    # the connection is still usable afterwards
    assert shop_db.execute_readonly("SELECT 1").rows == [[1]]


def test_execution_error_message(shop_db: SQLiteDatabase) -> None:
    with pytest.raises(ExecutionError) as info:
        shop_db.execute_readonly("SELECT nope FROM customers")
    assert "no such column" in str(info.value)
    assert not isinstance(info.value, QueryTimeout)


def test_cells_are_cleaned(tmp_path: Path) -> None:
    path = tmp_path / "blob.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE b (data BLOB, txt TEXT)")
    conn.execute("INSERT INTO b VALUES (?, ?)", (b"\x00\x01\x02", "x" * (MAX_CELL_CHARS + 50)))
    conn.commit()
    conn.close()
    db = SQLiteDatabase(path)
    row = db.execute_readonly("SELECT data, txt FROM b").rows[0]
    assert row[0] == "<3 bytes>"
    assert len(row[1]) == MAX_CELL_CHARS + 3
    db.close()


def test_invalid_utf8_does_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "latin.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (name TEXT)")
    conn.execute("INSERT INTO t VALUES (CAST(X'e96c6576' AS TEXT))")
    conn.commit()
    conn.close()
    db = SQLiteDatabase(path)
    assert "lev" in db.execute_readonly("SELECT name FROM t").rows[0][0]


def test_preview() -> None:
    result = QueryResult(columns=["a", "b"], rows=[[1, None], [2, "x"], [3, "y"]], truncated=True)
    text = result.preview(max_rows=2)
    assert text.splitlines()[0] == "a | b"
    assert text.splitlines()[1] == "1 | NULL"
    assert "3+ rows total" in text
    assert "rows total" not in QueryResult(columns=["a"], rows=[[1]]).preview()


def test_missing_sqlite_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SQLiteDatabase(tmp_path / "missing.db")


def test_connect_urls(shop_path: Path) -> None:
    assert isinstance(connect(str(shop_path)), SQLiteDatabase)
    db = connect(f"sqlite:///{shop_path}")
    assert isinstance(db, SQLiteDatabase)
    assert db.name == "shop"
    assert connect(str(shop_path), name="custom").name == "custom"
    with pytest.raises(ValueError, match="Unsupported"):
        connect("mysql://localhost/db")
    with pytest.raises(ValueError, match="sqlite:///"):
        connect("sqlite://relative.db")


def test_connect_postgres_url_builds_postgres_database() -> None:
    pytest.importorskip("psycopg")
    db = connect("postgresql://u:p@localhost:1/analytics")
    assert isinstance(db, PostgresDatabase)
    assert db.name == "analytics"
    assert db.dialect == "postgres"
