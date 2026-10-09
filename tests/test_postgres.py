"""Postgres integration (executor + memory). Runs in CI against the Postgres service container.

Set SQLPILOT_TEST_PG_URL=postgresql://user:pass@host:5432/db to run locally.
"""

from __future__ import annotations

import os
import uuid

import pytest

PG_URL = os.environ.get("SQLPILOT_TEST_PG_URL")
pytestmark = [pytest.mark.postgres, pytest.mark.skipif(not PG_URL, reason="SQLPILOT_TEST_PG_URL not set")]


@pytest.fixture
def pg_schema():  # type: ignore[no-untyped-def]
    psycopg = pytest.importorskip("psycopg")
    schema = f"t_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(PG_URL, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA {schema}")
        conn.execute(f"CREATE TABLE {schema}.customers (id INT PRIMARY KEY, name TEXT)")
        conn.execute(
            f"CREATE TABLE {schema}.orders (id INT PRIMARY KEY, customer_id INT REFERENCES "
            f"{schema}.customers(id), total NUMERIC)"
        )
        conn.execute(f"INSERT INTO {schema}.customers VALUES (1, 'Ada'), (2, 'Linus')")
        conn.execute(f"INSERT INTO {schema}.orders VALUES (1, 1, 10.5), (2, 1, 3), (3, 2, 7)")
    yield schema
    with psycopg.connect(PG_URL, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")


def test_introspection(pg_schema: str) -> None:
    from sqlpilot.db import PostgresDatabase

    db = PostgresDatabase(PG_URL, schema_name=pg_schema)  # type: ignore[arg-type]
    assert db.list_tables() == ["customers", "orders"]
    orders = db.describe_table("ORDERS")
    cols = {c.name: c for c in orders.columns}
    assert cols["id"].primary_key
    assert cols["customer_id"].references == "customers.id"
    db.close()


def test_readonly_execution(pg_schema: str) -> None:
    from sqlpilot.db import ExecutionError, PostgresDatabase, QueryTimeout

    db = PostgresDatabase(PG_URL, schema_name=pg_schema)  # type: ignore[arg-type]
    res = db.execute_readonly(f"SELECT name FROM {pg_schema}.customers ORDER BY id", max_rows=1)
    assert res.rows == [["Ada"]] and res.truncated
    with pytest.raises(ExecutionError, match="read-only"):
        db.execute_readonly(f"DELETE FROM {pg_schema}.customers")
    with pytest.raises(QueryTimeout):
        db.execute_readonly("SELECT pg_sleep(2)", timeout_s=0.2)
    assert db.execute_readonly(f"SELECT COUNT(*) FROM {pg_schema}.customers").rows == [[2]]
    db.close()


def test_agent_on_postgres(pg_schema: str) -> None:
    from sqlpilot.agent import AgentConfig, SQLAgent
    from sqlpilot.db import PostgresDatabase
    from sqlpilot.llm import ScriptedLLM

    db = PostgresDatabase(PG_URL, schema_name=pg_schema)  # type: ignore[arg-type]
    llm = ScriptedLLM([f"SELECT SUM(total) FROM {pg_schema}.orders"])
    result = SQLAgent(db, llm, config=AgentConfig(llm_summary=False)).ask("total of all orders")
    assert result.ok
    assert float(result.result["rows"][0][0]) == 20.5  # type: ignore[index]


def test_postgres_memory_store() -> None:
    from sqlpilot.memory import open_memory

    store = open_memory(PG_URL)  # type: ignore[arg-type]
    tenant = f"t-{uuid.uuid4().hex[:6]}"
    store.add_message(tenant, "s", "user", "hello")
    store.record_query(tenant, "count customers", "SELECT COUNT(*) FROM customers", 1)
    assert store.history(tenant, "s") == [{"role": "user", "content": "hello"}]
    assert store.similar_queries(tenant, "customers count")[0].sql == "SELECT COUNT(*) FROM customers"
    assert store.similar_queries("other-" + tenant, "customers count") == []
    store.close()
