from __future__ import annotations

from pathlib import Path

import pytest

from sqlpilot.memory import PostgresMemoryStore, SQLiteMemoryStore, open_memory


def test_history_is_ordered_limited_and_scoped(memory: SQLiteMemoryStore) -> None:
    for i in range(5):
        memory.add_message("acme", "s1", "user", f"q{i}")
        memory.add_message("acme", "s1", "assistant", f"a{i}")
    memory.add_message("acme", "s2", "user", "other session")
    memory.add_message("globex", "s1", "user", "other tenant")
    hist = memory.history("acme", "s1", limit=4)
    assert hist == [
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "q4"},
        {"role": "assistant", "content": "a4"},
    ]
    assert memory.history("acme", "s2") == [{"role": "user", "content": "other session"}]
    assert memory.history("globex", "s1") == [{"role": "user", "content": "other tenant"}]
    assert memory.history("nobody", "s1") == []


def test_similar_queries_rank_and_scope(memory: SQLiteMemoryStore) -> None:
    memory.record_query("acme", "How many customers live in Germany?", "SELECT COUNT(*) FROM customers", 1)
    memory.record_query("acme", "Total invoice amount per year", "SELECT 1", 3)
    memory.record_query("globex", "How many customers are in France?", "SELECT 'globex'", 1)
    found = memory.similar_queries("acme", "How many customers are in Brazil?", k=2)
    assert [p.sql for p in found] == ["SELECT COUNT(*) FROM customers"]
    assert found[0].score > 0
    assert all(p.sql != "SELECT 'globex'" for p in found)


def test_similar_queries_k_and_dedupe(memory: SQLiteMemoryStore) -> None:
    for _ in range(3):
        memory.record_query("t", "count the tracks", "SELECT COUNT(*) FROM tracks", 1)
    memory.record_query("t", "longest tracks", "SELECT * FROM tracks ORDER BY ms DESC", 5)
    found = memory.similar_queries("t", "tracks", k=5)
    assert len(found) == 2
    assert memory.similar_queries("t", "tracks", k=1)[0].question in {"count the tracks", "longest tracks"}
    assert len(memory.similar_queries("t", "tracks", k=1)) == 1
    assert memory.similar_queries("t", "tracks", k=0) == []
    assert memory.similar_queries("empty-tenant", "tracks") == []


def test_dedupe_keeps_newest_sql(memory: SQLiteMemoryStore) -> None:
    memory.record_query("t", "Count tracks", "SELECT 1", 1)
    memory.record_query("t", "count tracks ", "SELECT 2", 1)
    found = memory.similar_queries("t", "count tracks")
    assert [p.sql for p in found] == ["SELECT 2"]


def test_scan_limit(memory: SQLiteMemoryStore) -> None:
    memory.scan_limit = 2
    memory.record_query("t", "albums by artist", "SELECT old", 1)
    memory.record_query("t", "x", "SELECT x", 1)
    memory.record_query("t", "y", "SELECT y", 1)
    assert memory.similar_queries("t", "albums by artist") == []


def test_file_store_persists(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "memory.db"
    store = SQLiteMemoryStore(path)
    store.add_message("t", "s", "user", "hello")
    store.close()
    again = SQLiteMemoryStore(path)
    assert again.history("t", "s") == [{"role": "user", "content": "hello"}]
    again.close()


def test_open_memory_urls(tmp_path: Path) -> None:
    assert isinstance(open_memory(":memory:"), SQLiteMemoryStore)
    store = open_memory(f"sqlite:///{tmp_path}/m.db")
    assert isinstance(store, SQLiteMemoryStore)
    assert store.path == f"{tmp_path}/m.db"  # type: ignore[attr-defined]
    assert isinstance(open_memory(str(tmp_path / "plain.db")), SQLiteMemoryStore)


def test_postgres_placeholder_style() -> None:
    assert PostgresMemoryStore.placeholder == "%s"
    assert PostgresMemoryStore.serial.startswith("BIGSERIAL")
    assert SQLiteMemoryStore.placeholder == "?"


def test_open_memory_postgres_url_requires_server() -> None:
    pytest.importorskip("psycopg")
    with pytest.raises(Exception):
        open_memory("postgresql://nobody@127.0.0.1:1/none?connect_timeout=1")
