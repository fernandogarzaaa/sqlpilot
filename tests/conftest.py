from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sqlpilot.agent import AgentConfig, SQLAgent
from sqlpilot.config import demo_db_path
from sqlpilot.db import SQLiteDatabase
from sqlpilot.llm import MockLLM
from sqlpilot.memory import SQLiteMemoryStore

SHOP_SQL = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT);
CREATE TABLE products (id INTEGER PRIMARY KEY, title TEXT, price REAL);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER
);
CREATE TABLE secrets (id INTEGER PRIMARY KEY, value TEXT);
INSERT INTO customers VALUES (1, 'Ada', 'UK'), (2, 'Linus', 'FI'), (3, 'Grace', 'US');
INSERT INTO products VALUES (1, 'Keyboard', 49.5), (2, 'Mouse', 19.0), (3, 'Monitor', 199.99);
INSERT INTO orders VALUES (1, 1, 1, 2), (2, 1, 2, 1), (3, 2, 3, 1), (4, 3, 2, 5);
INSERT INTO secrets VALUES (1, 'hunter2');
"""


def make_shop_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(SHOP_SQL)
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def shop_path(tmp_path: Path) -> Path:
    return make_shop_db(tmp_path / "shop.db")


@pytest.fixture
def shop_db(shop_path: Path) -> SQLiteDatabase:
    db = SQLiteDatabase(shop_path)
    yield db
    db.close()


@pytest.fixture
def chinook() -> SQLiteDatabase:
    db = SQLiteDatabase(demo_db_path())
    yield db
    db.close()


@pytest.fixture
def memory() -> SQLiteMemoryStore:
    store = SQLiteMemoryStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def mock_agent(chinook: SQLiteDatabase, memory: SQLiteMemoryStore) -> SQLAgent:
    return SQLAgent(chinook, MockLLM(), memory=memory, config=AgentConfig(llm_summary=False))
