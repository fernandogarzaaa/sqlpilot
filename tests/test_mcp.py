from __future__ import annotations

import json

import pytest

from sqlpilot.agent import AgentConfig, SQLAgent
from sqlpilot.db import SQLiteDatabase
from sqlpilot.llm import MockLLM
from sqlpilot.mcp_server import CAPABILITIES, SQLPilotTools, build_server


@pytest.fixture
def tools(chinook: SQLiteDatabase) -> SQLPilotTools:
    return SQLPilotTools(SQLAgent(chinook, MockLLM(), config=AgentConfig(llm_summary=False, max_rows=100)))


def test_list_and_describe(tools: SQLPilotTools) -> None:
    listing = tools.list_tables()
    assert listing["dialect"] == "sqlite"
    assert {"name": "Artist", "row_count": 275, "columns": 2} in listing["tables"]
    assert tools.describe_table("artist")["name"] == "Artist"
    assert tools.describe_table("nope")["error"] == "unknown_table"


def test_run_readonly_sql(tools: SQLPilotTools) -> None:
    assert tools.run_readonly_sql("SELECT COUNT(*) FROM Track")["rows"] == [[3503]]
    assert tools.run_readonly_sql("SELECT * FROM Track")["row_count"] == 50  # default_rows
    assert tools.run_readonly_sql("SELECT * FROM Track", max_rows=5)["row_count"] == 5
    assert tools.run_readonly_sql("SELECT * FROM Track", max_rows=10_000)["row_count"] == 100  # server cap
    rejected = tools.run_readonly_sql("DELETE FROM Track")
    assert rejected["error"] == "guard_rejected" and rejected["reason"] == "not_select"
    assert tools.run_readonly_sql("SELECT nope FROM Track")["error"] == "execution_error"


def test_run_readonly_sql_timeout(chinook: SQLiteDatabase) -> None:
    tools = SQLPilotTools(SQLAgent(chinook, MockLLM(), config=AgentConfig(timeout_s=0.1)))
    out = tools.run_readonly_sql(
        "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n"
    )
    assert out["error"] == "timeout"


def test_ask(tools: SQLPilotTools) -> None:
    out = tools.ask("How many genres are there?")
    assert out["ok"] is True
    assert out["result"]["rows"] == [[25]]


def test_capability_contracts_are_explicit() -> None:
    assert set(CAPABILITIES) == {"list_tables", "describe_table", "run_readonly_sql", "ask"}
    for contract in CAPABILITIES.values():
        assert set(contract) == {"reads", "side_effects", "limits", "errors"}


def attr(obj: object, snake: str, camel: str) -> object:
    """mcp 2.x uses snake_case model attributes, mcp 1.x camelCase."""
    return getattr(obj, snake) if hasattr(obj, snake) else getattr(obj, camel)


async def test_server_registers_tools_with_annotations(chinook: SQLiteDatabase) -> None:
    pytest.importorskip("mcp")
    server = build_server(SQLAgent(chinook, MockLLM(), config=AgentConfig(llm_summary=False)))
    listed = {t.name: t for t in await server.list_tools()}
    assert set(listed) == {"list_tables", "describe_table", "run_readonly_sql", "ask"}
    for tool in listed.values():
        assert attr(tool.annotations, "read_only_hint", "readOnlyHint") is True
        assert attr(tool.annotations, "destructive_hint", "destructiveHint") is False
        assert attr(tool.annotations, "open_world_hint", "openWorldHint") is False
        assert "Contract:" in tool.description
    assert attr(listed["run_readonly_sql"].annotations, "idempotent_hint", "idempotentHint") is True
    assert attr(listed["ask"].annotations, "idempotent_hint", "idempotentHint") is False
    schema = attr(listed["run_readonly_sql"], "input_schema", "inputSchema")
    assert "sql" in schema["properties"]  # type: ignore[index]


async def test_server_call_tool_and_resource(chinook: SQLiteDatabase) -> None:
    pytest.importorskip("mcp")
    server = build_server(SQLAgent(chinook, MockLLM(), config=AgentConfig(llm_summary=False)))
    result = await server.call_tool("run_readonly_sql", {"sql": "SELECT COUNT(*) AS n FROM Genre"})
    if isinstance(result, tuple):  # mcp 1.x FastMCP returns (content, structured)
        payload = result[1]
    else:
        payload = attr(result, "structured_content", "structuredContent")
    assert payload["rows"] == [[25]]  # type: ignore[index]
    contents = list(await server.read_resource("sqlpilot://capabilities"))
    text = contents[0].content if hasattr(contents[0], "content") else contents[0].text
    assert json.loads(text) == CAPABILITIES
