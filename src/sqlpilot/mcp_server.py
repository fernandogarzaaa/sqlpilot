"""MCP server (Python MCP SDK, stdio or streamable HTTP) over one database.

Tools, each with an explicit capability contract (what it reads, what it can never do, its
limits and its failure modes) in its description and MCP tool annotations
(``readOnlyHint=True``, ``destructiveHint=False``, ``openWorldHint=False``):

* ``list_tables()``                        table names, row counts and column counts
* ``describe_table(table)``                columns, types, primary/foreign keys
* ``run_readonly_sql(sql, max_rows=50)``   guarded, read-only, row- and time-limited execution
* ``ask(question, session_id="mcp")``      the full agent: retrieval, SQL writing, self-correction

The same contract is published as JSON at the resource ``sqlpilot://capabilities``.
Errors are returned as data (``{"error": ..., "reason": ...}``), never as exceptions, so the
calling model can read them and adjust.
"""

from __future__ import annotations

import json
from typing import Any

from sqlpilot.agent import SQLAgent
from sqlpilot.db import ExecutionError, QueryTimeout
from sqlpilot.guard import GuardRejected

__all__ = ["CAPABILITIES", "SQLPilotTools", "build_server"]

CAPABILITIES: dict[str, dict[str, Any]] = {
    "list_tables": {
        "reads": "database catalog (table names, row counts)",
        "side_effects": "none",
        "limits": "tables outside the configured allow-list are hidden",
        "errors": "none expected",
    },
    "describe_table": {
        "reads": "database catalog (columns, types, primary and foreign keys)",
        "side_effects": "none",
        "limits": "only tables returned by list_tables",
        "errors": "{error: 'unknown_table'} for a missing or hidden table",
    },
    "run_readonly_sql": {
        "reads": "table data",
        "side_effects": "none: exactly one SELECT / WITH ... SELECT / set operation is allowed; "
        "INSERT, UPDATE, DELETE, DDL, PRAGMA, ATTACH, COPY, SET, transactions, multiple statements, "
        "row locks and dangerous functions are rejected before execution, and the connection itself "
        "is read-only",
        "limits": "max_rows is capped by the server (default 200); per-query time limit (default 10 s)",
        "errors": "{error: 'guard_rejected', reason}, {error: 'timeout'}, {error: 'execution_error'}",
    },
    "ask": {
        "reads": "table data via SQL the agent writes; conversation memory for session_id",
        "side_effects": "appends the turn to sqlpilot's own memory store (never to the queried database)",
        "limits": "same guard and limits as run_readonly_sql; at most max_retries self-corrections; "
        "calls the configured LLM",
        "errors": "returned in the result as error / error_kind (llm_error, guard_rejected, "
        "execution_error, timeout, write_request, invalid)",
    },
}


def _contract(name: str, summary: str) -> str:
    c = CAPABILITIES[name]
    return (
        f"{summary}\n\nContract: reads {c['reads']}. Side effects: {c['side_effects']}. "
        f"Limits: {c['limits']}. Errors: {c['errors']}."
    )


class SQLPilotTools:
    """Plain-Python implementations of the MCP tools (testable without an MCP client)."""

    def __init__(self, agent: SQLAgent, default_rows: int = 50) -> None:
        self.agent = agent
        self.default_rows = default_rows

    def list_tables(self) -> dict[str, Any]:
        return {
            "database": self.agent.db.name,
            "dialect": self.agent.db.dialect,
            "tables": [
                {"name": t.name, "row_count": t.row_count, "columns": len(t.columns)} for t in self.agent.list_tables()
            ],
        }

    def describe_table(self, table: str) -> dict[str, Any]:
        try:
            return self.agent.describe_table(table).to_dict()
        except KeyError:
            return {"error": "unknown_table", "detail": f"No table named {table!r}. Call list_tables first."}

    def run_readonly_sql(self, sql: str, max_rows: int | None = None) -> dict[str, Any]:
        try:
            result = self.agent.run_sql(sql, max_rows or self.default_rows)
        except GuardRejected as err:
            return {"error": "guard_rejected", "reason": err.result.reason, "detail": err.result.message}
        except QueryTimeout as err:
            return {"error": "timeout", "detail": str(err)}
        except ExecutionError as err:
            return {"error": "execution_error", "detail": str(err)}
        return result.to_dict()

    def ask(self, question: str, session_id: str = "mcp") -> dict[str, Any]:
        return self.agent.ask(question, session_id).to_dict()


def build_server(agent: SQLAgent, name: str = "sqlpilot") -> Any:
    """Create the MCP server object (mcp 2.x ``MCPServer`` or mcp 1.x ``FastMCP``)."""
    import importlib

    from mcp.types import ToolAnnotations

    try:
        Server: Any = importlib.import_module("mcp.server.mcpserver").MCPServer
    except ImportError:  # mcp 1.x
        Server = importlib.import_module("mcp.server.fastmcp").FastMCP

    tools = SQLPilotTools(agent)
    server = Server(
        name,
        instructions=(
            "Read-only analyst over one SQL database. Start with list_tables/describe_table, then "
            "run_readonly_sql for precise queries or ask for natural-language questions. Nothing here "
            "can modify data."
        ),
    )
    # model_validate accepts the protocol's camelCase names on both mcp 1.x and 2.x.
    hints = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
    read_only = ToolAnnotations.model_validate({**hints, "idempotentHint": True})
    agentic = ToolAnnotations.model_validate({**hints, "idempotentHint": False})

    @server.tool(description=_contract("list_tables", "List the tables in the database."), annotations=read_only)
    def list_tables() -> dict[str, Any]:
        return tools.list_tables()

    @server.tool(
        description=_contract("describe_table", "Describe one table's columns and keys."),
        annotations=read_only,
    )
    def describe_table(table: str) -> dict[str, Any]:
        return tools.describe_table(table)

    @server.tool(
        description=_contract("run_readonly_sql", "Run one read-only SQL query and return rows."),
        annotations=read_only,
    )
    def run_readonly_sql(sql: str, max_rows: int = 50) -> dict[str, Any]:
        return tools.run_readonly_sql(sql, max_rows)

    @server.tool(
        description=_contract(
            "ask",
            "Answer a natural-language question about the data (writes, checks and runs the SQL for you).",
        ),
        annotations=agentic,
    )
    def ask(question: str, session_id: str = "mcp") -> dict[str, Any]:
        return tools.ask(question, session_id)

    @server.resource(
        "sqlpilot://capabilities",
        name="capabilities",
        mime_type="application/json",
        description="Machine-readable capability contracts for every tool.",
    )
    def capabilities() -> str:
        return json.dumps(CAPABILITIES, indent=2)

    return server
