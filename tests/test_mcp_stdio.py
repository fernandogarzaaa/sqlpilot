"""End-to-end: spawn `sqlpilot mcp` and speak JSON-RPC over stdio like an MCP client would."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import pytest

pytest.importorskip("mcp")


def test_mcp_stdio_roundtrip(tmp_path: Any) -> None:
    env = dict(os.environ, SQLPILOT_MEMORY_URL=":memory:")
    proc = subprocess.Popen(
        [sys.executable, "-m", "sqlpilot", "mcp", "--llm", "mock"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    assert proc.stdin is not None and proc.stdout is not None

    def send(msg: dict[str, Any]) -> None:
        proc.stdin.write(json.dumps(msg) + "\n")  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]

    def recv() -> dict[str, Any]:
        while True:
            line = proc.stdout.readline()  # type: ignore[union-attr]
            assert line, "server closed stdout"
            msg = json.loads(line)
            if "id" in msg:
                return msg  # type: ignore[no-any-return]

    try:
        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "0"},
                },
            }
        )
        assert recv()["result"]["serverInfo"]["name"] == "sqlpilot"
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = {t["name"]: t for t in recv()["result"]["tools"]}
        assert set(tools) == {"list_tables", "describe_table", "run_readonly_sql", "ask"}
        assert all(t["annotations"]["readOnlyHint"] is True for t in tools.values())
        send(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "run_readonly_sql", "arguments": {"sql": "DELETE FROM Track"}},
            }
        )
        rejected = recv()["result"]
        assert "guard_rejected" in json.dumps(rejected)
        send(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "ask", "arguments": {"question": "How many employees are there?"}},
            }
        )
        answer = recv()["result"]
        assert "[[8]]" in json.dumps(answer).replace(" ", "").replace("\\n", "")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
