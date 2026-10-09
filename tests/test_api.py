from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from sqlpilot.api import create_app
from sqlpilot.auth import Tenant, TenantRegistry, hash_key
from sqlpilot.config import AgentPool, Settings, demo_db_path
from sqlpilot.llm import MockLLM
from sqlpilot.memory import SQLiteMemoryStore

ACME_KEY = "acme-secret-key-1"
SHOP_KEY = "shop-secret-key-2"


@pytest.fixture
def client(shop_path: Path) -> TestClient:
    registry = TenantRegistry.from_tenants(
        [
            Tenant(id="acme", name="Acme", database=str(demo_db_path()), api_key_hashes=(hash_key(ACME_KEY),)),
            Tenant(
                id="shop",
                database=f"sqlite:///{shop_path}",
                api_key_hashes=(hash_key(SHOP_KEY),),
                allowed_tables=frozenset({"customers", "orders", "products"}),
                max_rows=2,
            ),
        ]
    )
    settings = Settings(llm="mock", llm_summary=False)
    pool = AgentPool(registry, MockLLM(), SQLiteMemoryStore(":memory:"), settings)
    with TestClient(create_app(pool=pool, settings=settings)) as c:
        yield c


def H(key: str) -> dict[str, str]:
    return {"X-API-Key": key}


def sse(text: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for chunk in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in chunk.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health_needs_no_auth(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-API-Key": "wrong"}, {"Authorization": "Bearer wrong"}, {"Authorization": f"Basic {ACME_KEY}"}],
)
def test_auth_required(client: TestClient, headers: dict[str, str]) -> None:
    for method, path in [
        ("get", "/v1/me"),
        ("get", "/v1/tables"),
        ("get", "/v1/tables/x"),
        ("post", "/v1/sql"),
        ("post", "/v1/ask"),
        ("post", "/v1/ask/stream"),
    ]:
        resp = getattr(client, method)(path, headers=headers, **({"json": {}} if method == "post" else {}))
        assert resp.status_code == 401, path
        assert resp.headers["www-authenticate"] == "Bearer"


def test_bearer_auth_and_me(client: TestClient) -> None:
    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {ACME_KEY}"})
    assert resp.status_code == 200
    me = resp.json()
    assert me["id"] == "acme" and me["dialect"] == "sqlite" and me["model"] == "mock"
    assert "database" not in me and "api_key_hashes" not in me


def test_tenants_see_their_own_databases(client: TestClient) -> None:
    acme = {t["name"] for t in client.get("/v1/tables", headers=H(ACME_KEY)).json()["tables"]}
    shop = {t["name"] for t in client.get("/v1/tables", headers=H(SHOP_KEY)).json()["tables"]}
    assert "Artist" in acme and "customers" not in acme
    assert shop == {"customers", "orders", "products"}  # 'secrets' hidden by the allow-list


def test_describe_table(client: TestClient) -> None:
    resp = client.get("/v1/tables/artist", headers=H(ACME_KEY))
    assert resp.status_code == 200
    assert [c["name"] for c in resp.json()["columns"]] == ["ArtistId", "Name"]
    assert client.get("/v1/tables/nope", headers=H(ACME_KEY)).status_code == 404
    assert client.get("/v1/tables/secrets", headers=H(SHOP_KEY)).status_code == 404


def test_sql_endpoint(client: TestClient) -> None:
    ok = client.post("/v1/sql", headers=H(ACME_KEY), json={"sql": "SELECT COUNT(*) FROM Artist"})
    assert ok.status_code == 200
    assert ok.json()["rows"] == [[275]]
    limited = client.post("/v1/sql", headers=H(SHOP_KEY), json={"sql": "SELECT * FROM orders"})
    assert limited.json()["row_count"] == 2 and limited.json()["truncated"] is True


@pytest.mark.parametrize(
    ("key", "sql", "status", "reason"),
    [
        (ACME_KEY, "DROP TABLE Artist", 400, "not_select"),
        (ACME_KEY, "SELECT 1; DELETE FROM Artist", 400, "multiple_statements"),
        (SHOP_KEY, "SELECT * FROM secrets", 400, "table_not_allowed"),
        (ACME_KEY, "SELECT nope FROM Artist", 422, None),
    ],
)
def test_sql_endpoint_errors(client: TestClient, key: str, sql: str, status: int, reason: str | None) -> None:
    resp = client.post("/v1/sql", headers=H(key), json={"sql": sql})
    assert resp.status_code == status
    if reason:
        assert resp.json()["reason"] == reason
    assert client.post("/v1/sql", headers=H(ACME_KEY), json={"sql": "SELECT COUNT(*) FROM Artist"}).json()["rows"] == [
        [275]
    ]


def test_sql_endpoint_validation(client: TestClient) -> None:
    assert client.post("/v1/sql", headers=H(ACME_KEY), json={"sql": ""}).status_code == 422
    assert client.post("/v1/sql", headers=H(ACME_KEY), json={"sql": "SELECT 1", "max_rows": 0}).status_code == 422


def test_ask(client: TestClient) -> None:
    resp = client.post(
        "/v1/ask", headers=H(ACME_KEY), json={"question": "How many artists are there?", "session_id": "abc"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["session_id"] == "abc"
    assert data["result"]["rows"] == [[275]]


def test_ask_generates_session_and_validates(client: TestClient) -> None:
    data = client.post("/v1/ask", headers=H(ACME_KEY), json={"question": "hi"}).json()
    assert len(data["session_id"]) == 32
    assert client.post("/v1/ask", headers=H(ACME_KEY), json={"question": ""}).status_code == 422
    assert client.post("/v1/ask", headers=H(ACME_KEY), json={"question": "x" * 2001}).status_code == 422
    bad = client.post("/v1/ask", headers=H(ACME_KEY), json={"question": "hi", "session_id": "../etc"})
    assert bad.status_code == 422


def test_ask_stream_sse(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/ask/stream",
        headers=H(ACME_KEY),
        json={"question": "How many tracks are there?", "session_id": "s-1"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())
    events = sse(body)
    names = [e for e, _ in events]
    assert names[0] == "session" and names[-2:] == ["result", "done"]
    steps = [d["node"] for e, d in events if e == "step"]
    assert steps == ["plan", "retrieve", "write_sql", "execute", "summarize", "finalize"]
    result = dict(events)["result"]
    assert result["session_id"] == "s-1"
    assert result["result"]["rows"] == [[3503]]


def test_conversation_memory_is_per_tenant(client: TestClient) -> None:
    client.post("/v1/ask", headers=H(ACME_KEY), json={"question": "How many artists?", "session_id": "same"})
    with client.stream(
        "POST",
        "/v1/ask/stream",
        headers=H(SHOP_KEY),
        json={"question": "How many customers?", "session_id": "same"},
    ) as resp:
        events = sse("".join(resp.iter_text()))
    plan = next(d for e, d in events if e == "step" and d["node"] == "plan")
    assert plan["history_messages"] == 0


def test_stream_reports_worker_errors(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from sqlpilot.agent import SQLAgent

    def boom(self: SQLAgent, question: str, session_id: str = "x"):  # type: ignore[no-untyped-def]
        raise RuntimeError("kaboom")
        yield  # pragma: no cover

    monkeypatch.setattr(SQLAgent, "stream", boom)
    with client.stream("POST", "/v1/ask/stream", headers=H(ACME_KEY), json={"question": "q"}) as resp:
        events = sse("".join(resp.iter_text()))
    assert ("error", {"detail": "RuntimeError: kaboom"}) in events
    assert events[-1][0] == "done"


def test_unavailable_tenant_database(tmp_path: Path) -> None:
    registry = TenantRegistry.from_tenants(
        [Tenant(id="gone", database=str(tmp_path / "missing.db"), api_key_hashes=(hash_key("gone-key-123"),))]
    )
    settings = Settings(llm="mock")
    pool = AgentPool(registry, MockLLM(), None, settings)
    with TestClient(create_app(pool=pool, settings=settings)) as c:
        resp = c.get("/v1/tables", headers=H("gone-key-123"))
    assert resp.status_code == 503


def test_web_ui(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<title>sqlpilot</title>" in resp.text
    assert "innerHTML" not in resp.text.replace("never innerHTML", "")


def test_app_builds_pool_from_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = Settings(llm="mock", demo_api_key="demo-key-12345", memory_url=str(tmp_path / "m.db"))
    with TestClient(create_app(settings=settings)) as c:
        assert c.get("/v1/me", headers=H("demo-key-12345")).json()["id"] == "demo"


def test_cors_is_opt_in(client: TestClient) -> None:
    resp = client.options(
        "/v1/ask", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    )
    assert "access-control-allow-origin" not in resp.headers
    settings = Settings(
        llm="mock", demo_api_key="demo-key-12345", cors_origins=["https://ok.example"], memory_url=":memory:"
    )
    with TestClient(create_app(settings=settings)) as c:
        ok = c.options("/v1/ask", headers={"Origin": "https://ok.example", "Access-Control-Request-Method": "POST"})
        assert ok.headers["access-control-allow-origin"] == "https://ok.example"
