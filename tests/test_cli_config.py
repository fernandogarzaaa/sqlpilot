from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sqlpilot.auth import hash_key
from sqlpilot.cli import main
from sqlpilot.config import AgentPool, Settings, build_llm, demo_db_path
from sqlpilot.llm import MockLLM, OpenAICompatibleLLM


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("SQLPILOT_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("SQLPILOT_MEMORY_URL", ":memory:")


def test_ask_mock(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "--llm", "mock", "How many genres are there?"]) == 0
    out = capsys.readouterr()
    assert "SELECT COUNT(*) AS count FROM Genre" in out.out
    assert "25" in out.out
    assert "plan" in out.err


def test_ask_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "--llm", "mock", "--json", "--no-memory", "How many artists?"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["result"]["rows"] == [[275]]


def test_ask_write_request_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "--llm", "mock", "delete all artists"]) == 1


def test_sql_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sql", "SELECT COUNT(*) AS n FROM Album"]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == [[347]]
    assert main(["sql", "DROP TABLE Album"]) == 1
    assert "not_select" in capsys.readouterr().err


def test_hash_key_and_demo_db(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert main(["hash-key", "abc"]) == 0
    assert capsys.readouterr().out.strip() == hash_key("abc")
    assert main(["demo-db", str(tmp_path / "x" / "c.sqlite")]) == 0
    assert (tmp_path / "x" / "c.sqlite").stat().st_size == demo_db_path().stat().st_size


def test_bench_requires_data(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert main(["bench", "--spider-dir", str(tmp_path / "none")]) == 2
    assert "make bench-data" in capsys.readouterr().err


def test_serve_requires_tenants(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("uvicorn")
    assert main(["serve", "--llm", "mock"]) == 2


def test_settings_from_env() -> None:
    s = Settings.from_env(
        {
            "SQLPILOT_LLM": "mock",
            "SQLPILOT_LLM_SUMMARY": "false",
            "SQLPILOT_MAX_RETRIES": "5",
            "SQLPILOT_CORS_ORIGINS": "https://a,https://b",
            "SQLPILOT_LLM_API_KEY": "",
        }
    )
    assert s.llm == "mock"
    assert s.llm_summary is False
    assert s.max_retries == 5
    assert s.cors_origins == ["https://a", "https://b"]
    assert s.llm_api_key is None
    d = Settings.from_env({})
    assert d.llm == "openai" and d.llm_base_url == "http://127.0.0.1:8080/v1" and d.llm_summary is True
    assert d.cors_origins == []


def test_registry_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SQLPILOT_TENANTS_FILE"):
        Settings().registry()
    demo = Settings(demo_api_key="demo-key-123").registry()
    assert demo.authenticate("demo-key-123").id == "demo"  # type: ignore[union-attr]
    path = tmp_path / "t.toml"
    path.write_text(f'[[tenants]]\nid = "x"\ndatabase = "x.db"\napi_key_sha256 = ["{hash_key("k")}"]\n')
    assert Settings(tenants_file=str(path), demo_api_key="ignored").registry().get("x") is not None


def test_build_llm() -> None:
    assert isinstance(build_llm(Settings(llm="mock")), MockLLM)
    llm = build_llm(Settings(llm="openai", llm_model="qwen"))
    assert isinstance(llm, OpenAICompatibleLLM) and llm.model == "qwen"
    with pytest.raises(ValueError):
        build_llm(Settings(llm="other"))


def test_agent_pool_caches_per_tenant(shop_path: Path) -> None:
    from sqlpilot.auth import Tenant, TenantRegistry

    reg = TenantRegistry.from_tenants(
        [
            Tenant(
                id="a",
                database=str(shop_path),
                api_key_hashes=(hash_key("key-aaaaaa"),),
                max_rows=7,
                timeout_s=3,
            ),
            Tenant(id="b", database=str(demo_db_path()), api_key_hashes=(hash_key("key-bbbbbb"),)),
        ]
    )
    pool = AgentPool(reg, MockLLM(), None, Settings(max_retries=1, top_k_tables=2, llm_summary=False))
    a1 = pool.agent(reg.get("a"))  # type: ignore[arg-type]
    assert pool.agent(reg.get("a")) is a1  # type: ignore[arg-type]
    assert pool.agent(reg.get("b")) is not a1  # type: ignore[arg-type]
    assert a1.config.max_rows == 7 and a1.config.timeout_s == 3 and a1.config.max_retries == 1
    assert a1.config.top_k_tables == 2 and a1.config.llm_summary is False
    assert a1.tenant_id == "a"
    assert pool.stats() == {"tenants": 2, "loaded_agents": 2}
    pool.close()
    assert pool.stats()["loaded_agents"] == 0
