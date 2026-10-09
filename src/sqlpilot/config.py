"""Runtime settings (environment variables) and the per-tenant agent pool."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from sqlpilot.agent import AgentConfig, SQLAgent
from sqlpilot.auth import Tenant, TenantRegistry, hash_key, load_tenants
from sqlpilot.db import Database, connect
from sqlpilot.llm import LLMClient, MockLLM, OpenAICompatibleLLM
from sqlpilot.memory import MemoryStore, open_memory
from sqlpilot.tracing import Tracing, agentlens_from_env

__all__ = ["AgentPool", "Settings", "build_llm", "demo_db_path"]

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8080/v1"  # llama.cpp `llama-server` default port


def demo_db_path() -> Path:
    """The bundled Chinook sample database (read-only)."""
    return Path(str(resources.files("sqlpilot") / "data" / "chinook.sqlite"))


@dataclass
class Settings:
    """All knobs, read from ``SQLPILOT_*`` environment variables by :meth:`from_env`."""

    llm: str = "openai"  # openai | mock
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model: str = "local"
    llm_api_key: str | None = None
    llm_timeout_s: float = 120.0
    tenants_file: str | None = None
    demo_api_key: str | None = None
    memory_url: str = "sqlite:///.sqlpilot/memory.db"
    otel_exporter: str = "none"
    max_retries: int = 2
    top_k_tables: int = 4
    llm_summary: bool = True
    cors_origins: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        e = dict(os.environ if env is None else env)

        def flag(name: str, default: bool) -> bool:
            value = e.get(name)
            return default if value is None else value.lower() in {"1", "true", "yes", "on"}

        return cls(
            llm=e.get("SQLPILOT_LLM", "openai"),
            llm_base_url=e.get("SQLPILOT_LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
            llm_model=e.get("SQLPILOT_LLM_MODEL", "local"),
            llm_api_key=e.get("SQLPILOT_LLM_API_KEY") or None,
            llm_timeout_s=float(e.get("SQLPILOT_LLM_TIMEOUT_S", "120")),
            tenants_file=e.get("SQLPILOT_TENANTS_FILE") or None,
            demo_api_key=e.get("SQLPILOT_DEMO_API_KEY") or None,
            memory_url=e.get("SQLPILOT_MEMORY_URL", "sqlite:///.sqlpilot/memory.db"),
            otel_exporter=e.get("SQLPILOT_OTEL_EXPORTER", "none"),
            max_retries=int(e.get("SQLPILOT_MAX_RETRIES", "2")),
            top_k_tables=int(e.get("SQLPILOT_TOP_K_TABLES", "4")),
            llm_summary=flag("SQLPILOT_LLM_SUMMARY", True),
            cors_origins=[o for o in e.get("SQLPILOT_CORS_ORIGINS", "").split(",") if o],
        )

    def registry(self) -> TenantRegistry:
        """Tenants from the TOML file, or a single ``demo`` tenant on the bundled Chinook DB."""
        if self.tenants_file:
            return load_tenants(self.tenants_file)
        if self.demo_api_key:
            demo = Tenant(
                id="demo",
                name="Demo (Chinook)",
                database=str(demo_db_path()),
                api_key_hashes=(hash_key(self.demo_api_key),),
            )
            return TenantRegistry.from_tenants([demo])
        raise ValueError("Set SQLPILOT_TENANTS_FILE (see examples/tenants.toml) or SQLPILOT_DEMO_API_KEY")


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm == "mock":
        return MockLLM()
    if settings.llm == "openai":
        return OpenAICompatibleLLM(
            settings.llm_base_url, settings.llm_model, settings.llm_api_key, timeout_s=settings.llm_timeout_s
        )
    raise ValueError(f"SQLPILOT_LLM must be 'openai' or 'mock', got {settings.llm!r}")


class AgentPool:
    """One lazily-built :class:`SQLAgent` (and database connection) per tenant."""

    def __init__(
        self,
        registry: TenantRegistry,
        llm: LLMClient,
        memory: MemoryStore | None,
        settings: Settings | None = None,
        tracing: Tracing | None = None,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.memory = memory
        self.settings = settings or Settings()
        self.tracing = tracing or Tracing(agentlens_from_env())
        self._agents: dict[str, SQLAgent] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> AgentPool:
        return cls(settings.registry(), build_llm(settings), open_memory(settings.memory_url), settings)

    def agent(self, tenant: Tenant) -> SQLAgent:
        with self._lock:
            agent = self._agents.get(tenant.id)
            if agent is None:
                db: Database = connect(tenant.database, name=tenant.id)
                config = AgentConfig(
                    max_retries=self.settings.max_retries,
                    top_k_tables=self.settings.top_k_tables,
                    llm_summary=self.settings.llm_summary,
                    max_rows=tenant.max_rows,
                    timeout_s=tenant.timeout_s,
                    allowed_tables=tenant.allowed_tables,
                )
                agent = SQLAgent(
                    db, self.llm, memory=self.memory, config=config, tracing=self.tracing, tenant_id=tenant.id
                )
                self._agents[tenant.id] = agent
            return agent

    def close(self) -> None:
        with self._lock:
            for agent in self._agents.values():
                agent.db.close()
            self._agents.clear()
        if self.memory is not None:
            self.memory.close()

    def stats(self) -> dict[str, Any]:
        return {"tenants": len(self.registry), "loaded_agents": len(self._agents)}
