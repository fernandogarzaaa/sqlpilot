"""Multi-tenant FastAPI service with per-tenant API keys and SSE streaming of agent steps.

Endpoints (all under ``/v1`` need ``X-API-Key: <key>`` or ``Authorization: Bearer <key>``):

====================  =====================================================================
``GET  /healthz``     liveness (no auth)
``GET  /v1/me``       the calling tenant (id, limits; never keys or DB URLs)
``GET  /v1/tables``   tables visible to the tenant
``GET  /v1/tables/{name}``  columns, keys and row count of one table
``POST /v1/sql``      run caller-supplied SQL through the guard + read-only executor
``POST /v1/ask``      ask a question; returns the agent result as JSON
``POST /v1/ask/stream``  same, as Server-Sent Events: ``step`` events per graph node, then
                      ``result``, then ``done`` (``error`` on failure)
``GET  /``            the web UI
====================  =====================================================================
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from sqlpilot import __version__
from sqlpilot.agent import SQLAgent
from sqlpilot.auth import Tenant, extract_api_key
from sqlpilot.config import AgentPool, Settings
from sqlpilot.db import ExecutionError, QueryTimeout
from sqlpilot.guard import GuardRejected
from sqlpilot.routing import MAX_QUESTION_CHARS

__all__ = ["create_app"]

_SESSION = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str | None = Field(default=None, description="Reuse to keep conversation memory.")

    @field_validator("session_id")
    @classmethod
    def _session(cls, value: str | None) -> str | None:
        if value is not None and not _SESSION.match(value):
            raise ValueError("session_id: 1-64 characters from [A-Za-z0-9_.:-]")
        return value


class SQLRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=20_000)
    max_rows: int | None = Field(default=None, ge=1, le=100_000)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _get_pool(request: Request) -> AgentPool:
    pool: AgentPool = request.app.state.pool
    return pool


def _get_tenant(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Tenant:
    tenant = _get_pool(request).registry.authenticate(extract_api_key(authorization, x_api_key))
    if tenant is None:
        raise HTTPException(
            status_code=401, detail="Invalid or missing API key", headers={"WWW-Authenticate": "Bearer"}
        )
    return tenant


TenantDep = Annotated[Tenant, Depends(_get_tenant)]


def create_app(pool: AgentPool | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Pass a ready :class:`AgentPool` (tests) or let it come from the environment."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "pool", None) is None:
            app.state.pool = AgentPool.from_settings(settings)
        yield
        app.state.pool.close()

    app = FastAPI(
        title="sqlpilot",
        version=__version__,
        lifespan=lifespan,
        description="Text-to-SQL analyst agent: multi-tenant, read-only, streaming.",
    )
    app.state.pool = pool
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "X-API-Key", "Content-Type"],
        )

    def agent_for(request: Request, tenant: Tenant) -> SQLAgent:
        try:
            return _get_pool(request).agent(tenant)
        except (FileNotFoundError, ValueError, ImportError, ExecutionError) as err:
            raise HTTPException(status_code=503, detail=f"Tenant database unavailable: {err}") from err

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/me")
    def me(tenant: TenantDep, request: Request) -> dict[str, Any]:
        agent = agent_for(request, tenant)
        return {**tenant.public(), "dialect": agent.db.dialect, "model": agent.llm.model}

    @app.get("/v1/tables")
    def tables(tenant: TenantDep, request: Request) -> dict[str, Any]:
        agent = agent_for(request, tenant)
        return {
            "tables": [
                {"name": t.name, "row_count": t.row_count, "columns": len(t.columns)} for t in agent.list_tables()
            ]
        }

    @app.get("/v1/tables/{name}")
    def table(name: str, tenant: TenantDep, request: Request) -> dict[str, Any]:
        agent = agent_for(request, tenant)
        try:
            return agent.describe_table(name).to_dict()
        except KeyError as err:
            raise HTTPException(status_code=404, detail=f"Unknown table: {name}") from err

    @app.post("/v1/sql")
    def run_sql(body: SQLRequest, tenant: TenantDep, request: Request) -> JSONResponse:
        agent = agent_for(request, tenant)
        try:
            result = agent.run_sql(body.sql, body.max_rows)
        except GuardRejected as err:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "guard_rejected",
                    "reason": err.result.reason,
                    "detail": err.result.message,
                },
            )
        except QueryTimeout as err:
            return JSONResponse(status_code=408, content={"error": "timeout", "detail": str(err)})
        except ExecutionError as err:
            return JSONResponse(status_code=422, content={"error": "execution_error", "detail": str(err)})
        return JSONResponse(content=result.to_dict())

    @app.post("/v1/ask")
    async def ask(body: AskRequest, tenant: TenantDep, request: Request) -> dict[str, Any]:
        agent = agent_for(request, tenant)
        session = body.session_id or uuid.uuid4().hex
        result = await asyncio.to_thread(agent.ask, body.question, session)
        return {**result.to_dict(), "session_id": session}

    @app.post("/v1/ask/stream")
    async def ask_stream(body: AskRequest, tenant: TenantDep, request: Request) -> StreamingResponse:
        agent = agent_for(request, tenant)
        session = body.session_id or uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        def worker() -> None:
            # The whole graph run happens in one thread so tracing context stays intact.
            try:
                for event in agent.stream(body.question, session):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as err:  # surfaced to the client as an `error` event
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "detail": f"{type(err).__name__}: {err}"})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        async def events() -> AsyncIterator[str]:
            task = loop.run_in_executor(None, worker)
            yield _sse("session", {"session_id": session})
            while True:
                event = await queue.get()
                if event is None:
                    break
                if event["type"] == "step":
                    yield _sse("step", {"node": event["node"], **event["detail"]})
                elif event["type"] == "result":
                    yield _sse("result", {**event["result"], "session_id": session})
                else:
                    yield _sse("error", {"detail": event.get("detail", "unknown error")})
            await task
            yield _sse("done", {})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        html = (resources.files("sqlpilot") / "web" / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    return app
