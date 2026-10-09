from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from sqlpilot.agent import AgentConfig, SQLAgent
from sqlpilot.db import SQLiteDatabase
from sqlpilot.llm import MockLLM
from sqlpilot.tracing import Tracing, agentlens_from_env, setup_otel


class FakeLensSpan:
    def __init__(self, kind: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.kind, self.args, self.kwargs = kind, args, kwargs
        self.attributes: dict[str, Any] = {}
        self.usage: list[dict[str, Any]] = []
        self.output: Any = None
        self.result: Any = None
        self.errors: list[Any] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def record_usage(self, **kwargs: Any) -> None:
        self.usage.append(kwargs)

    def set_output(self, value: Any) -> None:
        self.output = value

    def set_result(self, value: Any) -> None:
        self.result = value

    def record_error(self, err: Any) -> None:
        self.errors.append(err)


class FakeLens:
    """Mimics agentlens.tracer.Tracer's public context-manager API."""

    def __init__(self) -> None:
        self.spans: list[FakeLensSpan] = []

    def _make(self, kind: str):  # type: ignore[no-untyped-def]
        @contextmanager
        def factory(*args: Any, **kwargs: Any) -> Iterator[FakeLensSpan]:
            span = FakeLensSpan(kind, args, kwargs)
            self.spans.append(span)
            yield span

        return factory

    def __getattr__(self, name: str) -> Any:
        if name in {"run", "agent_step", "llm_call", "tool_call"}:
            return self._make(name)
        raise AttributeError(name)


def test_agentlens_hook_receives_run_steps_llm_and_tools(chinook: SQLiteDatabase) -> None:
    lens = FakeLens()
    agent = SQLAgent(chinook, MockLLM(), config=AgentConfig(llm_summary=False), tracing=Tracing(lens))
    result = agent.ask("How many genres are there?")
    kinds = [s.kind for s in lens.spans]
    assert kinds[0] == "run"
    assert kinds.count("llm_call") == 1
    assert kinds.count("tool_call") == 1
    steps = [s.kwargs["name"] for s in lens.spans if s.kind == "agent_step"]
    assert steps == [
        "sqlpilot.plan",
        "sqlpilot.retrieve",
        "sqlpilot.write_sql",
        "sqlpilot.execute",
        "sqlpilot.summarize",
    ]  # no memory store -> finalize opens no span
    run = lens.spans[0]
    assert run.kwargs == {"input": "How many genres are there?"}
    assert run.output == result.answer
    assert run.usage == []  # usage lives on the LLM spans only (no double counting)
    assert run.attributes["sqlpilot.prompt_tokens"] == result.prompt_tokens
    assert run.attributes["sqlpilot.llm_calls"] == 1
    llm = next(s for s in lens.spans if s.kind == "llm_call")
    assert llm.args == ("mock",)
    assert llm.usage[0]["input_tokens"] > 0
    tool = next(s for s in lens.spans if s.kind == "tool_call")
    assert tool.args == ("run_readonly_sql",)
    assert tool.result == "1 rows"


def test_agentlens_hook_records_errors(shop_db: SQLiteDatabase) -> None:
    from sqlpilot.llm import ScriptedLLM

    lens = FakeLens()
    agent = SQLAgent(
        shop_db,
        ScriptedLLM(["SELECT nope FROM customers"]),
        config=AgentConfig(max_retries=0),
        tracing=Tracing(lens),
    )
    agent.ask("x")
    tool = next(s for s in lens.spans if s.kind == "tool_call")
    assert tool.errors and "no such column" in str(tool.errors[0])


class ExplodingLens:
    def __getattr__(self, name: str) -> Any:
        raise RuntimeError("tracer is broken")


def test_broken_agentlens_never_breaks_queries(chinook: SQLiteDatabase) -> None:
    agent = SQLAgent(chinook, MockLLM(), config=AgentConfig(llm_summary=False), tracing=Tracing(ExplodingLens()))
    assert agent.ask("How many genres are there?").ok


class BadMethodSpan(FakeLensSpan):
    def set_attribute(self, key: str, value: Any) -> None:
        raise RuntimeError("nope")


def test_span_handle_swallows_hook_errors() -> None:
    from sqlpilot.tracing import SpanHandle

    class Otel:
        def __init__(self) -> None:
            self.attrs: dict[str, Any] = {}

        def set_attribute(self, k: str, v: Any) -> None:
            self.attrs[k] = v

    otel = Otel()
    handle = SpanHandle(otel, BadMethodSpan("x", (), {}))
    handle.set("a", 1).set("b", {"x": 1}).set("c", None)
    assert otel.attrs == {"a": 1, "b": "{'x': 1}"}


def test_otel_spans_are_emitted(chinook: SQLiteDatabase) -> None:
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("sqlpilot")
    import sqlpilot.tracing as tracing_mod

    original = tracing_mod._otel
    tracing_mod._otel = tracer
    try:
        SQLAgent(chinook, MockLLM(), config=AgentConfig(llm_summary=False)).ask("How many genres are there?")
    finally:
        tracing_mod._otel = original
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "sqlpilot.ask" in spans
    assert "sqlpilot.node.write_sql" in spans
    assert "execute_tool run_readonly_sql" in spans
    llm = spans["chat mock"]
    assert llm.attributes["gen_ai.request.model"] == "mock"
    assert llm.attributes["gen_ai.usage.input_tokens"] > 0
    root = spans["sqlpilot.ask"]
    assert spans["sqlpilot.node.plan"].parent.span_id == root.context.span_id
    assert trace.get_tracer_provider() is not None


def test_agentlens_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SQLPILOT_AGENTLENS", raising=False)
    assert agentlens_from_env() is None
    monkeypatch.setenv("SQLPILOT_AGENTLENS", "0")
    assert agentlens_from_env() is None
    monkeypatch.setenv("SQLPILOT_AGENTLENS", "1")
    try:
        import agentlens.tracer  # noqa: F401
    except ImportError:
        assert agentlens_from_env() is None
    else:  # pragma: no cover - only when agentlens is installed
        assert agentlens_from_env() is not None


def test_setup_otel_choices() -> None:
    assert setup_otel("none") is False
    with pytest.raises(ValueError):
        setup_otel("carrier-pigeon")
