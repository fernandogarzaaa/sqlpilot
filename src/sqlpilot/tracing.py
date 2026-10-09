"""Tracing: OpenTelemetry spans for every agent run, graph node, LLM call and tool call.

Spans always go through the OpenTelemetry API (a no-op until an SDK is configured, see
:func:`setup_otel`). Attribute names follow the OTel GenAI semantic conventions
(``gen_ai.request.model``, ``gen_ai.usage.input_tokens``, ``gen_ai.tool.name``...).

**agentlens hook.** When an agentlens tracer is supplied (or ``SQLPILOT_AGENTLENS=1`` and the
``agentlens`` package is importable), every span is mirrored into it with the matching kind
(``run`` / ``agent_step`` / ``llm`` / ``tool``), so agentlens can show cost, token and step
accounting for sqlpilot runs. The hook is duck-typed against agentlens' public ``Tracer`` API
(``run``, ``agent_step``, ``llm_call``, ``tool_call`` context managers whose spans have
``set_attribute``, ``record_usage``, ``set_output``, ``set_result`` and ``record_error``), so any
tracer with that shape works, and a failure inside the hook never breaks a query.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

from opentelemetry import trace

__all__ = ["SpanHandle", "Tracing", "agentlens_from_env", "setup_otel"]

log = logging.getLogger(__name__)
_otel = trace.get_tracer("sqlpilot")


class SpanHandle:
    """One logical span, fanned out to OpenTelemetry and (optionally) agentlens."""

    def __init__(self, otel_span: Any, lens_span: Any | None) -> None:
        self.otel = otel_span
        self.lens = lens_span

    def _lens(self, method: str, *args: Any, **kwargs: Any) -> None:
        if self.lens is None:
            return
        try:
            getattr(self.lens, method)(*args, **kwargs)
        except Exception:  # never let tracing break a query
            log.debug("agentlens hook failed on %s", method, exc_info=True)

    def set(self, key: str, value: Any) -> SpanHandle:
        if isinstance(value, (str, bool, int, float)):
            self.otel.set_attribute(key, value)
        elif value is not None:
            self.otel.set_attribute(key, str(value)[:4000])
        self._lens("set_attribute", key, value)
        return self

    def usage(self, input_tokens: int, output_tokens: int, model: str | None = None) -> SpanHandle:
        self.otel.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        self.otel.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        if model:
            self.otel.set_attribute("gen_ai.response.model", model)
        self._lens("record_usage", input_tokens=input_tokens, output_tokens=output_tokens, model=model)
        return self

    def output(self, value: Any) -> SpanHandle:
        self.otel.set_attribute("sqlpilot.output", str(value)[:4000])
        self._lens("set_output", value)
        return self

    def result(self, value: Any) -> SpanHandle:
        self.otel.set_attribute("gen_ai.tool.call.result", str(value)[:4000])
        self._lens("set_result", value)
        return self

    def error(self, err: BaseException | str) -> SpanHandle:
        from opentelemetry.trace import Status, StatusCode

        message = f"{type(err).__name__}: {err}" if isinstance(err, BaseException) else err
        self.otel.set_status(Status(StatusCode.ERROR, message))
        if isinstance(err, BaseException):
            self.otel.record_exception(err)
        self._lens("record_error", err)
        return self


class Tracing:
    """Factory for sqlpilot spans. ``agentlens`` is an optional agentlens ``Tracer``."""

    def __init__(self, agentlens: Any | None = None) -> None:
        self.agentlens = agentlens

    @contextmanager
    def _span(
        self,
        name: str,
        lens_factory: str | None,
        lens_args: tuple[Any, ...],
        lens_kwargs: dict[str, Any],
        attributes: dict[str, Any],
    ) -> Iterator[SpanHandle]:
        with ExitStack() as stack:
            otel_span = stack.enter_context(_otel.start_as_current_span(name, attributes=attributes))
            lens_span = None
            if self.agentlens is not None and lens_factory is not None:
                try:
                    lens_span = stack.enter_context(getattr(self.agentlens, lens_factory)(*lens_args, **lens_kwargs))
                except Exception:
                    log.debug("agentlens hook failed to open %s", lens_factory, exc_info=True)
            yield SpanHandle(otel_span, lens_span)

    def run(self, question: str, tenant: str = "", session: str = "") -> Any:
        attrs = {"sqlpilot.question": question[:2000], "sqlpilot.tenant": tenant, "sqlpilot.session": session}
        return self._span("sqlpilot.ask", "run", ("sqlpilot.ask",), {"input": question}, attrs)

    def step(self, node: str, index: int | None = None) -> Any:
        return self._span(
            f"sqlpilot.node.{node}",
            "agent_step",
            (index,),
            {"name": f"sqlpilot.{node}"},
            {"sqlpilot.node": node},
        )

    def llm(self, model: str, purpose: str) -> Any:
        attrs = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": model,
            "sqlpilot.llm.purpose": purpose,
        }
        return self._span(f"chat {model}", "llm_call", (model,), {"purpose": purpose}, attrs)

    def tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        attrs = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name}
        return self._span(f"execute_tool {name}", "tool_call", (name,), {"arguments": arguments or {}}, attrs)


def agentlens_from_env() -> Any | None:
    """The process-wide agentlens tracer when ``SQLPILOT_AGENTLENS`` is truthy and agentlens is installed."""
    if os.environ.get("SQLPILOT_AGENTLENS", "").lower() not in {"1", "true", "yes", "on"}:
        return None
    try:
        import agentlens.tracer as lens
    except ImportError:
        log.warning("SQLPILOT_AGENTLENS is set but agentlens is not installed; using OpenTelemetry only")
        return None
    return lens.get_tracer()


def setup_otel(exporter: str = "none", service_name: str = "sqlpilot") -> bool:
    """Install an OpenTelemetry SDK provider. ``exporter``: none | console | otlp. Needs ``sqlpilot[otel]``."""
    if exporter == "none":
        return False
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter
    except ImportError:
        log.warning("opentelemetry-sdk is not installed (pip install 'sqlpilot[otel]'); tracing stays no-op")
        return False
    span_exporter: SpanExporter
    if exporter == "console":
        span_exporter = ConsoleSpanExporter()
    elif exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            log.warning("OTLP exporter missing (pip install 'sqlpilot[otel]')")
            return False
        span_exporter = OTLPSpanExporter()
    else:
        raise ValueError(f"unknown exporter {exporter!r}; expected none, console or otlp")
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)
    return True
