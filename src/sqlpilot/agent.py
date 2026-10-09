"""The multi-agent text-to-SQL graph (LangGraph).

```
START -> plan --(sql)--------> retrieve -> write_sql -> execute --ok--> summarize -> finalize -> END
             \\--(schema)-----> schema_answer ------------------------------------> finalize
             \\--(other)------> respond -------------------------------------------> finalize
                                                execute --error, retries left--> correct -> execute
                                                execute --error, no retries----> fail -----> finalize
```

* **plan**: deterministic router (:func:`sqlpilot.routing.classify_intent`) plus memory load:
  conversation history for the session and similar past successful queries.
* **retrieve**: :class:`sqlpilot.retrieval.SchemaRetriever` picks the relevant tables/columns
  (or the full schema in zero-shot mode).
* **write_sql** / **correct**: the LLM writes SQL; on failure the error and the failed SQL go back
  to the LLM (self-correction), bounded by ``max_retries``.
* **execute**: :class:`sqlpilot.guard.SQLGuard` then the read-only executor (row and time limits).
* **summarize**: the LLM (or a template) turns the result into a short answer.
* **finalize**: persists the turn to memory and records successful queries as future examples.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from sqlpilot.db import Database, ExecutionError, QueryResult, QueryTimeout, TableInfo
from sqlpilot.guard import GuardConfig, GuardRejected, SQLGuard
from sqlpilot.llm import LLMClient, LLMError, extract_sql
from sqlpilot.memory import MemoryStore, PastQuery
from sqlpilot.prompts import summary_messages, writer_messages
from sqlpilot.retrieval import Embedder, SchemaRetriever
from sqlpilot.routing import classify_intent, route_after_execute, route_after_plan
from sqlpilot.tracing import Tracing

__all__ = ["AgentConfig", "AgentResult", "AgentState", "SQLAgent"]


@dataclass
class AgentConfig:
    use_retrieval: bool = True
    self_correct: bool = True
    max_retries: int = 2
    top_k_tables: int = 4
    use_examples: bool = True
    history_turns: int = 6
    max_rows: int = 200
    timeout_s: float = 10.0
    llm_summary: bool = True
    writer_max_tokens: int = 300
    summary_max_tokens: int = 160
    allowed_tables: frozenset[str] | None = None


class AgentState(TypedDict, total=False):
    question: str
    tenant_id: str
    session_id: str
    intent: str
    history: list[dict[str, str]]
    examples: list[dict[str, str]]
    tables: list[str]
    schema_ddl: str
    sql: str
    attempts: int
    error: str | None
    error_kind: str | None
    result: dict[str, Any] | None
    answer: str
    failed: list[dict[str, str]]
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]
    llm_calls: Annotated[int, operator.add]


@dataclass
class AgentResult:
    question: str
    intent: str
    answer: str
    sql: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    error_kind: str | None = None
    attempts: int = 0
    tables: list[str] = field(default_factory=list)
    failed_attempts: list[dict[str, str]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


_SMALLTALK_ANSWER = (
    "I answer questions about this database by writing and running read-only SQL. "
    'Try something like "How many customers are there?" or "What tables are there?".'
)
_WRITE_ANSWER = (
    "I can only read data: this assistant runs read-only queries, so I can't insert, update, "
    "delete or change the schema. I can help you find the rows you have in mind."
)
_INVALID_ANSWER = "Please ask a question about the data (non-empty, under 2000 characters)."


class SQLAgent:
    """One agent per (tenant, database). Thread-safe for concurrent ``ask`` calls."""

    def __init__(
        self,
        db: Database,
        llm: LLMClient,
        memory: MemoryStore | None = None,
        config: AgentConfig | None = None,
        tracing: Tracing | None = None,
        tenant_id: str = "default",
        embedder: Embedder | None = None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.memory = memory
        self.config = config or AgentConfig()
        self.tracing = tracing or Tracing()
        self.tenant_id = tenant_id
        self.guard = SQLGuard(GuardConfig(dialect=db.dialect, allowed_tables=self.config.allowed_tables))
        tables = db.schema()
        if self.config.allowed_tables is not None:
            tables = [t for t in tables if t.name.lower() in self.config.allowed_tables]
        self.schema_tables = tables
        self.retriever = SchemaRetriever(tables, embedder=embedder, k=self.config.top_k_tables)
        self.graph = self._build_graph()

    # ---- graph ---------------------------------------------------------------------------
    def _build_graph(self) -> Any:
        g: StateGraph[AgentState] = StateGraph(AgentState)
        g.add_node("plan", self._plan)
        g.add_node("retrieve", self._retrieve)
        g.add_node("write_sql", self._write_sql)
        g.add_node("execute", self._execute)
        g.add_node("correct", self._correct)
        g.add_node("summarize", self._summarize)
        g.add_node("schema_answer", self._schema_answer)
        g.add_node("respond", self._respond)
        g.add_node("fail", self._fail)
        g.add_node("finalize", self._finalize)
        g.add_edge(START, "plan")
        g.add_conditional_edges(
            "plan",
            route_after_plan,
            {"retrieve": "retrieve", "schema_answer": "schema_answer", "respond": "respond"},
        )
        g.add_edge("retrieve", "write_sql")
        g.add_edge("write_sql", "execute")
        max_retries = self.config.max_retries if self.config.self_correct else 0
        g.add_conditional_edges(
            "execute",
            lambda s: route_after_execute(s, max_retries),
            {"summarize": "summarize", "correct": "correct", "fail": "fail"},
        )
        g.add_edge("correct", "execute")
        for node in ("summarize", "schema_answer", "respond", "fail"):
            g.add_edge(node, "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    # ---- nodes ---------------------------------------------------------------------------
    def _plan(self, state: AgentState) -> dict[str, Any]:
        with self.tracing.step("plan") as span:
            intent = classify_intent(state.get("question", ""))
            span.set("sqlpilot.intent", intent)
            history: list[dict[str, str]] = []
            examples: list[PastQuery] = []
            if self.memory is not None and intent == "sql":
                if self.config.history_turns > 0:
                    history = self.memory.history(state["tenant_id"], state["session_id"], self.config.history_turns)
                if self.config.use_examples:
                    examples = self.memory.similar_queries(state["tenant_id"], state["question"], k=2)
            return {
                "intent": intent,
                "history": history,
                "examples": [{"question": e.question, "sql": e.sql} for e in examples],
                "attempts": 0,
                "error": None,
                "error_kind": None,
                "result": None,
                "failed": [],
            }

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        with self.tracing.step("retrieve") as span:
            if self.config.use_retrieval:
                found = self.retriever.retrieve(state["question"])
                tables, ddl = found.table_names, found.ddl()
            else:
                tables = [t.name for t in self.schema_tables]
                ddl = "\n\n".join(t.ddl() for t in self.schema_tables)
            span.set("sqlpilot.tables", ",".join(tables))
            return {"tables": tables, "schema_ddl": ddl}

    def _call_llm(self, messages: list[dict[str, str]], purpose: str, max_tokens: int) -> tuple[str, dict[str, int]]:
        with self.tracing.llm(self.llm.model, purpose) as span:
            try:
                resp = self.llm.chat(messages, max_tokens=max_tokens, temperature=0.0)
            except LLMError as err:
                span.error(err)
                raise
            span.usage(resp.prompt_tokens, resp.completion_tokens, resp.model)
            usage = {
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "llm_calls": 1,
            }
            return resp.text, usage

    def _examples(self, state: AgentState) -> list[PastQuery]:
        return [PastQuery(e["question"], e["sql"]) for e in state.get("examples") or []]

    def _write_sql(self, state: AgentState) -> dict[str, Any]:
        with self.tracing.step("write_sql") as span:
            messages = writer_messages(
                state["question"],
                state.get("schema_ddl", ""),
                self.db.dialect,
                self._examples(state),
                state.get("history") or [],
            )
            try:
                text, usage = self._call_llm(messages, "write_sql", self.config.writer_max_tokens)
            except LLMError as err:
                return {"sql": "", "attempts": 1, "error": str(err), "error_kind": "llm_error"}
            sql = extract_sql(text)
            span.set("sqlpilot.sql", sql)
            return {"sql": sql, "attempts": 1, **usage}

    def _correct(self, state: AgentState) -> dict[str, Any]:
        with self.tracing.step("correct", index=state.get("attempts")) as span:
            failed = [
                *(state.get("failed") or []),
                {"sql": state.get("sql", ""), "error": state.get("error") or ""},
            ]
            messages = writer_messages(
                state["question"],
                state.get("schema_ddl", ""),
                self.db.dialect,
                self._examples(state),
                state.get("history") or [],
                failed_sql=state.get("sql", ""),
                error=state.get("error"),
            )
            attempts = int(state.get("attempts") or 0) + 1
            try:
                text, usage = self._call_llm(messages, "fix_sql", self.config.writer_max_tokens)
            except LLMError as err:
                return {"failed": failed, "attempts": attempts, "error": str(err), "error_kind": "llm_error"}
            sql = extract_sql(text)
            span.set("sqlpilot.sql", sql)
            return {
                "sql": sql,
                "attempts": attempts,
                "failed": failed,
                "error": None,
                "error_kind": None,
                **usage,
            }

    def _execute(self, state: AgentState) -> dict[str, Any]:
        if state.get("error_kind") == "llm_error":
            return {}
        sql = state.get("sql", "")
        with (
            self.tracing.step("execute") as step,
            self.tracing.tool("run_readonly_sql", {"sql": sql}) as tool,
        ):
            if not sql:
                tool.error("no SQL produced")
                return {
                    "result": None,
                    "error": "The model did not produce a SQL query.",
                    "error_kind": "no_sql",
                }
            verdict = self.guard.check(sql)
            if not verdict.ok:
                step.set("sqlpilot.guard.reason", verdict.reason)
                tool.error(verdict.message)
                return {
                    "result": None,
                    "error": f"Rejected by the SQL guard: {verdict.message}",
                    "error_kind": "guard_rejected",
                }
            try:
                result = self.db.execute_readonly(
                    verdict.sql, max_rows=self.config.max_rows, timeout_s=self.config.timeout_s
                )
            except QueryTimeout as err:
                tool.error(err)
                return {"result": None, "error": str(err), "error_kind": "timeout"}
            except ExecutionError as err:
                tool.error(err)
                return {"result": None, "error": f"Database error: {err}", "error_kind": "execution_error"}
            tool.result(f"{result.row_count} rows{' (truncated)' if result.truncated else ''}")
            return {"result": result.to_dict(), "sql": verdict.sql, "error": None, "error_kind": None}

    def _summarize(self, state: AgentState) -> dict[str, Any]:
        with self.tracing.step("summarize") as span:
            result = state.get("result") or {}
            rows = result.get("rows") or []
            columns = result.get("columns") or []
            preview_lines = [" | ".join(map(str, columns))] + [
                " | ".join("NULL" if v is None else str(v) for v in row) for row in rows[:10]
            ]
            preview = "\n".join(preview_lines)
            count = len(rows)
            answer = _template_answer(columns, rows, bool(result.get("truncated")))
            usage: dict[str, int] = {}
            if self.config.llm_summary and count > 0:
                try:
                    text, usage = self._call_llm(
                        summary_messages(state["question"], state.get("sql", ""), count, preview),
                        "summarize",
                        self.config.summary_max_tokens,
                    )
                    if text.strip():
                        answer = text.strip()
                except LLMError as err:
                    span.error(err)
            span.output(answer)
            return {"answer": answer, **usage}

    def _schema_answer(self, state: AgentState) -> dict[str, Any]:
        with self.tracing.step("schema_answer"):
            lines = [
                f"- {t.name} ({', '.join(c.name for c in t.columns[:12])}{', ...' if len(t.columns) > 12 else ''})"
                for t in self.schema_tables
            ]
            return {
                "answer": f"The database has {len(lines)} tables:\n" + "\n".join(lines),
                "tables": [t.name for t in self.schema_tables],
            }

    def _respond(self, state: AgentState) -> dict[str, Any]:
        with self.tracing.step("respond"):
            intent = state.get("intent")
            if intent == "write_request":
                return {
                    "answer": _WRITE_ANSWER,
                    "error": "This assistant is read-only.",
                    "error_kind": "write_request",
                }
            if intent == "invalid":
                return {"answer": _INVALID_ANSWER, "error": "Invalid question.", "error_kind": "invalid"}
            return {"answer": _SMALLTALK_ANSWER}

    def _fail(self, state: AgentState) -> dict[str, Any]:
        with self.tracing.step("fail") as span:
            span.error(state.get("error") or "failed")
            return {"answer": f"I couldn't answer that: {state.get('error') or 'unknown error'}"}

    def _finalize(self, state: AgentState) -> dict[str, Any]:
        if self.memory is None:
            return {}
        with self.tracing.step("finalize"):
            tenant, session = state["tenant_id"], state["session_id"]
            self.memory.add_message(tenant, session, "user", state["question"])
            reply = state.get("answer", "")
            result = state.get("result")
            if state.get("sql") and result is not None:
                reply = f"{reply}\n(SQL: {state['sql']})"
                self.memory.record_query(tenant, state["question"], state["sql"], len(result["rows"]))
            self.memory.add_message(tenant, session, "assistant", reply)
            return {}

    # ---- public API ----------------------------------------------------------------------
    def list_tables(self) -> list[TableInfo]:
        """Tables visible to this agent (the tenant allow-list applied)."""
        return list(self.schema_tables)

    def describe_table(self, name: str) -> TableInfo:
        for table in self.schema_tables:
            if table.name.lower() == name.lower():
                return table
        raise KeyError(f"Unknown table: {name}")

    def run_sql(self, sql: str, max_rows: int | None = None) -> QueryResult:
        """Guard + execute a caller-supplied query. Raises GuardRejected / ExecutionError."""
        verdict = self.guard.check(sql)
        if not verdict.ok:
            raise GuardRejected(verdict)
        limit = self.config.max_rows if max_rows is None else max(1, min(max_rows, self.config.max_rows))
        with self.tracing.tool("run_readonly_sql", {"sql": verdict.sql}) as span:
            try:
                result = self.db.execute_readonly(verdict.sql, max_rows=limit, timeout_s=self.config.timeout_s)
            except ExecutionError as err:
                span.error(err)
                raise
            span.result(f"{result.row_count} rows")
            return result

    def _initial(self, question: str, session_id: str) -> AgentState:
        return {
            "question": question,
            "tenant_id": self.tenant_id,
            "session_id": session_id,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_calls": 0,
        }

    def stream(self, question: str, session_id: str = "default") -> Iterator[dict[str, Any]]:
        """Yield ``{"type": "step", ...}`` per graph node, then ``{"type": "result", ...}``."""
        start = time.perf_counter()
        state: dict[str, Any] = dict(self._initial(question, session_id))
        with self.tracing.run(question, self.tenant_id, session_id) as run:
            for update in self.graph.stream(state, stream_mode="updates"):
                for node, delta in update.items():
                    delta = delta or {}
                    for key, value in delta.items():
                        if key in ("prompt_tokens", "completion_tokens", "llm_calls"):
                            state[key] = int(state.get(key) or 0) + int(value or 0)
                        else:
                            state[key] = value
                    yield {"type": "step", "node": node, "detail": _public_step(node, delta)}
            result = _to_result(state, (time.perf_counter() - start) * 1000)
            run.set("sqlpilot.ok", result.ok)
            run.output(result.answer)
            # Totals as attributes, not usage: the LLM spans already carry usage, and tracers that
            # aggregate usage over a trace (agentlens) would otherwise count it twice.
            run.set("sqlpilot.prompt_tokens", result.prompt_tokens)
            run.set("sqlpilot.completion_tokens", result.completion_tokens)
            run.set("sqlpilot.llm_calls", result.llm_calls)
        yield {"type": "result", "result": result.to_dict()}

    def ask(self, question: str, session_id: str = "default") -> AgentResult:
        final: dict[str, Any] = {}
        for event in self.stream(question, session_id):
            if event["type"] == "result":
                final = event["result"]
        final.pop("ok", None)
        return AgentResult(**final)


def _template_answer(columns: list[str], rows: list[list[Any]], truncated: bool) -> str:
    if not rows:
        return "The query returned no rows."
    if len(rows) == 1 and len(columns) == 1:
        return f"{columns[0]}: {rows[0][0]}"
    more = "+" if truncated else ""
    return f"The query returned {len(rows)}{more} rows."


def _public_step(node: str, delta: dict[str, Any]) -> dict[str, Any]:
    """What a client sees for each step (no prompts, no raw history)."""
    keys = {
        "plan": ("intent",),
        "retrieve": ("tables",),
        "write_sql": ("sql", "error"),
        "correct": ("sql", "attempts", "error"),
        "execute": ("error", "error_kind"),
        "summarize": ("answer",),
        "schema_answer": ("answer",),
        "respond": ("answer",),
        "fail": ("answer",),
    }.get(node, ())
    detail = {k: delta[k] for k in keys if k in delta}
    if node == "plan":
        detail["examples"] = len(delta.get("examples") or [])
        detail["history_messages"] = len(delta.get("history") or [])
    if node == "execute" and delta.get("result") is not None:
        detail["row_count"] = len(delta["result"]["rows"])
        detail["truncated"] = delta["result"]["truncated"]
    return detail


def _to_result(state: dict[str, Any], latency_ms: float) -> AgentResult:
    return AgentResult(
        question=state.get("question", ""),
        intent=state.get("intent", ""),
        answer=state.get("answer", ""),
        sql=state.get("sql") or None,
        result=state.get("result"),
        error=state.get("error"),
        error_kind=state.get("error_kind"),
        attempts=int(state.get("attempts") or 0),
        tables=list(state.get("tables") or []),
        failed_attempts=list(state.get("failed") or []),
        prompt_tokens=int(state.get("prompt_tokens") or 0),
        completion_tokens=int(state.get("completion_tokens") or 0),
        llm_calls=int(state.get("llm_calls") or 0),
        latency_ms=round(latency_ms, 1),
    )
