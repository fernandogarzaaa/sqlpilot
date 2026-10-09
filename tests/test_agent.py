from __future__ import annotations

from collections.abc import Sequence

import pytest

from sqlpilot.agent import AgentConfig, AgentResult, SQLAgent, _public_step, _template_answer
from sqlpilot.db import ExecutionError, SQLiteDatabase
from sqlpilot.guard import GuardRejected
from sqlpilot.llm import LLMError, LLMResponse, MockLLM, ScriptedLLM
from sqlpilot.memory import SQLiteMemoryStore


def nodes(agent: SQLAgent, question: str, session: str = "s") -> list[str]:
    return [e["node"] for e in agent.stream(question, session) if e["type"] == "step"]


def test_happy_path_mock(mock_agent: SQLAgent) -> None:
    result = mock_agent.ask("How many customers are there?")
    assert result.ok
    assert result.intent == "sql"
    assert result.sql == "SELECT COUNT(*) AS count FROM Customer"
    assert result.result is not None and result.result["rows"] == [[59]]
    assert result.answer == "count: 59"
    assert result.attempts == 1
    assert result.llm_calls == 1
    assert result.prompt_tokens > 0
    assert "Customer" in result.tables
    assert result.latency_ms >= 0


def test_stream_emits_every_node_then_result(mock_agent: SQLAgent) -> None:
    events = list(mock_agent.stream("How many tracks are there?", "s"))
    assert [e["node"] for e in events[:-1]] == [
        "plan",
        "retrieve",
        "write_sql",
        "execute",
        "summarize",
        "finalize",
    ]
    assert events[-1]["type"] == "result"
    execute = events[3]["detail"]
    assert execute["row_count"] == 1 and execute["truncated"] is False


def test_self_correction_recovers(shop_db: SQLiteDatabase) -> None:
    llm = ScriptedLLM(["SELECT nam FROM customers", "SELECT name FROM customers ORDER BY id"])
    agent = SQLAgent(shop_db, llm, config=AgentConfig(llm_summary=False))
    events = list(agent.stream("customer names", "s"))
    assert [e["node"] for e in events[:-1]] == [
        "plan",
        "retrieve",
        "write_sql",
        "execute",
        "correct",
        "execute",
        "summarize",
        "finalize",
    ]
    assert events[4]["detail"]["attempts"] == 2
    result = AgentResult(**{k: v for k, v in events[-1]["result"].items() if k != "ok"})
    assert result.ok
    assert result.attempts == 2
    assert result.sql == "SELECT name FROM customers ORDER BY id"
    assert result.failed_attempts == [{"sql": "SELECT nam FROM customers", "error": result.failed_attempts[0]["error"]}]
    assert "no such column" in result.failed_attempts[0]["error"]
    # the fix prompt carries the failed SQL and the error
    fix_prompt = llm.calls[-1][-1]["content"]
    assert "SELECT nam FROM customers" in fix_prompt
    assert "no such column" in fix_prompt


def test_guard_rejection_is_fed_back(shop_db: SQLiteDatabase) -> None:
    llm = ScriptedLLM(["DELETE FROM customers", "SELECT COUNT(*) FROM customers"])
    result = SQLAgent(shop_db, llm, config=AgentConfig(llm_summary=False)).ask("count customers")
    assert result.ok
    assert result.result["rows"] == [[3]]  # type: ignore[index]
    assert "Rejected by the SQL guard" in llm.calls[1][-1]["content"]
    assert shop_db.execute_readonly("SELECT COUNT(*) FROM customers").rows == [[3]]


def test_retry_budget_is_respected(shop_db: SQLiteDatabase) -> None:
    llm = ScriptedLLM(["SELECT broken FROM nowhere"])
    agent = SQLAgent(shop_db, llm, config=AgentConfig(llm_summary=False, max_retries=2))
    result = agent.ask("anything")
    assert not result.ok
    assert result.error_kind == "execution_error"
    assert result.attempts == 3
    assert len(llm.calls) == 3
    assert len(result.failed_attempts) == 2
    assert result.answer.startswith("I couldn't answer that")


def test_self_correct_disabled(shop_db: SQLiteDatabase) -> None:
    llm = ScriptedLLM(["SELECT broken FROM nowhere", "SELECT 1"])
    agent = SQLAgent(shop_db, llm, config=AgentConfig(llm_summary=False, self_correct=False, max_retries=5))
    result = agent.ask("anything")
    assert not result.ok
    assert len(llm.calls) == 1


def test_no_sql_is_retried(shop_db: SQLiteDatabase) -> None:
    llm = ScriptedLLM(["", "SELECT 1 AS one"])
    result = SQLAgent(shop_db, llm, config=AgentConfig(llm_summary=False)).ask("one")
    assert result.ok
    assert result.failed_attempts[0]["error"] == "The model did not produce a SQL query."


def test_timeout_is_not_retried(shop_db: SQLiteDatabase) -> None:
    slow = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n"
    llm = ScriptedLLM([slow, "SELECT 1"])
    result = SQLAgent(shop_db, llm, config=AgentConfig(llm_summary=False, timeout_s=0.2)).ask("count forever")
    assert result.error_kind == "timeout"
    assert len(llm.calls) == 1


class BrokenLLM:
    model = "broken"

    def chat(self, messages: Sequence[dict[str, str]], max_tokens: int = 512, temperature: float = 0.0) -> LLMResponse:
        raise LLMError("endpoint down")


def test_llm_failure_is_reported(shop_db: SQLiteDatabase) -> None:
    result = SQLAgent(shop_db, BrokenLLM()).ask("how many customers")
    assert not result.ok
    assert result.error_kind == "llm_error"
    assert "endpoint down" in result.answer


class FailSecondLLM:
    model = "flaky"

    def __init__(self) -> None:
        self.n = 0

    def chat(self, messages: Sequence[dict[str, str]], max_tokens: int = 512, temperature: float = 0.0) -> LLMResponse:
        self.n += 1
        if self.n == 1:
            return LLMResponse("SELECT nope FROM customers", 1, 1)
        raise LLMError("down during fix")


def test_llm_failure_during_correction(shop_db: SQLiteDatabase) -> None:
    result = SQLAgent(shop_db, FailSecondLLM()).ask("x")
    assert result.error_kind == "llm_error"
    assert result.attempts == 2


def test_summary_uses_llm_and_falls_back(shop_db: SQLiteDatabase) -> None:
    llm = ScriptedLLM(["SELECT name FROM customers", "Three customers: Ada, Linus and Grace."])
    result = SQLAgent(shop_db, llm, config=AgentConfig(llm_summary=True)).ask("customer names")
    assert result.answer == "Three customers: Ada, Linus and Grace."
    assert result.llm_calls == 2
    blank = ScriptedLLM(["SELECT name FROM customers", "   "])
    assert SQLAgent(shop_db, blank).ask("names").answer == "The query returned 3 rows."


def test_summary_skips_llm_for_empty_results(shop_db: SQLiteDatabase) -> None:
    llm = ScriptedLLM(["SELECT name FROM customers WHERE 1 = 0", "should not be used"])
    result = SQLAgent(shop_db, llm).ask("nobody")
    assert result.answer == "The query returned no rows."
    assert len(llm.calls) == 1


def test_template_answer() -> None:
    assert _template_answer(["n"], [], False) == "The query returned no rows."
    assert _template_answer(["n"], [[5]], False) == "n: 5"
    assert _template_answer(["a", "b"], [[1, 2]], False) == "The query returned 1 rows."
    assert _template_answer(["a"], [[1], [2]], True) == "The query returned 2+ rows."


@pytest.mark.parametrize(
    ("question", "intent", "error_kind"),
    [
        ("delete every invoice", "write_request", "write_request"),
        ("hello", "smalltalk", None),
        ("   ", "invalid", "invalid"),
    ],
)
def test_non_sql_turns_skip_the_llm(
    chinook: SQLiteDatabase, question: str, intent: str, error_kind: str | None
) -> None:
    llm = MockLLM()
    result = SQLAgent(chinook, llm).ask(question)
    assert result.intent == intent
    assert result.error_kind == error_kind
    assert result.sql is None
    assert llm.calls == []


def test_schema_question(chinook: SQLiteDatabase) -> None:
    result = SQLAgent(chinook, MockLLM()).ask("What tables are there?")
    assert result.intent == "schema"
    assert result.answer.startswith("The database has 11 tables:")
    assert "- Artist (ArtistId, Name)" in result.answer
    assert "..." in result.answer  # wide tables are abbreviated
    assert len(result.tables) == 11


def test_memory_history_examples_and_isolation(chinook: SQLiteDatabase, memory: SQLiteMemoryStore) -> None:
    llm = MockLLM()
    agent = SQLAgent(chinook, llm, memory=memory, config=AgentConfig(llm_summary=False), tenant_id="acme")
    agent.ask("How many artists are there?", "s1")
    history = memory.history("acme", "s1")
    assert history[0] == {"role": "user", "content": "How many artists are there?"}
    assert "(SQL: SELECT COUNT(*) AS count FROM Artist)" in history[1]["content"]
    events = list(agent.stream("How many artists have albums?", "s1"))
    assert events[0]["detail"] == {"intent": "sql", "examples": 1, "history_messages": 2}
    prompt = llm.calls[-1][-1]["content"]
    assert "Examples of past questions" in prompt
    assert "Conversation so far" in prompt
    other = SQLAgent(chinook, llm, memory=memory, config=AgentConfig(llm_summary=False), tenant_id="globex")
    events = list(other.stream("How many artists are there?", "s1"))
    assert events[0]["detail"] == {"intent": "sql", "examples": 0, "history_messages": 0}


def test_failed_queries_are_not_recorded_as_examples(shop_db: SQLiteDatabase, memory: SQLiteMemoryStore) -> None:
    agent = SQLAgent(shop_db, ScriptedLLM(["SELECT broken"]), memory=memory, config=AgentConfig(max_retries=0))
    agent.ask("broken question", "s")
    assert memory.similar_queries("default", "broken question") == []
    assert len(memory.history("default", "s")) == 2


def test_memory_knobs(chinook: SQLiteDatabase, memory: SQLiteMemoryStore) -> None:
    memory.record_query("default", "How many artists are there?", "SELECT 1", 1)
    memory.add_message("default", "s", "user", "earlier")
    cfg = AgentConfig(llm_summary=False, use_examples=False, history_turns=0)
    events = list(SQLAgent(chinook, MockLLM(), memory=memory, config=cfg).stream("How many artists?", "s"))
    assert events[0]["detail"] == {"intent": "sql", "examples": 0, "history_messages": 0}


def test_zero_shot_uses_full_schema(chinook: SQLiteDatabase) -> None:
    llm = MockLLM()
    result = SQLAgent(chinook, llm, config=AgentConfig(use_retrieval=False, llm_summary=False)).ask("How many genres?")
    assert len(result.tables) == 11
    assert llm.calls[0][-1]["content"].count("CREATE TABLE") == 11
    llm2 = MockLLM()
    small = SQLAgent(chinook, llm2, config=AgentConfig(top_k_tables=2, llm_summary=False)).ask("How many genres?")
    assert len(small.tables) < 11


def test_allowed_tables_restrict_schema_and_sql(shop_db: SQLiteDatabase) -> None:
    cfg = AgentConfig(llm_summary=False, allowed_tables=frozenset({"customers", "orders"}), max_retries=0)
    agent = SQLAgent(shop_db, ScriptedLLM(["SELECT * FROM secrets"]), config=cfg)
    assert [t.name for t in agent.list_tables()] == ["customers", "orders"]
    with pytest.raises(KeyError):
        agent.describe_table("secrets")
    assert agent.describe_table("CUSTOMERS").name == "customers"
    result = agent.ask("show secrets")
    assert result.error_kind == "guard_rejected"
    with pytest.raises(GuardRejected) as info:
        agent.run_sql("SELECT * FROM secrets")
    assert info.value.result.reason == "table_not_allowed"


def test_run_sql_limits(shop_db: SQLiteDatabase) -> None:
    agent = SQLAgent(shop_db, MockLLM(), config=AgentConfig(max_rows=2))
    assert agent.run_sql("SELECT * FROM orders").row_count == 2
    assert agent.run_sql("SELECT * FROM orders", max_rows=1).row_count == 1
    assert agent.run_sql("SELECT * FROM orders", max_rows=100).row_count == 2
    assert agent.run_sql("SELECT * FROM orders", max_rows=0).row_count == 1
    with pytest.raises(ExecutionError):
        agent.run_sql("SELECT nope FROM orders")
    with pytest.raises(GuardRejected):
        agent.run_sql("DROP TABLE orders")


def test_result_serialisation(mock_agent: SQLAgent) -> None:
    data = mock_agent.ask("How many albums?").to_dict()
    assert data["ok"] is True
    rebuilt = AgentResult(**{k: v for k, v in data.items() if k != "ok"})
    assert rebuilt.ok


def test_public_step_hides_internals() -> None:
    assert _public_step("plan", {"intent": "sql", "history": [1, 2], "examples": [1]}) == {
        "intent": "sql",
        "examples": 1,
        "history_messages": 2,
    }
    assert _public_step(
        "execute", {"error": None, "error_kind": None, "result": {"rows": [[1]], "truncated": True}}
    ) == {"error": None, "error_kind": None, "row_count": 1, "truncated": True}
    assert _public_step("execute", {"error": "x", "error_kind": "timeout", "result": None}) == {
        "error": "x",
        "error_kind": "timeout",
    }
    assert _public_step("finalize", {"anything": 1}) == {}
    assert _public_step("write_sql", {"sql": "S", "prompt_tokens": 9}) == {"sql": "S"}
