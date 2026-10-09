"""Pure routing logic for the agent graph (no I/O, so it is cheap to test and mutation-grade).

The planner classifies each question into an intent, and the conditional edges of the
LangGraph state machine call :func:`route_after_plan` and :func:`route_after_execute`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

__all__ = [
    "MAX_QUESTION_CHARS",
    "RETRYABLE_ERRORS",
    "Intent",
    "classify_intent",
    "route_after_execute",
    "route_after_plan",
]

Intent = Literal["sql", "schema", "smalltalk", "write_request", "invalid"]

MAX_QUESTION_CHARS = 2000

# Error kinds the self-correction loop may retry. Timeouts are not retried: a rewrite rarely
# makes a runaway query cheap, and each retry costs a full LLM call.
RETRYABLE_ERRORS = frozenset({"execution_error", "guard_rejected", "no_sql"})

_SMALLTALK = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank you|thx|good (morning|afternoon|evening)|who are you|"
    r"what can you do|help)\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_SCHEMA = re.compile(
    r"\b(what|which|list|show|describe|display)\b.{0,20}\b(tables|schema|columns of the database|"
    r"database structure|data model)\b",
    re.IGNORECASE,
)
_WRITE = re.compile(
    r"^\s*(please\s+|can you\s+|could you\s+)?(delete|drop|insert|update|truncate|alter|create|remove|"
    r"modify|rename|grant|revoke)\b",
    re.IGNORECASE,
)


def classify_intent(question: object) -> Intent:
    """Deterministic planner: decide what kind of turn this is before any LLM call."""
    if not isinstance(question, str):
        return "invalid"
    text = question.strip()
    if not text or len(text) > MAX_QUESTION_CHARS:
        return "invalid"
    if _SMALLTALK.match(text):
        return "smalltalk"
    if _WRITE.match(text):
        return "write_request"
    if _SCHEMA.search(text):
        return "schema"
    return "sql"


def route_after_plan(state: Mapping[str, Any]) -> str:
    """Edge out of the planner: ``retrieve`` for data questions, ``schema_answer``, or ``respond``."""
    intent = state.get("intent")
    if intent == "sql":
        return "retrieve"
    if intent == "schema":
        return "schema_answer"
    return "respond"


def route_after_execute(state: Mapping[str, Any], max_retries: int) -> str:
    """Edge out of the executor: ``summarize`` on success, ``correct`` to retry, ``fail`` otherwise.

    ``attempts`` counts SQL-writing attempts so far (the first write is attempt 1), so with
    ``max_retries=2`` the agent writes at most three queries.
    """
    if state.get("error") is None and state.get("result") is not None:
        return "summarize"
    attempts = int(state.get("attempts") or 0)
    if state.get("error_kind") in RETRYABLE_ERRORS and attempts <= max_retries:
        return "correct"
    return "fail"
