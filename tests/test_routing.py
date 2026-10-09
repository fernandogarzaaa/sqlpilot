from __future__ import annotations

import pytest

from sqlpilot.routing import (
    MAX_QUESTION_CHARS,
    RETRYABLE_ERRORS,
    classify_intent,
    route_after_execute,
    route_after_plan,
)


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("How many customers are there?", "sql"),
        ("Top 5 artists by number of albums", "sql"),
        ("What is the total revenue per country?", "sql"),
        ("Show me the invoices from 2021", "sql"),
        ("which customer updated their address most recently", "sql"),
        ("list the deleted accounts", "sql"),
        ("hi", "smalltalk"),
        ("Hello!", "smalltalk"),
        ("  thanks  ", "smalltalk"),
        ("thank you.", "smalltalk"),
        ("good morning", "smalltalk"),
        ("who are you?", "smalltalk"),
        ("What can you do?", "smalltalk"),
        ("help", "smalltalk"),
        ("hi, how many tracks are there?", "sql"),
        ("What tables are there?", "schema"),
        ("which tables exist in this db", "schema"),
        ("Show me the schema", "schema"),
        ("describe the database structure", "schema"),
        ("list all tables", "schema"),
        ("Delete all customers", "write_request"),
        ("please drop the invoices table", "write_request"),
        ("can you update the price of track 1", "write_request"),
        ("could you remove duplicate rows", "write_request"),
        ("INSERT INTO artist VALUES (1)", "write_request"),
        ("truncate orders", "write_request"),
        ("create a new table", "write_request"),
        ("", "invalid"),
        ("    ", "invalid"),
    ],
)
def test_classify_intent(question: str, intent: str) -> None:
    assert classify_intent(question) == intent


@pytest.mark.parametrize("value", [None, 1, ["hi"], b"hi"])
def test_classify_non_string_is_invalid(value: object) -> None:
    assert classify_intent(value) == "invalid"


def test_question_length_boundary() -> None:
    assert classify_intent("x" * MAX_QUESTION_CHARS) == "sql"
    assert classify_intent("x" * (MAX_QUESTION_CHARS + 1)) == "invalid"
    # surrounding whitespace does not count towards the limit
    assert classify_intent("  " + "x" * MAX_QUESTION_CHARS + "  ") == "sql"


def test_schema_question_wins_over_smalltalk_only_when_not_pure_greeting() -> None:
    assert classify_intent("hello, what tables are there?") == "schema"


@pytest.mark.parametrize(
    ("intent", "route"),
    [
        ("sql", "retrieve"),
        ("schema", "schema_answer"),
        ("smalltalk", "respond"),
        ("write_request", "respond"),
        ("invalid", "respond"),
        (None, "respond"),
    ],
)
def test_route_after_plan(intent: str | None, route: str) -> None:
    assert route_after_plan({"intent": intent}) == route


def test_route_after_execute_success() -> None:
    assert route_after_execute({"error": None, "result": {"rows": []}, "attempts": 5}, max_retries=0) == "summarize"


def test_route_after_execute_missing_result_is_not_success() -> None:
    assert route_after_execute({"error": None, "result": None, "error_kind": None, "attempts": 1}, 2) == "fail"


@pytest.mark.parametrize("kind", sorted(RETRYABLE_ERRORS))
def test_retryable_errors_retry_until_budget_is_spent(kind: str) -> None:
    state = {"error": "boom", "result": None, "error_kind": kind}
    assert route_after_execute({**state, "attempts": 1}, max_retries=2) == "correct"
    assert route_after_execute({**state, "attempts": 2}, max_retries=2) == "correct"
    assert route_after_execute({**state, "attempts": 3}, max_retries=2) == "fail"


def test_zero_retries_never_corrects() -> None:
    state = {"error": "boom", "result": None, "error_kind": "execution_error", "attempts": 1}
    assert route_after_execute(state, max_retries=0) == "fail"
    assert route_after_execute({**state, "attempts": 0}, max_retries=0) == "correct"


@pytest.mark.parametrize("kind", ["timeout", "llm_error", "write_request", None])
def test_non_retryable_errors_fail(kind: str | None) -> None:
    state = {"error": "boom", "result": None, "error_kind": kind, "attempts": 1}
    assert route_after_execute(state, max_retries=5) == "fail"


def test_error_with_stale_result_is_not_success() -> None:
    state = {"error": "boom", "result": {"rows": []}, "error_kind": "execution_error", "attempts": 1}
    assert route_after_execute(state, max_retries=2) == "correct"


def test_missing_attempts_counts_as_zero() -> None:
    assert route_after_execute({"error": "x", "error_kind": "no_sql"}, max_retries=0) == "correct"


def test_retryable_set() -> None:
    assert frozenset({"execution_error", "guard_rejected", "no_sql"}) == RETRYABLE_ERRORS
