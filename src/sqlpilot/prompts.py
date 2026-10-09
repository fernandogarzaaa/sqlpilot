"""Prompt templates. Each system prompt starts with a ``TASK:`` marker (the mock backend keys on it)."""

from __future__ import annotations

from collections.abc import Sequence

from sqlpilot.memory import PastQuery

WRITER_SYSTEM = """TASK: write_sql
You are an expert data analyst. You translate questions into one {dialect} SQL query.
Rules:
- Write exactly one read-only SELECT statement (WITH ... SELECT is fine). Never modify data.
- Use only tables and columns that appear in the schema. Quote identifiers only when needed.
- Return only the SQL inside a ```sql code block, with no explanation."""

FIX_INSTRUCTIONS = """The previous query failed.
Previous SQL:
```sql
{sql}
```
Error: {error}
Write a corrected query that answers the question. Return only the SQL in a ```sql code block."""

SUMMARY_SYSTEM = """TASK: summarize
You explain SQL query results to a business user in one to three sentences.
Use only facts and numbers that appear in the result preview. Do not invent values."""


def writer_messages(
    question: str,
    schema_ddl: str,
    dialect: str,
    examples: Sequence[PastQuery] = (),
    history: Sequence[dict[str, str]] = (),
    failed_sql: str | None = None,
    error: str | None = None,
) -> list[dict[str, str]]:
    parts = [f"Schema:\n{schema_ddl}"]
    if examples:
        shots = "\n\n".join(f"Question: {ex.question}\nSQL: {ex.sql}" for ex in examples)
        parts.append(f"Examples of past questions that were answered correctly:\n{shots}")
    if history:
        convo = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)
        parts.append(f"Conversation so far (use it to resolve follow-up questions):\n{convo}")
    parts.append(f"Question: {question}")
    if failed_sql is not None:
        parts.append(FIX_INSTRUCTIONS.format(sql=failed_sql, error=error or "unknown error"))
    return [
        {"role": "system", "content": WRITER_SYSTEM.format(dialect=dialect)},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def summary_messages(question: str, sql: str, row_count: int, preview: str) -> list[dict[str, str]]:
    user = f"Question: {question}\nSQL: {sql}\nRows returned: {row_count}\nResult preview:\n{preview}\n\nAnswer:"
    return [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": user}]
