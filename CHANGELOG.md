# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- Schema retrieval: hybrid (BM25 + embeddings) scores are clamped to [0, 1] when cosine
  similarity is negative; empty-schema handling is simplified to one early return.
- MCP server passes its version to FastMCP on mcp 2.x and falls back cleanly on mcp 1.x.
- `auth.py` imports `tomllib` with a fallback to `tomli` that type-checks under mypy --strict.

### Added
- End-to-end MCP stdio test and stronger tests for the guard, retrieval, routing and auth
  modules (mutation scores in `docs/mutation/`).
- Spider dev benchmark results for Qwen2.5-0.5B-Instruct (Q4_K_M) in `bench/results/`.
- Full README: quickstart, benchmark, mutation scores, architecture, security model, limitations.

## [0.1.0] - 2026-10-09

### Added
- LangGraph agent: deterministic planner/router, BM25 schema retrieval with bridge-table expansion
  and column pruning (optional embeddings), SQL writer, guarded read-only executor, bounded
  self-correction loop, answer summarizer and a memory-persisting finalize step.
- SQL safety guard (sqlglot): single statement, SELECT / WITH / set operations only, write and DDL
  nodes rejected anywhere in the tree (including data-modifying CTEs and `SELECT ... INTO`),
  engine commands, row locks, dangerous functions and per-tenant table allow-lists.
- Read-only executors: SQLite (`mode=ro`, `query_only`, authorizer, progress-handler timeout) and
  Postgres (`READ ONLY` transaction, `statement_timeout`, always rolled back); row limits.
- Memory: per-session conversation history and past successful queries (few-shot examples),
  SQLite or Postgres, scoped by tenant.
- Multi-tenant FastAPI service: per-tenant API keys (SHA-256), per-tenant databases and limits,
  `/v1/ask/stream` Server-Sent Events showing every agent step.
- MCP server (`list_tables`, `describe_table`, `run_readonly_sql`, `ask`) with capability contracts,
  tool annotations and a `sqlpilot://capabilities` resource; works with mcp 1.x and 2.x.
- Single-page web UI (chat, steps, SQL, result table, table browser).
- OpenAI-compatible LLM client (llama.cpp default), offline mock and scripted backends.
- OpenTelemetry spans (GenAI semantic conventions) with an agentlens hook.
- `sqlpilot serve | mcp | ask | sql | bench | hash-key | demo-db` CLI.
- Spider dev execution-accuracy benchmark (zero-shot vs retrieval vs retrieval + self-correction).
- CI (lint, mypy, tests on 3.10-3.12 with Postgres, wheel and Docker smoke tests, non-blocking
  mutation job), release workflow, Dockerfile, docker-compose with seeded Chinook.
