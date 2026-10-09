"""Command line: ``sqlpilot serve | mcp | ask | bench | sql | hash-key | demo-db``."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from sqlpilot import __version__

_LLM_HELP = "LLM backend: 'openai' (any OpenAI-compatible endpoint, default llama.cpp) or 'mock' (offline)"


def _add_llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--llm", choices=["openai", "mock"], default=None, help=_LLM_HELP + " [env SQLPILOT_LLM]")
    p.add_argument("--llm-base-url", default=None, help="OpenAI-compatible base URL [env SQLPILOT_LLM_BASE_URL]")
    p.add_argument("--llm-model", default=None, help="model name sent to the endpoint [env SQLPILOT_LLM_MODEL]")


def _settings(args: argparse.Namespace) -> Any:
    from sqlpilot.config import Settings

    settings = Settings.from_env()
    if getattr(args, "llm", None):
        settings.llm = args.llm
    if getattr(args, "llm_base_url", None):
        settings.llm_base_url = args.llm_base_url
    if getattr(args, "llm_model", None):
        settings.llm_model = args.llm_model
    return settings


def _single_agent(args: argparse.Namespace) -> Any:
    from sqlpilot.agent import AgentConfig, SQLAgent
    from sqlpilot.config import build_llm, demo_db_path
    from sqlpilot.db import connect
    from sqlpilot.memory import open_memory
    from sqlpilot.tracing import Tracing, agentlens_from_env, setup_otel

    settings = _settings(args)
    setup_otel(settings.otel_exporter)
    url = args.db or os.environ.get("SQLPILOT_DATABASE_URL") or str(demo_db_path())
    memory = open_memory(settings.memory_url) if not getattr(args, "no_memory", False) else None
    config = AgentConfig(
        max_retries=settings.max_retries, top_k_tables=settings.top_k_tables, llm_summary=settings.llm_summary
    )
    return SQLAgent(
        connect(url),
        build_llm(settings),
        memory=memory,
        config=config,
        tracing=Tracing(agentlens_from_env()),
        tenant_id="cli",
    )


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from sqlpilot.api import create_app
    from sqlpilot.tracing import setup_otel

    if args.tenants:
        os.environ["SQLPILOT_TENANTS_FILE"] = args.tenants
    if args.demo_key:
        os.environ["SQLPILOT_DEMO_API_KEY"] = args.demo_key
    settings = _settings(args)
    setup_otel(settings.otel_exporter)
    if not settings.tenants_file and not settings.demo_api_key:
        print(
            "error: pass --tenants FILE or --demo-key KEY (or set SQLPILOT_TENANTS_FILE / SQLPILOT_DEMO_API_KEY)",
            file=sys.stderr,
        )
        return 2
    uvicorn.run(create_app(settings=settings), host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from sqlpilot.mcp_server import build_server

    server = build_server(_single_agent(args))
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    agent = _single_agent(args)
    if args.json:
        result = agent.ask(args.question, args.session)
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.ok else 1
    final: dict[str, Any] = {}
    for event in agent.stream(args.question, args.session):
        if event["type"] == "step":
            detail = ", ".join(f"{k}={v}" for k, v in event["detail"].items() if k not in {"answer", "sql"})
            print(f"  · {event['node']}{': ' + detail if detail else ''}", file=sys.stderr)
        else:
            final = event["result"]
    if final.get("sql"):
        print(f"\nSQL:\n  {final['sql']}\n")
    result_data = final.get("result")
    if result_data:
        cols = result_data["columns"]
        print(" | ".join(cols))
        for row in result_data["rows"][: args.rows]:
            print(" | ".join("NULL" if v is None else str(v) for v in row))
        if result_data["row_count"] > args.rows:
            print(f"... {result_data['row_count']} rows")
        print()
    print(final.get("answer", ""))
    print(
        f"\n[{final.get('latency_ms')} ms · {final.get('llm_calls')} LLM calls · "
        f"{final.get('prompt_tokens')}+{final.get('completion_tokens')} tokens]",
        file=sys.stderr,
    )
    return 0 if final.get("error") is None else 1


def cmd_sql(args: argparse.Namespace) -> int:
    from sqlpilot.guard import GuardRejected

    args.no_memory = True
    args.llm = "mock"
    agent = _single_agent(args)
    try:
        result = agent.run_sql(args.sql, args.rows)
    except GuardRejected as err:
        print(f"rejected ({err.result.reason}): {err.result.message}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from sqlpilot.bench import CONFIGS, run_benchmark
    from sqlpilot.config import build_llm

    settings = _settings(args)
    spider = Path(args.spider_dir)
    if not (spider / "dev.json").is_file():
        print(
            f"error: {spider}/dev.json not found. Run `make bench-data` (downloads Spider) first.",
            file=sys.stderr,
        )
        return 2
    configs = args.configs.split(",") if args.configs else list(CONFIGS)
    summary = run_benchmark(
        spider,
        build_llm(settings),
        configs,
        n=args.n,
        out_dir=Path(args.out),
        seed=args.seed,
        max_retries=settings.max_retries,
        top_k=settings.top_k_tables,
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_hash_key(args: argparse.Namespace) -> int:
    from sqlpilot.auth import hash_key

    print(hash_key(args.key))
    return 0


def cmd_demo_db(args: argparse.Namespace) -> int:
    from sqlpilot.config import demo_db_path

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(demo_db_path(), dest)
    print(f"Chinook sample database written to {dest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sqlpilot", description="Text-to-SQL analyst agent.")
    parser.add_argument("--version", action="version", version=f"sqlpilot {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the multi-tenant HTTP API and web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tenants", help="tenants TOML file [env SQLPILOT_TENANTS_FILE]")
    p.add_argument("--demo-key", help="serve a single 'demo' tenant on the bundled Chinook DB with this API key")
    p.add_argument("--log-level", default="info")
    _add_llm_args(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("mcp", help="run the MCP server over one database")
    p.add_argument("--db", help="database URL or SQLite path (default: bundled Chinook) [env SQLPILOT_DATABASE_URL]")
    p.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    _add_llm_args(p)
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser("ask", help="ask one question from the terminal")
    p.add_argument("question")
    p.add_argument("--db", help="database URL or SQLite path (default: bundled Chinook)")
    p.add_argument("--session", default="cli")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.add_argument("--rows", type=int, default=20, help="rows to print")
    p.add_argument("--no-memory", action="store_true", help="do not read or write conversation memory")
    _add_llm_args(p)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("sql", help="run SQL through the guard and the read-only executor")
    p.add_argument("sql")
    p.add_argument("--db", help="database URL or SQLite path (default: bundled Chinook)")
    p.add_argument("--rows", type=int, default=50)
    p.set_defaults(func=cmd_sql)

    p = sub.add_parser("bench", help="execution accuracy on a Spider dev sample")
    p.add_argument("--spider-dir", default=".bench_data/spider_data")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--configs", help="comma-separated subset of zero_shot,retrieval,retrieval_fix")
    p.add_argument("--out", default="bench/results/latest")
    _add_llm_args(p)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("hash-key", help="print the SHA-256 of an API key for the tenants file")
    p.add_argument("key")
    p.set_defaults(func=cmd_hash_key)

    p = sub.add_parser("demo-db", help="copy the bundled Chinook SQLite database to a path")
    p.add_argument("dest", nargs="?", default="chinook.sqlite")
    p.set_defaults(func=cmd_demo_db)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
