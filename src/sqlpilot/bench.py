"""Execution-accuracy benchmark on Spider dev (SQLite).

Compares three agent configurations on the same deterministic sample of Spider dev questions:

* ``zero_shot``      full schema in the prompt, one attempt, no self-correction;
* ``retrieval``      BM25 schema retrieval (top-k tables + bridge tables), one attempt;
* ``retrieval_fix``  schema retrieval plus the self-correction loop (``max_retries`` rewrites).

Execution accuracy (EX): the predicted and the gold query are both executed on the question's
SQLite database and their results compared. Rows are compared as multisets, or as ordered lists
when the gold query has ``ORDER BY``; floats are rounded to 4 places; a column permutation of the
gold result is accepted (as in the Spider test-suite evaluation) for up to 5 columns. This is a
close re-implementation, not the official evaluation script.

Results are appended to ``<out>/<config>.jsonl`` (resumable: finished question ids are skipped),
and ``<out>/summary.json`` / ``summary.md`` are rewritten at the end.
"""

from __future__ import annotations

import itertools
import json
import random
import re
import statistics
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlpilot.agent import AgentConfig, SQLAgent
from sqlpilot.db import ExecutionError, SQLiteDatabase
from sqlpilot.llm import LLMClient

__all__ = ["CONFIGS", "load_spider_sample", "results_match", "run_benchmark", "summarize"]

CONFIGS: dict[str, dict[str, Any]] = {
    "zero_shot": {"use_retrieval": False, "self_correct": False},
    "retrieval": {"use_retrieval": True, "self_correct": False},
    "retrieval_fix": {"use_retrieval": True, "self_correct": True},
}


@dataclass(frozen=True)
class Example:
    idx: int
    db_id: str
    question: str
    gold: str


def load_spider_sample(spider_dir: Path, n: int, seed: int = 42) -> list[Example]:
    """Deterministic sample of ``n`` questions from ``dev.json`` (shuffled with ``seed``)."""
    data = json.loads((spider_dir / "dev.json").read_text())
    order = list(range(len(data)))
    random.Random(seed).shuffle(order)  # noqa: S311 - deterministic sampling, not crypto
    picked = sorted(order[:n])
    return [Example(i, data[i]["db_id"], data[i]["question"], data[i]["query"]) for i in picked]


def _norm(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def results_match(pred: Sequence[Sequence[Any]], gold: Sequence[Sequence[Any]], ordered: bool) -> bool:
    """Compare two result sets (see module docstring for the rules)."""
    gold_rows = [tuple(_norm(v) for v in row) for row in gold]
    pred_rows = [tuple(_norm(v) for v in row) for row in pred]
    if len(gold_rows) != len(pred_rows):
        return False
    if not gold_rows:
        return True
    width = len(gold_rows[0])
    if any(len(r) != width for r in pred_rows):
        return False
    perms: Iterable[tuple[int, ...]]
    perms = itertools.permutations(range(width)) if width <= 5 else [tuple(range(width))]
    gold_counter = None if ordered else Counter(gold_rows)
    for perm in perms:
        candidate = [tuple(row[i] for i in perm) for row in pred_rows]
        if ordered and candidate == gold_rows:
            return True
        if not ordered and Counter(candidate) == gold_counter:
            return True
    return False


def _execute(db: SQLiteDatabase, sql: str) -> list[list[Any]] | None:
    try:
        return db.execute_readonly(sql, max_rows=100_000, timeout_s=30).rows
    except ExecutionError:
        return None


def run_benchmark(
    spider_dir: Path,
    llm: LLMClient,
    configs: Sequence[str],
    n: int = 150,
    out_dir: Path = Path("bench/results"),
    seed: int = 42,
    max_retries: int = 2,
    top_k: int = 4,
    log: Any = print,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = load_spider_sample(spider_dir, n, seed)
    dbs: dict[str, SQLiteDatabase] = {}

    def db_for(db_id: str) -> SQLiteDatabase:
        if db_id not in dbs:
            dbs[db_id] = SQLiteDatabase(spider_dir / "database" / db_id / f"{db_id}.sqlite")
        return dbs[db_id]

    for name in configs:
        if name not in CONFIGS:
            raise ValueError(f"unknown config {name!r}; choose from {sorted(CONFIGS)}")
        path = out_dir / f"{name}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(line)["idx"] for line in path.read_text().splitlines() if line.strip()}
        agents: dict[str, SQLAgent] = {}
        config = AgentConfig(
            **CONFIGS[name],
            max_retries=max_retries,
            top_k_tables=top_k,
            use_examples=False,
            llm_summary=False,
            history_turns=0,
            timeout_s=30,
        )
        with path.open("a") as fh:
            for ex in examples:
                if ex.idx in done:
                    continue
                if ex.db_id not in agents:
                    agents[ex.db_id] = SQLAgent(db_for(ex.db_id), llm, memory=None, config=config)
                start = time.perf_counter()
                res = agents[ex.db_id].ask(ex.question)
                latency = time.perf_counter() - start
                gold_rows = _execute(db_for(ex.db_id), ex.gold)
                pred_rows = res.result["rows"] if res.result else None
                ordered = bool(re.search(r"\border\s+by\b", ex.gold, re.IGNORECASE))
                correct = (
                    gold_rows is not None and pred_rows is not None and results_match(pred_rows, gold_rows, ordered)
                )
                record = {
                    "idx": ex.idx,
                    "db_id": ex.db_id,
                    "question": ex.question,
                    "gold": ex.gold,
                    "pred": res.sql,
                    "correct": correct,
                    "error_kind": res.error_kind,
                    "attempts": res.attempts,
                    "llm_calls": res.llm_calls,
                    "tables": res.tables,
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "latency_s": round(latency, 3),
                }
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                log(f"[{name}] #{ex.idx} {'OK ' if correct else 'BAD'} {latency:5.1f}s {ex.question[:60]}")
    for db in dbs.values():
        db.close()
    return summarize(out_dir, configs, llm.model, n, seed)


def summarize(out_dir: Path, configs: Sequence[str], model: str = "", n: int = 0, seed: int = 42) -> dict[str, Any]:
    summary: dict[str, Any] = {"model": model, "n": n, "seed": seed, "configs": {}}
    for name in configs:
        path = out_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        lat = sorted(r["latency_s"] for r in rows)
        summary["configs"][name] = {
            "questions": len(rows),
            "ex_pct": round(100 * sum(r["correct"] for r in rows) / len(rows), 1),
            "correct": sum(r["correct"] for r in rows),
            "exec_error_pct": round(100 * sum(r["error_kind"] is not None for r in rows) / len(rows), 1),
            "latency_mean_s": round(statistics.mean(lat), 2),
            "latency_p50_s": round(lat[len(lat) // 2], 2),
            "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2),
            "prompt_tokens_mean": round(statistics.mean(r["prompt_tokens"] for r in rows), 1),
            "completion_tokens_mean": round(statistics.mean(r["completion_tokens"] for r in rows), 1),
            "llm_calls_mean": round(statistics.mean(r["llm_calls"] for r in rows), 2),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        f"Model: `{model}` · Spider dev sample: {n} questions (seed {seed})",
        "",
        "| Config | EX % | Correct | Failed to execute % | Mean latency (s) | p50 / p95 (s) "
        "| Prompt tok / q | Completion tok / q | LLM calls / q |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summary["configs"].items():
        lines.append(
            f"| {name} | {s['ex_pct']} | {s['correct']}/{s['questions']} | {s['exec_error_pct']} | "
            f"{s['latency_mean_s']} | {s['latency_p50_s']} / {s['latency_p95_s']} | {s['prompt_tokens_mean']} | "
            f"{s['completion_tokens_mean']} | {s['llm_calls_mean']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return summary
