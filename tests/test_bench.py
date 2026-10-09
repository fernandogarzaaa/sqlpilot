from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sqlpilot.bench import load_spider_sample, results_match, run_benchmark, summarize
from sqlpilot.llm import MockLLM


@pytest.mark.parametrize(
    ("pred", "gold", "ordered", "match"),
    [
        ([[1], [2]], [[2], [1]], False, True),
        ([[1], [2]], [[2], [1]], True, False),
        ([[1], [2]], [[1], [2]], True, True),
        ([[1], [1]], [[1]], False, False),
        ([[1], [1], [2]], [[1], [2], [2]], False, False),
        ([], [], False, True),
        ([[1.00001]], [[1.0]], False, True),
        ([[1.01]], [[1.0]], False, False),
        ([["a", 1]], [[1, "a"]], False, True),  # column permutation
        ([["a", 1, 2]], [["a", 1]], False, False),
        ([[1, 2], [3]], [[1, 2], [3, 4]], False, False),
        ([[None]], [[None]], False, True),
    ],
)
def test_results_match(pred: list[list[object]], gold: list[list[object]], ordered: bool, match: bool) -> None:
    assert results_match(pred, gold, ordered) is match


def test_wide_results_do_not_permute() -> None:
    gold = [[1, 2, 3, 4, 5, 6]]
    assert results_match([[1, 2, 3, 4, 5, 6]], gold, False)
    assert not results_match([[6, 5, 4, 3, 2, 1]], gold, False)


def make_spider(tmp_path: Path) -> Path:
    root = tmp_path / "spider"
    db_dir = root / "database" / "pets"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "pets.sqlite")
    conn.executescript(
        "CREATE TABLE pets (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);"
        "INSERT INTO pets VALUES (1, 'Rex', 3), (2, 'Tom', 5);"
    )
    conn.close()
    dev = [
        {"db_id": "pets", "question": "How many pets are there?", "query": "SELECT count(*) FROM pets"},
        {"db_id": "pets", "question": "List the names of pets", "query": "SELECT name FROM pets"},
        {
            "db_id": "pets",
            "question": "What is the oldest pet?",
            "query": "SELECT name FROM pets ORDER BY age DESC LIMIT 1",
        },
    ]
    (root / "dev.json").write_text(json.dumps(dev))
    return root


def test_sample_is_deterministic_and_sorted(tmp_path: Path) -> None:
    root = make_spider(tmp_path)
    a = load_spider_sample(root, 2, seed=1)
    b = load_spider_sample(root, 2, seed=1)
    assert a == b
    assert [e.idx for e in a] == sorted(e.idx for e in a)
    assert len(load_spider_sample(root, 10)) == 3


def test_run_benchmark_end_to_end_and_resume(tmp_path: Path) -> None:
    root = make_spider(tmp_path)
    out = tmp_path / "out"
    logs: list[str] = []
    summary = run_benchmark(root, MockLLM(), ["zero_shot", "retrieval_fix"], n=3, out_dir=out, log=logs.append)
    zs = summary["configs"]["zero_shot"]
    assert zs["questions"] == 3
    assert zs["correct"] == 1  # the mock only gets COUNT(*) right
    assert zs["ex_pct"] == pytest.approx(33.3)
    assert zs["llm_calls_mean"] == 1
    assert (out / "summary.md").read_text().count("| zero_shot |") == 1
    assert len(logs) == 6
    # resume: nothing re-run
    logs.clear()
    run_benchmark(root, MockLLM(), ["zero_shot"], n=3, out_dir=out, log=logs.append)
    assert logs == []
    records = [json.loads(line) for line in (out / "zero_shot.jsonl").read_text().splitlines()]
    assert len(records) == 3
    assert {"idx", "pred", "gold", "correct", "latency_s", "prompt_tokens"} <= set(records[0])


def test_unknown_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown config"):
        run_benchmark(make_spider(tmp_path), MockLLM(), ["nope"], n=1, out_dir=tmp_path / "o")


def test_summarize_skips_missing(tmp_path: Path) -> None:
    (tmp_path / "retrieval.jsonl").write_text("")
    assert summarize(tmp_path, ["zero_shot", "retrieval"])["configs"] == {}
