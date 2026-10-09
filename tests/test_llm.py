from __future__ import annotations

import json

import httpx
import pytest

from sqlpilot.llm import (
    LLMError,
    MockLLM,
    OpenAICompatibleLLM,
    OpenAIEmbedder,
    ScriptedLLM,
    _is_loopback,
    extract_sql,
)


@pytest.mark.parametrize(
    ("text", "sql"),
    [
        ("SELECT 1", "SELECT 1"),
        ("```sql\nSELECT * FROM t\n```", "SELECT * FROM t"),
        ("```SQL\nselect a from b;\n```\nThis query selects a.", "select a from b"),
        ("```\nSELECT 2\n```", "SELECT 2"),
        ("Here is the query:\nSELECT name FROM singer; -- done", "SELECT name FROM singer"),
        ("SELECT name FROM t WHERE x = 'a;b';", "SELECT name FROM t WHERE x = 'a;b'"),
        ('SELECT "we;ird" FROM t; trailing', 'SELECT "we;ird" FROM t'),
        ("WITH x AS (SELECT 1) SELECT * FROM x", "WITH x AS (SELECT 1) SELECT * FROM x"),
        ("SELECT 1\n\nThis returns one.", "SELECT 1"),
        ("I cannot answer that.", "I cannot answer that."),
        ("  ;  ", ""),
        ("", ""),
    ],
)
def test_extract_sql(text: str, sql: str) -> None:
    assert extract_sql(text) == sql


@pytest.mark.parametrize(
    ("url", "loopback"),
    [
        ("http://127.0.0.1:8080/v1", True),
        ("http://localhost:8080/v1", True),
        ("http://[::1]:8080/v1", True),
        ("https://api.openai.com/v1", False),
        ("http://10.0.0.5/v1", False),
        ("not a url", False),
    ],
)
def test_is_loopback(url: str, loopback: bool) -> None:
    assert _is_loopback(url) is loopback


def _transport(handler):  # type: ignore[no-untyped-def]
    return httpx.MockTransport(handler)


def test_openai_client_request_and_usage() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "qwen",
                "choices": [{"message": {"content": "SELECT 1"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    llm = OpenAICompatibleLLM(
        "http://127.0.0.1:9/v1/",
        model="m",
        api_key="sk-test",
        transport=_transport(handler),
        extra_body={"top_p": 0.9},
    )
    resp = llm.chat([{"role": "user", "content": "hi"}], max_tokens=7, temperature=0.2)
    assert resp.text == "SELECT 1"
    assert (resp.prompt_tokens, resp.completion_tokens, resp.model) == (12, 3, "qwen")
    assert seen["url"] == "http://127.0.0.1:9/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["max_tokens"] == 7 and body["temperature"] == 0.2 and body["model"] == "m" and body["top_p"] == 0.9
    llm.close()


def test_openai_client_without_key_or_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})

    resp = OpenAICompatibleLLM("http://127.0.0.1:9/v1", model="m", transport=_transport(handler)).chat([])
    assert resp.text == ""
    assert resp.prompt_tokens == 0 and resp.completion_tokens == 0
    assert resp.model == "m"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, text="nope"),
    ],
)
def test_openai_client_errors(response: httpx.Response) -> None:
    llm = OpenAICompatibleLLM("http://127.0.0.1:9/v1", transport=_transport(lambda r: response))
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "x"}])


def test_openai_client_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="failed"):
        OpenAICompatibleLLM("http://127.0.0.1:9/v1", transport=_transport(handler)).chat([])


def test_embedder() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["input"] == ["a", "b"]
        return httpx.Response(
            200, json={"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]}
        )

    emb = OpenAIEmbedder("http://127.0.0.1:9/v1", api_key="k", transport=_transport(handler))
    assert emb.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    bad = OpenAIEmbedder("http://127.0.0.1:9/v1", transport=_transport(lambda r: httpx.Response(500)))
    with pytest.raises(LLMError):
        bad.embed(["a"])


SCHEMA = (
    "Schema:\nCREATE TABLE Artist (\n  ArtistId INTEGER PRIMARY KEY,\n  Name NVARCHAR\n);\n\n"
    "CREATE TABLE Track (\n  TrackId INTEGER PRIMARY KEY,\n  Name NVARCHAR,\n  Milliseconds INTEGER,\n"
    "  Bytes INTEGER\n);"
)


def _ask(question: str) -> str:
    msgs = [
        {"role": "system", "content": "TASK: write_sql"},
        {"role": "user", "content": f"{SCHEMA}\n\nQuestion: {question}"},
    ]
    return extract_sql(MockLLM().chat(msgs).text)


def test_mock_llm_patterns() -> None:
    assert _ask("How many artists are there?") == "SELECT COUNT(*) AS count FROM Artist"
    assert _ask("number of tracks") == "SELECT COUNT(*) AS count FROM Track"
    assert _ask("top 3 tracks by bytes") == "SELECT * FROM Track ORDER BY Bytes DESC LIMIT 3"
    assert _ask("the longest track") == "SELECT * FROM Track ORDER BY Milliseconds DESC LIMIT 1"
    assert _ask("list artists") == "SELECT * FROM Artist LIMIT 10"
    assert _ask("something unrelated") == "SELECT * FROM Artist LIMIT 10"


def test_mock_llm_uses_last_question_and_handles_no_schema() -> None:
    msgs = [
        {"role": "system", "content": "TASK: write_sql"},
        {
            "role": "user",
            "content": f"{SCHEMA}\n\nQuestion: list artists\nSQL: x\n\nQuestion: how many tracks",
        },
    ]
    assert "COUNT" in MockLLM().chat(msgs).text
    assert "SELECT 1" in MockLLM().chat([{"role": "user", "content": "no schema"}]).text


def test_mock_llm_summary() -> None:
    llm = MockLLM()
    out = llm.chat(
        [
            {"role": "system", "content": "TASK: summarize"},
            {"role": "user", "content": "Rows returned: 2\nResult preview:\nname\nAC/DC\nAccept\n\nAnswer:"},
        ]
    )
    assert out.text == "The query returned 2 row(s). First row: AC/DC."
    assert len(llm.calls) == 1
    assert out.prompt_tokens > 0 and out.completion_tokens > 0


def test_scripted_llm_replays_then_repeats_last() -> None:
    llm = ScriptedLLM(["a", "b"])
    assert [llm.chat([]).text for _ in range(4)] == ["a", "b", "b", "b"]
    assert len(llm.calls) == 4
    assert ScriptedLLM([]).chat([]).text == ""
