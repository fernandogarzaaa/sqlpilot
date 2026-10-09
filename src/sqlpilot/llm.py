"""LLM backends: any OpenAI-compatible chat endpoint, plus deterministic mock and scripted backends.

* :class:`OpenAICompatibleLLM` talks to ``POST {base_url}/chat/completions``: llama.cpp's
  ``llama-server`` (the default, ``http://127.0.0.1:8080/v1``), vLLM, Ollama, OpenAI, OpenRouter...
* :class:`MockLLM` needs no model and no key. It writes simple but valid SQL from the schema in the
  prompt (``COUNT(*)`` for "how many", ``ORDER BY ... DESC LIMIT n`` for "top n", otherwise a
  ``SELECT ... LIMIT``), so the quickstart, the tests and CI run offline.
* :class:`ScriptedLLM` replays fixed replies; tests use it to drive the self-correction loop.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from sqlpilot.retrieval import tokenize

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "MockLLM",
    "OpenAICompatibleLLM",
    "OpenAIEmbedder",
    "ScriptedLLM",
    "extract_sql",
]

Message = dict[str, str]


class LLMError(RuntimeError):
    """The LLM endpoint failed or returned something unusable."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


class LLMClient(Protocol):
    model: str

    def chat(self, messages: Sequence[Message], max_tokens: int = 512, temperature: float = 0.0) -> LLMResponse: ...


_FENCE = re.compile(r"```(?:sql|sqlite|postgresql|postgres)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_START = re.compile(r"\b(WITH|SELECT)\b", re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull one SQL statement out of a model reply (code fences, prose, trailing explanations)."""
    match = _FENCE.search(text)
    body = match.group(1) if match else text
    start = _START.search(body)
    if start is None:
        return body.strip().rstrip(";").strip()
    body = body[start.start() :]
    # Stop at the first semicolon outside quotes; models often append explanations after it.
    in_single = in_double = False
    for i, ch in enumerate(body):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == ";" and not in_single and not in_double:
            return body[:i].strip()
    # No semicolon: drop a trailing prose paragraph separated by a blank line.
    return body.split("\n\n")[0].strip()


def _is_loopback(url: str) -> bool:
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class OpenAICompatibleLLM:
    """Chat completions over HTTP. Loopback endpoints bypass HTTP(S)_PROXY settings."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "local",
        api_key: str | None = None,
        timeout_s: float = 120.0,
        extra_body: dict[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.extra_body = extra_body or {}
        self._client = httpx.Client(
            timeout=timeout_s, headers=headers, trust_env=not _is_loopback(base_url), transport=transport
        )

    def chat(self, messages: Sequence[Message], max_tokens: int = 512, temperature: float = 0.0) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self.extra_body,
        }
        try:
            resp = self._client.post(f"{self.base_url}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as err:
            raise LLMError(f"LLM request to {self.base_url} failed: {err}") from err
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            model=str(data.get("model") or self.model),
        )

    def close(self) -> None:
        self._client.close()


class OpenAIEmbedder:
    """``POST {base_url}/embeddings``; for hybrid schema retrieval (llama-server needs ``--embeddings``)."""

    def __init__(
        self,
        base_url: str,
        model: str = "local",
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            timeout=60.0, headers=headers, trust_env=not _is_loopback(base_url), transport=transport
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            resp = self._client.post(f"{self.base_url}/embeddings", json={"model": self.model, "input": list(texts)})
            resp.raise_for_status()
            items = sorted(resp.json()["data"], key=lambda d: d["index"])
        except (httpx.HTTPError, KeyError, ValueError) as err:
            raise LLMError(f"Embedding request failed: {err}") from err
        return [list(map(float, item["embedding"])) for item in items]


# --------------------------------------------------------------------------------------------
# Offline backends
# --------------------------------------------------------------------------------------------

_TABLE_RE = re.compile(r"CREATE TABLE (\w+) \((.*?)\n\);", re.DOTALL)
_COL_RE = re.compile(r"^\s+(\w+)\s*([A-Za-z]*)", re.MULTILINE)
_NUMERIC = ("int", "real", "num", "dec", "float", "double")


def _approx_tokens(messages: Sequence[Message]) -> int:
    return sum(len(m.get("content", "")) for m in messages) // 4


@dataclass
class MockLLM:
    """Deterministic offline "model". Reads the task marker and schema from the prompt."""

    model: str = "mock"
    calls: list[list[Message]] = field(default_factory=list)

    def chat(self, messages: Sequence[Message], max_tokens: int = 512, temperature: float = 0.0) -> LLMResponse:
        self.calls.append(list(messages))
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        if "TASK: summarize" in system:
            text = self._summarize(user)
        else:
            text = "```sql\n" + self._write_sql(user) + "\n```"
        return LLMResponse(
            text=text,
            prompt_tokens=_approx_tokens(messages),
            completion_tokens=len(text) // 4,
            model=self.model,
        )

    @staticmethod
    def _question(user: str) -> str:
        matches = re.findall(r"^Question:\s*(.+)$", user, re.MULTILINE)
        return matches[-1].strip() if matches else user

    def _write_sql(self, user: str) -> str:
        tables = {name: _COL_RE.findall(body) for name, body in _TABLE_RE.findall(user)}
        if not tables:
            return "SELECT 1"
        question = self._question(user)
        q_tokens = tokenize(question)
        # The schema block is already ranked by relevance, so the first table wins ties.
        table = max(tables, key=lambda t: sum(tok in tokenize(t) for tok in q_tokens) * 10 - list(tables).index(t))
        columns = tables[table]
        lowered = question.lower()
        if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", lowered):
            return f"SELECT COUNT(*) AS count FROM {table}"
        top = re.search(r"\btop (\d+)\b", lowered)
        if top or re.search(r"\b(most|highest|largest|biggest|longest)\b", lowered):
            numeric = [c for c, typ in columns if typ.lower().startswith(_NUMERIC) and not c.lower().endswith("id")]
            mentioned = [c for c, _ in columns if set(tokenize(c)) & set(q_tokens)]
            order = next((c for c in mentioned if c in numeric), numeric[0] if numeric else columns[0][0])
            limit = int(top.group(1)) if top else 1
            return f"SELECT * FROM {table} ORDER BY {order} DESC LIMIT {limit}"
        return f"SELECT * FROM {table} LIMIT 10"

    @staticmethod
    def _summarize(user: str) -> str:
        rows = re.search(r"Rows returned:\s*(\d+)", user)
        n = rows.group(1) if rows else "some"
        preview = re.search(r"Result preview:\n(.*?)(?:\n\n|\Z)", user, re.DOTALL)
        first = ""
        if preview:
            lines = preview.group(1).strip().splitlines()
            if len(lines) > 1:
                first = f" First row: {lines[1].strip()}."
        return f"The query returned {n} row(s).{first}"


@dataclass
class ScriptedLLM:
    """Replays ``replies`` in order (then repeats the last one). Records every call."""

    replies: list[str]
    model: str = "scripted"
    calls: list[list[Message]] = field(default_factory=list)

    def chat(self, messages: Sequence[Message], max_tokens: int = 512, temperature: float = 0.0) -> LLMResponse:
        self.calls.append(list(messages))
        idx = min(len(self.calls) - 1, len(self.replies) - 1)
        text = self.replies[idx] if self.replies else ""
        return LLMResponse(
            text=text,
            prompt_tokens=_approx_tokens(messages),
            completion_tokens=len(text) // 4,
            model=self.model,
        )
