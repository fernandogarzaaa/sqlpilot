"""Schema retrieval: pick the tables (and, for wide tables, the columns) a question needs.

Small local models get noticeably worse when a prompt carries a whole schema, so the SQL writer
only sees what this module selects. The default ranker is BM25 over one document per table
(table name boosted, column names, foreign-key targets), with identifiers split on ``_`` and
camelCase and a light plural stemmer. An optional :class:`Embedder` (any OpenAI-compatible
``/v1/embeddings`` endpoint) adds a dense score; the two are mixed with ``alpha``.

After ranking, *bridge tables* (tables with foreign keys into two or more selected tables, such
as ``singer_in_concert``) are added so join paths stay complete.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from sqlpilot.db import Column, TableInfo

__all__ = ["BM25", "Embedder", "RetrievalResult", "SchemaRetriever", "tokenize"]

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "for",
        "in",
        "on",
        "at",
        "to",
        "by",
        "with",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "how",
        "many",
        "much",
        "show",
        "list",
        "give",
        "find",
        "return",
        "me",
        "all",
        "each",
        "every",
        "that",
        "this",
        "these",
        "those",
        "there",
        "their",
        "its",
        "it",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "from",
        "as",
        "than",
        "then",
        "please",
        "tell",
        "number",
        "count",
        "total",
        "get",
    ]
)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[A-Za-z0-9]+")


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ses", "xes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-cased, stemmed word tokens; identifiers are split on ``_`` and camelCase."""
    out: list[str] = []
    for word in _WORD.findall(_CAMEL.sub(" ", text)):
        token = word.lower()
        if token in _STOPWORDS:
            continue
        out.append(_stem(token))
    return out


class BM25:
    """Okapi BM25 over pre-tokenized documents."""

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [Counter(doc) for doc in documents]
        self.lengths = [len(doc) for doc in documents]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def scores(self, query: Sequence[str]) -> list[float]:
        result: list[float] = []
        for doc, length in zip(self.docs, self.lengths):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len) if self.avg_len else self.k1
            for term in set(query):
                tf = doc.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            result.append(score)
        return result


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class RetrievalResult:
    tables: list[TableInfo]
    scores: dict[str, float] = field(default_factory=dict)
    bridges: list[str] = field(default_factory=list)

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def ddl(self) -> str:
        return "\n\n".join(t.ddl() for t in self.tables)


def _table_document(table: TableInfo) -> list[str]:
    tokens = tokenize(table.name) * 3
    for col in table.columns:
        tokens.extend(tokenize(col.name))
        if col.references:
            tokens.extend(tokenize(col.references.split(".")[0]))
    return tokens


def _references(table: TableInfo) -> set[str]:
    return {c.references.split(".")[0].lower() for c in table.columns if c.references}


class SchemaRetriever:
    """Ranks tables for a question and returns a pruned schema.

    ``k``: max tables chosen by score; ``min_relative_score``: drop tables scoring below this
    fraction of the best one; ``max_columns``: wide tables keep their keys plus the best-matching
    columns up to this many.
    """

    def __init__(
        self,
        tables: Sequence[TableInfo],
        embedder: Embedder | None = None,
        alpha: float = 0.5,
        k: int = 4,
        min_relative_score: float = 0.0,
        max_columns: int = 30,
        name_weight: float = 0.3,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not 0.0 <= name_weight <= 1.0:
            raise ValueError("name_weight must be between 0 and 1")
        if k < 1:
            raise ValueError("k must be at least 1")
        self.tables = list(tables)
        self.embedder = embedder
        self.alpha = alpha
        self.k = k
        self.min_relative_score = min_relative_score
        self.max_columns = max_columns
        self.name_weight = name_weight
        self._name_tokens = [set(tokenize(t.name)) for t in self.tables]
        self._bm25 = BM25([_table_document(t) for t in self.tables])
        self._table_vectors: list[list[float]] | None = None

    def _dense_scores(self, question: str) -> list[float] | None:
        if self.embedder is None or not self.tables:
            return None
        if self._table_vectors is None:
            self._table_vectors = self.embedder.embed([t.ddl() for t in self.tables])
        query_vec = self.embedder.embed([question])[0]
        return [_cosine(query_vec, vec) for vec in self._table_vectors]

    def score(self, question: str) -> dict[str, float]:
        """Score per table in [0, 1].

        Lexical score = ``(1 - name_weight) * BM25 / max(BM25) + name_weight * coverage``, where
        coverage is the fraction of the table name's tokens that appear in the question (so
        "invoices" prefers ``Invoice`` over ``InvoiceLine``). With an embedder, the lexical score
        is mixed with cosine similarity: ``(1 - alpha) * lexical + alpha * cosine``.
        """
        if not self.tables:
            return {}
        q_tokens = tokenize(question)
        q_set = set(q_tokens)
        raw = self._bm25.scores(q_tokens)
        top = max(raw)
        sparse = []
        for value, name_tokens in zip(raw, self._name_tokens):
            bm25 = value / top if top > 0 else 0.0
            coverage = len(name_tokens & q_set) / len(name_tokens) if name_tokens else 0.0
            sparse.append((1 - self.name_weight) * bm25 + self.name_weight * coverage)
        dense = self._dense_scores(question)
        if dense is None:
            combined = sparse
        else:
            # cosine can be negative; clamp so every score stays in [0, 1]
            combined = [max(0.0, (1 - self.alpha) * s + self.alpha * d) for s, d in zip(sparse, dense)]
        return {t.name: c for t, c in zip(self.tables, combined)}

    def _prune_columns(self, table: TableInfo, question_tokens: set[str]) -> TableInfo:
        if len(table.columns) <= self.max_columns:
            return table
        keep: list[Column] = [c for c in table.columns if c.primary_key or c.references]
        rest = [c for c in table.columns if c not in keep]
        rest.sort(key=lambda c: -len(question_tokens & set(tokenize(c.name))))
        keep.extend(rest[: max(0, self.max_columns - len(keep))])
        ordered = tuple(c for c in table.columns if c in keep)
        return TableInfo(name=table.name, columns=ordered, row_count=table.row_count)

    def retrieve(self, question: str) -> RetrievalResult:
        if not self.tables:
            return RetrievalResult(tables=[])
        scores = self.score(question)
        ranked = sorted(self.tables, key=lambda t: -scores[t.name])
        best = scores[ranked[0].name]
        chosen = [t for t in ranked[: self.k] if scores[t.name] >= best * self.min_relative_score]
        chosen_names = {t.name.lower() for t in chosen}
        bridges: list[str] = []
        for table in ranked:
            if table.name.lower() in chosen_names:
                continue
            if len(_references(table) & chosen_names) >= 2:
                bridges.append(table.name)
        selected = chosen + [t for t in ranked if t.name in bridges]
        q_tokens = set(tokenize(question))
        return RetrievalResult(
            tables=[self._prune_columns(t, q_tokens) for t in selected],
            scores=scores,
            bridges=bridges,
        )
