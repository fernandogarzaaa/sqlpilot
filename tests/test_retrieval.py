from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from sqlpilot.db import Column, SQLiteDatabase, TableInfo
from sqlpilot.retrieval import BM25, SchemaRetriever, _cosine, _stem, tokenize


def T(name: str, *cols: str, refs: dict[str, str] | None = None, pk: str | None = "id") -> TableInfo:
    refs = refs or {}
    columns = [Column("id", "INTEGER", primary_key=True)] if pk else []
    columns += [Column(c, "TEXT", references=refs.get(c)) for c in cols]
    return TableInfo(name, tuple(columns))


SINGERS = [
    T("singer", "name", "country", "age"),
    T("concert", "concert_name", "theme", "year", "stadium_id", refs={"stadium_id": "stadium.id"}),
    T("stadium", "location", "capacity"),
    T(
        "singer_in_concert",
        "concert_id",
        "singer_id",
        refs={"concert_id": "concert.id", "singer_id": "singer.id"},
        pk=None,
    ),
]


# ---- tokenizer ---------------------------------------------------------------------------------


def test_tokenize_splits_identifiers_and_drops_stopwords() -> None:
    assert tokenize("InvoiceLine") == ["invoice", "line"]
    assert tokenize("singer_in_concert") == ["singer", "concert"]
    assert tokenize("How many singers are there?") == ["singer"]
    assert tokenize("") == []


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("countries", "country"),
        ("cities", "city"),
        ("ties", "tie"),  # too short for the -ies rule, falls through to -s
        ("classes", "class"),
        ("boxes", "box"),
        ("matches", "match"),
        ("dishes", "dish"),
        ("singers", "singer"),
        ("bus", "bus"),
        ("class", "class"),
        ("is", "is"),
        ("gas", "gas"),
        ("cars", "car"),
    ],
)
def test_stem(word: str, stem: str) -> None:
    assert _stem(word) == stem


def test_tokenize_lowercases_and_handles_digits() -> None:
    assert tokenize("Top10 Albums2024") == ["top10", "albums2024"]
    assert tokenize("albumID") == ["album", "id"]


# ---- BM25 --------------------------------------------------------------------------------------


def test_bm25_prefers_matching_documents() -> None:
    bm = BM25([["apple", "pie"], ["banana", "split"], ["apple", "apple", "tart"]])
    scores = bm.scores(["apple"])
    assert scores[1] == 0.0
    assert scores[0] > 0
    assert scores[2] > scores[0]


def test_bm25_rare_terms_weigh_more() -> None:
    bm = BM25([["common", "rare"], ["common"], ["common"]])
    assert bm.idf["rare"] > bm.idf["common"]


def test_bm25_exact_values() -> None:
    bm = BM25([["a", "b"], ["b", "c"]], k1=1.5, b=0.75)
    idf_a = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    assert bm.idf["a"] == pytest.approx(idf_a)
    # tf=1, doc length == avg length -> norm = k1
    assert bm.scores(["a"])[0] == pytest.approx(idf_a * 1 * 2.5 / (1 + 1.5))
    # duplicate query terms are counted once
    assert bm.scores(["a", "a"])[0] == pytest.approx(bm.scores(["a"])[0])


def test_bm25_length_normalisation() -> None:
    bm = BM25([["x", "pad", "pad", "pad", "pad", "pad"], ["x", "y"]])
    short, long_ = bm.scores(["x"])[1], bm.scores(["x"])[0]
    assert short > long_


def test_bm25_empty_corpus() -> None:
    assert BM25([]).scores(["x"]) == []
    assert BM25([[]]).scores(["x"]) == [0.0]


# ---- retriever ---------------------------------------------------------------------------------


def test_retrieves_relevant_table_first() -> None:
    r = SchemaRetriever(SINGERS, k=1).retrieve("What is the average age of all singers?")
    assert r.table_names == ["singer"]
    assert r.scores["singer"] == pytest.approx(1.0)


def test_bridge_table_is_added_for_joins() -> None:
    tables = [
        T("singer", "name", "age"),
        T("concert", "title", "year"),
        T("lineup", "slot", "a", "b", refs={"a": "singer.id", "b": "concert.id"}),
        T("venue", "city"),
    ]
    r = SchemaRetriever(tables, k=2).retrieve("Show each singer name and concert title")
    assert r.table_names == ["singer", "concert", "lineup"]
    assert r.bridges == ["lineup"]


def test_bridge_needs_two_selected_references() -> None:
    r = SchemaRetriever(SINGERS, k=1).retrieve("singer ages")
    assert r.bridges == []
    assert "singer_in_concert" not in r.table_names


def test_k_limits_scored_tables() -> None:
    r = SchemaRetriever(SINGERS, k=2).retrieve("stadium capacity")
    assert len([n for n in r.table_names if n not in r.bridges]) == 2
    assert r.table_names[0] == "stadium"


def test_name_coverage_prefers_exact_table() -> None:
    tables = [
        T(
            "InvoiceLine",
            "InvoiceId",
            "TrackId",
            "UnitPrice",
            "Quantity",
            refs={"InvoiceId": "Invoice.InvoiceId"},
        ),
        T("Invoice", "CustomerId", "InvoiceDate", "BillingCity", "Total"),
    ]
    r = SchemaRetriever(tables, k=1).retrieve("How many invoices are there?")
    assert r.table_names == ["Invoice"]
    plain = SchemaRetriever(tables, k=2, name_weight=0.0).score("How many invoices are there?")
    weighted = SchemaRetriever(tables, k=2, name_weight=0.3).score("How many invoices are there?")
    assert weighted["Invoice"] - weighted["InvoiceLine"] > plain["Invoice"] - plain["InvoiceLine"]


def test_score_formula() -> None:
    tables = [T("cats", "name"), T("dogs", "name")]
    retr = SchemaRetriever(tables, k=2, name_weight=0.5)
    scores = retr.score("cats")
    assert scores["cats"] == pytest.approx(1.0)  # 0.5 * 1 (bm25 max) + 0.5 * 1 (coverage)
    assert scores["dogs"] == pytest.approx(0.0)
    half = SchemaRetriever([T("cat_food", "brand"), T("dogs", "x")], k=2, name_weight=1.0).score("cat")
    assert half["cat_food"] == pytest.approx(0.5)


def test_no_match_falls_back_to_first_k_tables() -> None:
    r = SchemaRetriever(SINGERS, k=2).retrieve("zzz qqq")
    assert r.table_names == ["singer", "concert", "singer_in_concert"]
    assert all(v == 0 for v in r.scores.values())


def test_min_relative_score_drops_weak_tables() -> None:
    loose = SchemaRetriever(SINGERS, k=4, min_relative_score=0.0).retrieve("stadium capacity")
    strict = SchemaRetriever(SINGERS, k=4, min_relative_score=0.9).retrieve("stadium capacity")
    assert len(strict.table_names) < len(loose.table_names)
    assert strict.table_names == ["stadium"]


def test_min_relative_score_keeps_ties_at_threshold() -> None:
    tables = [T("alpha", "x"), T("beta", "x")]
    r = SchemaRetriever(tables, k=2, min_relative_score=1.0).retrieve("x")
    assert sorted(r.table_names) == ["alpha", "beta"]


def test_wide_tables_are_pruned_but_keep_keys() -> None:
    cols = [Column("id", "INT", primary_key=True), Column("owner_id", "INT", references="owner.id")]
    cols += [Column(f"metric_{i}", "REAL") for i in range(40)] + [Column("revenue", "REAL")]
    wide = TableInfo("facts", tuple(cols))
    r = SchemaRetriever([wide], k=1, max_columns=5).retrieve("total revenue")
    names = [c.name for c in r.tables[0].columns]
    assert len(names) == 5
    assert names[:2] == ["id", "owner_id"]
    assert "revenue" in names
    # column order follows the original table
    assert names.index("revenue") == len(names) - 1


def test_narrow_tables_are_untouched() -> None:
    r = SchemaRetriever(SINGERS, k=1, max_columns=4).retrieve("singer")
    assert len(r.tables[0].columns) == 4
    r2 = SchemaRetriever(SINGERS, k=1, max_columns=3).retrieve("singer")
    assert len(r2.tables[0].columns) == 3


def test_ddl_joins_tables() -> None:
    r = SchemaRetriever(SINGERS, k=2).retrieve("stadium concert")
    assert r.ddl().count("CREATE TABLE") == len(r.tables)
    assert "\n\n" in r.ddl()


@pytest.mark.parametrize(
    "kwargs", [{"alpha": -0.1}, {"alpha": 1.1}, {"k": 0}, {"name_weight": 1.5}, {"name_weight": -0.5}]
)
def test_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SchemaRetriever(SINGERS, **kwargs)  # type: ignore[arg-type]


def test_boundary_parameters_are_valid() -> None:
    SchemaRetriever(SINGERS, alpha=0.0, name_weight=0.0, k=1)
    SchemaRetriever(SINGERS, alpha=1.0, name_weight=1.0)


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for text in texts:
            key = next((k for k in self.vectors if k in text), None)
            out.append(self.vectors[key] if key else [0.0, 0.0])
        return out


def test_dense_scores_can_override_lexical_ranking() -> None:
    tables = [T("singer", "name"), T("stadium", "capacity")]
    emb = FakeEmbedder({"CREATE TABLE singer": [1.0, 0.0], "CREATE TABLE stadium": [0.0, 1.0], "arena": [0.0, 1.0]})
    r = SchemaRetriever(tables, embedder=emb, alpha=1.0, k=1).retrieve("big arena")
    assert r.table_names == ["stadium"]
    assert r.scores["stadium"] == pytest.approx(1.0)
    SchemaRetriever(tables, embedder=emb, alpha=1.0, k=1).retrieve("big arena")


def test_dense_mixing_and_vector_cache() -> None:
    tables = [T("singer", "name"), T("stadium", "capacity")]
    emb = FakeEmbedder({"CREATE TABLE singer": [1.0, 0.0], "CREATE TABLE stadium": [0.0, 1.0], "arena": [0.0, 1.0]})
    retr = SchemaRetriever(tables, embedder=emb, alpha=0.5, name_weight=0.0, k=2)
    scores = retr.score("singer arena")
    assert scores["singer"] == pytest.approx(0.5)  # bm25 1.0 * 0.5 + cosine 0 * 0.5
    assert scores["stadium"] == pytest.approx(0.5)  # bm25 0 + cosine 1 * 0.5
    retr.score("again")
    assert emb.calls == 3  # tables embedded once, then one call per question


def test_cosine() -> None:
    assert _cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert _cosine([0, 0], [1, 1]) == 0.0
    assert _cosine([1, 1], [0, 0]) == 0.0
    assert _cosine([3, 4], [6, 8]) == pytest.approx(1.0)


def test_real_schema_chinook(chinook: SQLiteDatabase) -> None:
    retr = SchemaRetriever(chinook.schema(), k=3)
    assert set(retr.retrieve("Which artists have the most albums?").table_names[:2]) == {"Artist", "Album"}
    assert retr.retrieve("List the employees").table_names[0] == "Employee"
    assert retr.retrieve("tracks in each playlist").table_names[0] in {"PlaylistTrack", "Playlist", "Track"}
