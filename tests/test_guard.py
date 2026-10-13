"""The SQL guard is the most security-critical module: these tests are deliberately exhaustive."""

from __future__ import annotations

import pytest

from sqlpilot.guard import (
    DENIED_FUNCTIONS,
    FORBIDDEN_NODES,
    GuardConfig,
    GuardRejected,
    GuardResult,
    SQLGuard,
    check_sql,
)

# ---------------------------------------------------------------------------------------------
# Allowed queries
# ---------------------------------------------------------------------------------------------

ALLOWED = [
    "SELECT 1",
    "select * from customers",
    "SELECT name FROM customers WHERE country = 'UK'",
    "SELECT c.name, SUM(o.quantity) AS n FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.name",
    "WITH big AS (SELECT * FROM orders WHERE quantity > 1) SELECT COUNT(*) FROM big",
    "SELECT name FROM customers UNION SELECT title FROM products",
    "SELECT name FROM customers INTERSECT SELECT name FROM customers",
    "SELECT name FROM customers EXCEPT SELECT name FROM customers WHERE id = 1",
    "(SELECT 1)",
    "SELECT * FROM customers;",
    "SELECT * FROM customers ;  ",
    "  \n SELECT 1 \n ",
    "/* leading comment */ SELECT 1 -- trailing comment",
    "SELECT * FROM customers WHERE name = 'x; DROP TABLE customers'",
    "SELECT * FROM customers WHERE name = 'DELETE FROM orders'",
    "SELECT 'insert' AS word, 'update' AS other",
    "SELECT COUNT(*), MAX(price), AVG(price), UPPER(title) FROM products GROUP BY title",
    "SELECT * FROM (SELECT id FROM customers) AS sub",
    "SELECT id FROM customers WHERE id IN (SELECT customer_id FROM orders)",
    "SELECT CASE WHEN price > 100 THEN 'high' ELSE 'low' END FROM products",
    "SELECT * FROM customers ORDER BY name DESC LIMIT 5 OFFSET 2",
    "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 5) SELECT * FROM n",
    'SELECT "name" FROM "customers"',
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allows_read_only_queries(sql: str) -> None:
    result = check_sql(sql)
    assert result.ok, result.message
    assert result.reason == ""
    assert bool(result) is True


def test_allowed_result_strips_trailing_semicolon_and_whitespace() -> None:
    assert check_sql("  SELECT * FROM customers ;  ").sql == "SELECT * FROM customers"
    assert check_sql("SELECT 1;").sql == "SELECT 1"
    assert check_sql("SELECT 1").sql == "SELECT 1"


def test_allowed_result_lists_referenced_tables_without_ctes_or_duplicates() -> None:
    result = check_sql(
        "WITH big AS (SELECT * FROM orders) SELECT * FROM big JOIN customers ON 1=1 JOIN orders o2 ON 1=1"
    )
    assert result.ok
    assert sorted(result.tables) == ["customers", "orders"]


# ---------------------------------------------------------------------------------------------
# Writes, DDL and engine commands
# ---------------------------------------------------------------------------------------------

WRITES = [
    ("INSERT INTO customers VALUES (9, 'x', 'y')", "not_select"),
    ("UPDATE customers SET name = 'x'", "not_select"),
    ("DELETE FROM customers", "not_select"),
    ("DROP TABLE customers", "not_select"),
    ("CREATE TABLE t (x INT)", "not_select"),
    ("CREATE TEMP TABLE t AS SELECT * FROM customers", "not_select"),
    ("ALTER TABLE customers ADD COLUMN x INT", "not_select"),
    ("REPLACE INTO customers VALUES (1, 'x', 'y')", "not_select"),
    ("PRAGMA writable_schema = 1", "not_select"),
    ("PRAGMA table_info(customers)", "not_select"),
    ("ATTACH DATABASE '/tmp/evil.db' AS evil", "not_select"),
    ("DETACH DATABASE evil", "not_select"),
    ("VACUUM", "not_select"),
    ("VACUUM INTO '/tmp/copy.db'", "not_select"),
    ("BEGIN", "not_select"),
    ("COMMIT", "not_select"),
    ("ROLLBACK", "not_select"),
    ("SET statement_timeout = 0", "not_select"),
    ("GRANT ALL ON customers TO public", "not_select"),
    ("TRUNCATE TABLE customers", "not_select"),
    ("EXPLAIN SELECT 1", "not_select"),
    ("VALUES (1), (2)", "not_select"),
]


@pytest.mark.parametrize(("sql", "reason"), WRITES)
def test_rejects_writes_ddl_and_commands(sql: str, reason: str) -> None:
    result = check_sql(sql)
    assert not result.ok
    assert result.reason == reason
    assert bool(result) is False
    assert "allowed" in result.message


@pytest.mark.parametrize(
    ("sql", "dialect"),
    [
        ("COPY customers TO '/tmp/out.csv'", "postgres"),
        ("COPY customers FROM PROGRAM 'rm -rf /'", "postgres"),
        ("DELETE FROM customers WHERE id = 1 RETURNING *", "postgres"),
        ("INSERT INTO customers SELECT * FROM customers", "postgres"),
        ("MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN DELETE", "postgres"),
        ("LOCK TABLE customers", "postgres"),
        ("CALL do_something()", "postgres"),
        ("SHOW search_path", "postgres"),
        ("DO $$ BEGIN PERFORM 1; END $$", "postgres"),
    ],
)
def test_rejects_postgres_commands(sql: str, dialect: str) -> None:
    result = check_sql(sql, dialect=dialect)
    assert not result.ok
    assert result.reason in {"not_select", "parse_error", "forbidden_statement"}


# ---------------------------------------------------------------------------------------------
# Multi-statement and injection attempts
# ---------------------------------------------------------------------------------------------

INJECTIONS = [
    "SELECT 1; DROP TABLE customers",
    "SELECT * FROM customers; DELETE FROM customers",
    "SELECT * FROM customers WHERE name = '' ; DROP TABLE customers; --'",
    "SELECT 1;SELECT 2",
    "SELECT 1; ATTACH DATABASE '/tmp/x.db' AS x",
    "SELECT 1; PRAGMA query_only = OFF",
    "SELECT 1 /* hidden */ ; /* hidden */ UPDATE customers SET name = 'pwned'",
    "SELECT 1;\n\nINSERT INTO customers VALUES (4, 'x', 'y')",
]


@pytest.mark.parametrize("sql", INJECTIONS)
def test_rejects_stacked_statements(sql: str) -> None:
    result = check_sql(sql)
    assert not result.ok
    assert result.reason == "multiple_statements"
    assert "got 2" in result.message or "got 3" in result.message


def test_multiple_statements_message_counts_statements() -> None:
    assert check_sql("SELECT 1; SELECT 2; SELECT 3").message == "Only one statement is allowed, got 3."


def test_rejects_data_modifying_cte() -> None:
    result = check_sql("WITH gone AS (DELETE FROM customers RETURNING *) SELECT * FROM gone", dialect="postgres")
    assert not result.ok
    assert result.reason == "forbidden_statement"
    assert "DELETE" in result.message


def test_rejects_insert_inside_cte() -> None:
    result = check_sql(
        "WITH x AS (INSERT INTO customers VALUES (5, 'a', 'b') RETURNING id) SELECT * FROM x",
        dialect="postgres",
    )
    assert not result.ok
    assert result.reason == "forbidden_statement"


def test_rejects_select_into() -> None:
    result = check_sql("SELECT * INTO stolen FROM customers", dialect="postgres")
    assert not result.ok
    assert result.reason == "forbidden_statement"
    assert "INTO" in result.message


def test_rejects_row_locks() -> None:
    result = check_sql("SELECT * FROM customers FOR UPDATE", dialect="postgres")
    assert not result.ok
    assert result.reason == "forbidden_statement"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT load_extension('/tmp/evil.so')",
        "SELECT LOAD_EXTENSION('/tmp/evil.so')",
        "SELECT \"load_extension\"('/tmp/evil.so')",
        "SELECT readfile('/etc/passwd')",
        "SELECT writefile('/tmp/x', 'data')",
        "SELECT pg_sleep(100)",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT pg_terminate_backend(1)",
        "SELECT lo_import('/etc/passwd')",
        "SELECT dblink('host=evil', 'select 1')",
        "SELECT query_to_xml('delete from customers', true, true, '')",
        "SELECT nextval('seq')",
        "SELECT setval('seq', 1)",
        "SELECT set_config('statement_timeout', '0', false)",
        "SELECT current_setting('data_directory')",
        "SELECT * FROM customers WHERE id = (SELECT sleep(10))",
        "SELECT name FROM customers WHERE pg_sleep(1) IS NULL",
        "SELECT 1 UNION SELECT load_extension('x')",
    ],
)
def test_rejects_dangerous_functions(sql: str) -> None:
    result = check_sql(sql, dialect="postgres")
    assert not result.ok
    assert result.reason == "forbidden_function"
    assert "is not allowed" in result.message


def test_denied_function_list_covers_known_vectors() -> None:
    assert {
        "load_extension",
        "readfile",
        "writefile",
        "dblink",
        "nextval",
        "query_to_xml",
        "sleep",
    } <= DENIED_FUNCTIONS


def test_function_prefix_rule_catches_unlisted_pg_functions() -> None:
    result = check_sql("SELECT pg_ls_dir('.')", dialect="postgres")
    assert result.reason == "forbidden_function"
    assert "pg_ls_dir" in result.message


def test_safe_functions_are_not_caught_by_prefix_rule() -> None:
    assert check_sql("SELECT upper(name), length(name), coalesce(country, '?') FROM customers").ok


def test_custom_function_denylist() -> None:
    guard = SQLGuard(GuardConfig(denied_functions=frozenset({"upper"}), denied_function_prefixes=()))
    assert guard.check("SELECT upper(name) FROM customers").reason == "forbidden_function"
    assert guard.check("SELECT pg_sleep(1)").ok


# ---------------------------------------------------------------------------------------------
# Table allow / deny lists
# ---------------------------------------------------------------------------------------------


def test_allow_list_blocks_other_tables() -> None:
    result = check_sql("SELECT * FROM secrets", allowed_tables=["customers", "orders"])
    assert not result.ok
    assert result.reason == "table_not_allowed"
    assert "secrets" in result.message


def test_allow_list_is_case_insensitive_and_permits_listed_tables() -> None:
    result = check_sql("SELECT * FROM Customers JOIN ORDERS ON 1=1", allowed_tables=["CUSTOMERS", "orders"])
    assert result.ok


def test_allow_list_checks_subqueries_and_unions() -> None:
    allowed = ["customers"]
    assert check_sql("SELECT * FROM customers WHERE id IN (SELECT id FROM secrets)", allowed_tables=allowed).reason == (
        "table_not_allowed"
    )
    assert check_sql("SELECT name FROM customers UNION SELECT value FROM secrets", allowed_tables=allowed).reason == (
        "table_not_allowed"
    )


def test_cte_names_do_not_count_as_tables() -> None:
    result = check_sql("WITH secrets AS (SELECT 1 AS x) SELECT * FROM secrets", allowed_tables=["customers"])
    assert result.ok


@pytest.mark.parametrize("dialect", ["sqlite", "postgres"])
@pytest.mark.parametrize(
    "sql",
    [
        # an inner-scope CTE named like a real table used to hide the outer real table
        "SELECT * FROM secrets, (WITH secrets AS (SELECT 1 AS x) SELECT x FROM secrets) t",
        "SELECT * FROM secrets WHERE 1 IN (WITH secrets AS (SELECT 1) SELECT * FROM secrets)",
        # non-recursive CTE reading a real table of its own name (Postgres reads the table)
        "WITH secrets AS (SELECT * FROM secrets) SELECT * FROM secrets",
        # a CTE is not visible to CTEs defined before it
        "WITH b AS (SELECT * FROM secrets), secrets AS (SELECT 1) SELECT * FROM b",
        # a schema-qualified name never refers to a CTE
        "WITH secrets AS (SELECT 1) SELECT * FROM main.secrets",
    ],
)
def test_cte_shadowing_cannot_bypass_allow_list(sql: str, dialect: str) -> None:
    result = check_sql(sql, dialect=dialect, allowed_tables=["customers"])
    assert not result.ok
    assert result.reason == "table_not_allowed"
    assert "secrets" in result.message


def test_cte_shadowing_cannot_bypass_deny_list() -> None:
    sql = "WITH pg_authid AS (SELECT * FROM pg_authid) SELECT * FROM pg_authid"
    assert check_sql(sql, dialect="postgres").reason == "forbidden_table"


@pytest.mark.parametrize("dialect", ["sqlite", "postgres"])
@pytest.mark.parametrize(
    "sql",
    [
        "WITH a AS (SELECT * FROM customers), b AS (SELECT * FROM a) SELECT * FROM b JOIN a ON 1 = 1",
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5) SELECT * FROM r",
        "SELECT * FROM (WITH a AS (SELECT * FROM customers) SELECT * FROM a) t",
        "WITH a AS (SELECT id FROM customers) SELECT * FROM customers WHERE id IN (SELECT id FROM a)",
    ],
)
def test_scoped_cte_references_still_allowed(sql: str, dialect: str) -> None:
    result = check_sql(sql, dialect=dialect, allowed_tables=["customers"])
    assert result.ok, result.message
    assert set(result.tables) <= {"customers"}


def test_empty_allow_list_blocks_every_table() -> None:
    assert check_sql("SELECT * FROM customers", allowed_tables=[]).reason == "table_not_allowed"
    assert check_sql("SELECT 1", allowed_tables=[]).ok


@pytest.mark.parametrize("table", ["pg_authid", "pg_catalog.pg_shadow", "PG_AUTHID"])
def test_system_deny_list(table: str) -> None:
    result = check_sql(f"SELECT * FROM {table}", dialect="postgres")
    assert not result.ok
    assert result.reason == "forbidden_table"


# ---------------------------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sql", ["", "   ", "\n\t", ";", " ; ; "])
def test_rejects_empty(sql: str) -> None:
    result = check_sql(sql)
    assert not result.ok
    assert result.reason == "empty"


@pytest.mark.parametrize("value", [None, 42, b"SELECT 1", ["SELECT 1"]])
def test_rejects_non_strings(value: object) -> None:
    result = SQLGuard().check(value)
    assert not result.ok
    assert result.reason == "invalid_type"
    assert result.sql == ""


def test_rejects_nul_bytes() -> None:
    result = check_sql("SELECT 1\x00; DROP TABLE customers")
    assert not result.ok
    assert result.reason == "nul_byte"


def test_length_limit_boundary() -> None:
    guard = SQLGuard(GuardConfig(max_chars=30))
    exact = "SELECT 12345678901234567890123"
    assert len(exact) == 30
    assert guard.check(exact).ok
    too_long = exact + "4"
    result = guard.check(too_long)
    assert not result.ok
    assert result.reason == "too_long"
    assert len(result.sql) <= 200
    assert "30" in result.message


def test_too_long_truncates_echoed_sql() -> None:
    big = "SELECT " + "1 + " * 10_000 + "1"
    result = SQLGuard().check(big)
    assert result.reason == "too_long"
    assert len(result.sql) == 200


@pytest.mark.parametrize("sql", ["SELEC * FROM customers", "SELECT * FROM (", "SELECT 'unterminated"])
def test_rejects_unparseable(sql: str) -> None:
    result = check_sql(sql)
    assert not result.ok
    assert result.reason in {"parse_error", "not_select"}


def test_parse_error_reason_and_message() -> None:
    result = check_sql("SELECT * FROM (")
    assert result.reason == "parse_error"
    assert result.message.startswith("Could not parse the SQL")


def test_message_names_the_statement_type() -> None:
    assert check_sql("DELETE FROM customers").message == ("Only read-only SELECT queries are allowed (got DELETE).")


def test_guard_result_defaults_and_exception() -> None:
    result = GuardResult(ok=False, sql="x", reason="r", message="m")
    err = GuardRejected(result)
    assert str(err) == "m"
    assert err.result is result
    assert GuardResult(ok=True, sql="SELECT 1").tables == ()


def test_forbidden_nodes_cover_writes() -> None:
    names = {n.__name__ for n in FORBIDDEN_NODES}
    assert {"Insert", "Update", "Delete", "Drop", "Create", "Alter", "Command", "Into", "Lock"} <= names


def test_dialect_is_passed_to_the_parser() -> None:
    # Postgres-only syntax parses as a SELECT under postgres.
    assert check_sql("SELECT name::text FROM customers", dialect="postgres").ok
    assert SQLGuard(GuardConfig(dialect="postgres")).config.dialect == "postgres"


def test_check_sql_without_allow_list_permits_any_table() -> None:
    assert check_sql("SELECT * FROM anything_at_all").ok


def test_default_length_limit_is_20000_characters() -> None:
    assert GuardConfig().max_chars == 20_000
    sql = "SELECT 1"
    at_limit = sql + " " * 0 + "+0" * ((20_000 - len(sql)) // 2)
    assert len(at_limit) == 20_000
    assert SQLGuard().check(at_limit).reason != "too_long"
    assert SQLGuard().check(at_limit + "1").reason == "too_long"
