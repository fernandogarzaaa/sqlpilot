from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sqlpilot.auth import (
    MAX_KEY_LENGTH,
    MIN_KEY_LENGTH,
    Tenant,
    TenantConfigError,
    TenantRegistry,
    extract_api_key,
    hash_key,
    load_tenants,
)

KEY_A = "key-for-acme-123"
KEY_B = "key-for-globex-456"


def tenant(tid: str, *keys: str, **kw: object) -> Tenant:
    return Tenant(id=tid, database=f"{tid}.db", api_key_hashes=tuple(hash_key(k) for k in keys), **kw)  # type: ignore[arg-type]


@pytest.fixture
def registry() -> TenantRegistry:
    return TenantRegistry.from_tenants([tenant("acme", KEY_A, "second-acme-key"), tenant("globex", KEY_B)])


def test_hash_key_is_sha256_hex() -> None:
    assert hash_key("abc") == hashlib.sha256(b"abc").hexdigest()
    assert len(hash_key("x")) == 64


def test_authenticate_maps_keys_to_their_tenant(registry: TenantRegistry) -> None:
    acme = registry.authenticate(KEY_A)
    assert acme is not None and acme.id == "acme"
    assert registry.authenticate("second-acme-key").id == "acme"  # type: ignore[union-attr]
    assert registry.authenticate(KEY_B).id == "globex"  # type: ignore[union-attr]


@pytest.mark.parametrize("key", [None, "", "wrong-key", KEY_A.upper(), KEY_A + " ", hash_key(KEY_A)])
def test_authenticate_rejects_unknown_keys(registry: TenantRegistry, key: str | None) -> None:
    assert registry.authenticate(key) is None


def test_key_length_boundary() -> None:
    at_limit = "k" * MAX_KEY_LENGTH
    reg = TenantRegistry.from_tenants([tenant("t", at_limit), tenant("u", at_limit + "k")])
    assert reg.authenticate(at_limit).id == "t"  # type: ignore[union-attr]
    assert reg.authenticate(at_limit + "k") is None


def test_registry_lookup_and_len(registry: TenantRegistry) -> None:
    assert len(registry) == 2
    assert registry.get("acme").id == "acme"  # type: ignore[union-attr]
    assert registry.get("nope") is None


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(TenantConfigError, match="duplicate tenant id"):
        TenantRegistry.from_tenants([tenant("a", "k1-aaaaaa"), tenant("a", "k2-bbbbbb")])


def test_shared_keys_rejected() -> None:
    with pytest.raises(TenantConfigError, match="share an API key"):
        TenantRegistry.from_tenants([tenant("a", "same-key-1"), tenant("b", "same-key-1")])


def test_same_key_twice_in_one_tenant_is_fine() -> None:
    TenantRegistry.from_tenants([tenant("a", "same-key-1", "same-key-1")])


@pytest.mark.parametrize(
    ("authorization", "x_api_key", "expected"),
    [
        (None, "abc", "abc"),
        (None, "  abc  ", "abc"),
        ("Bearer xyz", None, "xyz"),
        ("bearer xyz", None, "xyz"),
        ("BEARER   xyz  ", None, "xyz"),
        ("Bearer xyz", "abc", "abc"),
        ("Basic dXNlcjpwYXNz", None, None),
        ("Bearer", None, None),
        ("Bearer    ", None, None),
        ("xyz", None, None),
        (None, None, None),
        (None, "   ", None),
        ("", "", None),
    ],
)
def test_extract_api_key(authorization: str | None, x_api_key: str | None, expected: str | None) -> None:
    assert extract_api_key(authorization, x_api_key) == expected


def test_public_view_hides_secrets() -> None:
    t = tenant("acme", KEY_A, name="Acme", allowed_tables=frozenset({"b", "a"}))
    pub = t.public()
    assert pub == {
        "id": "acme",
        "name": "Acme",
        "max_rows": 200,
        "timeout_s": 10.0,
        "allowed_tables": ["a", "b"],
    }
    assert "database" not in pub
    assert tenant("x", KEY_B).public()["name"] == "x"
    assert tenant("x", KEY_B).public()["allowed_tables"] is None


# ---- tenants file ------------------------------------------------------------------------------

GOOD = f"""
[[tenants]]
id = "acme"
name = "Acme"
database = "sqlite:///acme.db"
api_key_sha256 = ["{hash_key(KEY_A).upper()}"]
allowed_tables = ["Orders", "customers"]
max_rows = 50
timeout_s = 2.5

[[tenants]]
id = "globex_2"
database = "postgresql://reader@db/globex"
api_key_env = "GLOBEX_KEY"
"""


def test_load_tenants_file(tmp_path: Path) -> None:
    path = tmp_path / "tenants.toml"
    path.write_text(GOOD)
    reg = load_tenants(path, env={"GLOBEX_KEY": KEY_B})
    acme = reg.authenticate(KEY_A)
    assert acme is not None
    assert acme.name == "Acme"
    assert acme.allowed_tables == frozenset({"orders", "customers"})
    assert acme.max_rows == 50
    assert acme.timeout_s == 2.5
    globex = reg.authenticate(KEY_B)
    assert globex is not None and globex.id == "globex_2"
    assert globex.allowed_tables is None
    assert globex.max_rows == 200
    assert globex.timeout_s == 10.0


def test_env_key_is_optional_when_hashes_exist() -> None:
    data = {"tenants": [{"id": "a", "database": "a.db", "api_key_sha256": [hash_key(KEY_A)], "api_key_env": "X"}]}
    reg = load_tenants(data, env={})
    assert reg.authenticate(KEY_A) is not None
    assert len(reg.get("a").api_key_hashes) == 1  # type: ignore[union-attr]


def test_env_key_and_hash_deduplicated() -> None:
    data = {"tenants": [{"id": "a", "database": "a.db", "api_key_sha256": [hash_key(KEY_A)], "api_key_env": "X"}]}
    reg = load_tenants(data, env={"X": KEY_A})
    assert reg.get("a").api_key_hashes == (hash_key(KEY_A),)  # type: ignore[union-attr]


def test_env_key_minimum_length() -> None:
    data = {"tenants": [{"id": "a", "database": "a.db", "api_key_env": "X"}]}
    with pytest.raises(TenantConfigError, match="shorter than"):
        load_tenants(data, env={"X": "k" * (MIN_KEY_LENGTH - 1)})
    assert load_tenants(data, env={"X": "k" * MIN_KEY_LENGTH}).authenticate("k" * MIN_KEY_LENGTH) is not None


@pytest.mark.parametrize(
    ("tenant_data", "message"),
    [
        ({"id": "Bad Id", "database": "x.db", "api_key_sha256": [hash_key("k")]}, "invalid tenant id"),
        ({"id": "", "database": "x.db", "api_key_sha256": [hash_key("k")]}, "invalid tenant id"),
        ({"id": "-x", "database": "x.db", "api_key_sha256": [hash_key("k")]}, "invalid tenant id"),
        ({"id": "a" * 65, "database": "x.db", "api_key_sha256": [hash_key("k")]}, "invalid tenant id"),
        ({"database": "x.db", "api_key_sha256": [hash_key("k")]}, "invalid tenant id"),
        ({"id": "a", "api_key_sha256": [hash_key("k")]}, "'database' is required"),
        ({"id": "a", "database": "", "api_key_sha256": [hash_key("k")]}, "'database' is required"),
        ({"id": "a", "database": "x.db"}, "no API keys"),
        ({"id": "a", "database": "x.db", "api_key_sha256": ["abc"]}, "64 hex"),
        ({"id": "a", "database": "x.db", "api_key_sha256": ["z" * 64]}, "64 hex"),
        ({"id": "a", "database": "x.db", "api_key_sha256": [hash_key("k")], "max_rows": 0}, "max_rows"),
        ({"id": "a", "database": "x.db", "api_key_sha256": [hash_key("k")], "max_rows": 100_001}, "max_rows"),
        ({"id": "a", "database": "x.db", "api_key_sha256": [hash_key("k")], "timeout_s": 0}, "timeout_s"),
        ({"id": "a", "database": "x.db", "api_key_sha256": [hash_key("k")], "timeout_s": 300.5}, "timeout_s"),
    ],
)
def test_invalid_tenants(tenant_data: dict[str, object], message: str) -> None:
    with pytest.raises(TenantConfigError, match=message):
        load_tenants({"tenants": [tenant_data]}, env={})


@pytest.mark.parametrize(
    ("field", "value"), [("max_rows", 1), ("max_rows", 100_000), ("timeout_s", 300), ("timeout_s", 0.01)]
)
def test_limit_boundaries_are_valid(field: str, value: float) -> None:
    data = {"tenants": [{"id": "a", "database": "x.db", "api_key_sha256": [hash_key("k")], field: value}]}
    assert getattr(load_tenants(data, env={}).get("a"), field) == value


def test_tenant_id_boundaries() -> None:
    for tid in ("a", "a" * 64, "0team", "team_a-1"):
        data = {"tenants": [{"id": tid, "database": "x.db", "api_key_sha256": [hash_key("k")]}]}
        assert load_tenants(data, env={}).get(tid) is not None


@pytest.mark.parametrize("data", [{}, {"tenants": []}, {"tenants": "nope"}])
def test_tenants_file_needs_tenants(data: dict[str, object]) -> None:
    with pytest.raises(TenantConfigError, match="at least one"):
        load_tenants(data, env={})


def test_missing_and_invalid_files(tmp_path: Path) -> None:
    with pytest.raises(TenantConfigError, match="not found"):
        load_tenants(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[[tenants]\nid=")
    with pytest.raises(TenantConfigError, match="invalid TOML"):
        load_tenants(bad)


def test_load_tenants_defaults_to_process_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLOBEX_KEY", KEY_B)
    path = tmp_path / "t.toml"
    path.write_text(GOOD)
    assert load_tenants(path).authenticate(KEY_B) is not None
