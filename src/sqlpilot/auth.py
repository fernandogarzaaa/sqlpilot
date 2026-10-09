"""Multi-tenancy: tenants, API-key authentication and tenant configuration files.

Each tenant has its own database URL, its own API keys (stored only as SHA-256 hashes), an
optional table allow-list and its own row / time limits. Keys are compared with
:func:`hmac.compare_digest` against every configured hash, so lookup time does not depend on
which tenant (if any) a key belongs to.

Tenants file (TOML)::

    [[tenants]]
    id = "acme"
    name = "Acme Corp"
    database = "sqlite:///data/acme.db"          # or postgresql://reader:...@db/acme
    api_key_sha256 = ["5e88...", "..."]           # `sqlpilot hash-key <key>` prints one
    api_key_env = "ACME_API_KEY"                  # optional: also accept the key in this env var
    allowed_tables = ["orders", "customers"]      # optional allow-list
    max_rows = 200
    timeout_s = 10
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

__all__ = [
    "MAX_KEY_LENGTH",
    "Tenant",
    "TenantConfigError",
    "TenantRegistry",
    "extract_api_key",
    "hash_key",
    "load_tenants",
]

MAX_KEY_LENGTH = 256
MIN_KEY_LENGTH = 8
_TENANT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class TenantConfigError(ValueError):
    """The tenants file is invalid."""


def hash_key(key: str) -> str:
    """SHA-256 hex digest of an API key (what the tenants file stores)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Tenant:
    id: str
    database: str
    api_key_hashes: tuple[str, ...]
    name: str = ""
    allowed_tables: frozenset[str] | None = None
    max_rows: int = 200
    timeout_s: float = 10.0

    def public(self) -> dict[str, Any]:
        """Tenant info that is safe to return to the tenant itself (no keys, no DB URL)."""
        return {
            "id": self.id,
            "name": self.name or self.id,
            "max_rows": self.max_rows,
            "timeout_s": self.timeout_s,
            "allowed_tables": sorted(self.allowed_tables) if self.allowed_tables is not None else None,
        }


def extract_api_key(authorization: str | None, x_api_key: str | None) -> str | None:
    """API key from ``X-API-Key`` (preferred) or ``Authorization: Bearer <key>``."""
    if x_api_key:
        return x_api_key.strip() or None
    if authorization:
        scheme, _, credentials = authorization.strip().partition(" ")
        if scheme.lower() == "bearer" and credentials.strip():
            return credentials.strip()
    return None


@dataclass
class TenantRegistry:
    tenants: dict[str, Tenant] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: dict[str, str] = {}
        for tenant in self.tenants.values():
            for digest in tenant.api_key_hashes:
                if digest in seen and seen[digest] != tenant.id:
                    raise TenantConfigError(f"tenants {seen[digest]!r} and {tenant.id!r} share an API key")
                seen[digest] = tenant.id

    @classmethod
    def from_tenants(cls, tenants: Iterable[Tenant]) -> TenantRegistry:
        mapping: dict[str, Tenant] = {}
        for tenant in tenants:
            if tenant.id in mapping:
                raise TenantConfigError(f"duplicate tenant id {tenant.id!r}")
            mapping[tenant.id] = tenant
        return cls(mapping)

    def authenticate(self, api_key: str | None) -> Tenant | None:
        if not api_key or len(api_key) > MAX_KEY_LENGTH:
            return None
        digest = hash_key(api_key)
        match: Tenant | None = None
        for tenant in self.tenants.values():
            for expected in tenant.api_key_hashes:
                if hmac.compare_digest(digest, expected):
                    match = tenant
        return match

    def get(self, tenant_id: str) -> Tenant | None:
        return self.tenants.get(tenant_id)

    def __len__(self) -> int:
        return len(self.tenants)


def _tenant_from_mapping(raw: Mapping[str, Any], env: Mapping[str, str]) -> Tenant:
    tenant_id = raw.get("id")
    if not isinstance(tenant_id, str) or not _TENANT_ID.match(tenant_id):
        raise TenantConfigError(f"invalid tenant id {tenant_id!r} (lower-case letters, digits, '-', '_')")
    database = raw.get("database")
    if not isinstance(database, str) or not database:
        raise TenantConfigError(f"tenant {tenant_id!r}: 'database' is required")
    hashes = [str(h).lower() for h in raw.get("api_key_sha256", [])]
    for digest in hashes:
        if not _SHA256_HEX.match(digest):
            raise TenantConfigError(f"tenant {tenant_id!r}: api_key_sha256 entries must be 64 hex characters")
    env_name = raw.get("api_key_env")
    if env_name:
        key = env.get(str(env_name), "")
        if key:
            if len(key) < MIN_KEY_LENGTH:
                raise TenantConfigError(
                    f"tenant {tenant_id!r}: the key in ${env_name} is shorter than {MIN_KEY_LENGTH}"
                )
            hashes.append(hash_key(key))
    if not hashes:
        raise TenantConfigError(f"tenant {tenant_id!r}: no API keys (api_key_sha256 or api_key_env)")
    allowed = raw.get("allowed_tables")
    max_rows = int(raw.get("max_rows", 200))
    timeout_s = float(raw.get("timeout_s", 10.0))
    if not 1 <= max_rows <= 100_000:
        raise TenantConfigError(f"tenant {tenant_id!r}: max_rows must be between 1 and 100000")
    if not 0 < timeout_s <= 300:
        raise TenantConfigError(f"tenant {tenant_id!r}: timeout_s must be in (0, 300]")
    return Tenant(
        id=tenant_id,
        name=str(raw.get("name", "")),
        database=database,
        api_key_hashes=tuple(dict.fromkeys(hashes)),
        allowed_tables=frozenset(str(t).lower() for t in allowed) if allowed is not None else None,
        max_rows=max_rows,
        timeout_s=timeout_s,
    )


def load_tenants(source: str | Path | Mapping[str, Any], env: Mapping[str, str] | None = None) -> TenantRegistry:
    """Load a registry from a TOML file path or an already-parsed mapping."""
    env = os.environ if env is None else env
    if isinstance(source, Mapping):
        data: Mapping[str, Any] = source
    else:
        path = Path(source)
        try:
            data = tomllib.loads(path.read_text())
        except FileNotFoundError as err:
            raise TenantConfigError(f"tenants file not found: {path}") from err
        except tomllib.TOMLDecodeError as err:
            raise TenantConfigError(f"invalid TOML in {path}: {err}") from err
    raw_tenants = data.get("tenants")
    if not isinstance(raw_tenants, list) or not raw_tenants:
        raise TenantConfigError("the tenants file needs at least one [[tenants]] table")
    return TenantRegistry.from_tenants(_tenant_from_mapping(t, env) for t in raw_tenants)
