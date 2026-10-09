# Contributing

Thanks for helping improve sqlpilot.

## Setup

```bash
git clone https://github.com/fernandogarzaaa/sqlpilot && cd sqlpilot
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Before you open a pull request

```bash
make lint        # ruff check + ruff format --check
make typecheck   # mypy (strict)
make test        # pytest; Postgres tests run when SQLPILOT_TEST_PG_URL is set (or `make test-pg`)
make mutation    # testteeth mutation score of the core modules
```

- **The SQL guard is security-critical.** Any change to `src/sqlpilot/guard.py` or the executors in
  `src/sqlpilot/db.py` needs tests for the new behaviour *and* for the bypass it might open
  (stacked statements, writes hidden in CTEs, dangerous functions). Keep the mutation score of
  `guard.py` at its current level or higher.
- Keep the routing logic in `routing.py` pure (no I/O) so it stays cheap to test.
- Benchmark numbers in the README must come from `sqlpilot bench` output committed under
  `bench/results/`. Never edit them by hand; label anything that was not measured.
- Use [Conventional Commits](https://www.conventionalcommits.org/) style messages
  (`feat:`, `fix:`, `docs:`...) and add a line to `CHANGELOG.md` under *Unreleased*.

## Releasing

1. Bump `version` in `pyproject.toml` and `src/sqlpilot/__init__.py`, update `CHANGELOG.md`.
2. Tag: `git tag v0.2.0 && git push --tags`. The release workflow builds the sdist and wheel,
   attaches them to a GitHub Release and uploads to PyPI when the `PYPI_API_TOKEN` secret exists.
