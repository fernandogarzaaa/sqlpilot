# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately to **fernandogarzaaa@gmail.com** (subject: `sqlpilot
security`) or through GitHub's *Report a vulnerability* button on the repository. Do not open a
public issue. You can expect an acknowledgement within 72 hours and a fix or mitigation plan
within 14 days for confirmed issues.

## Threat model

sqlpilot runs SQL written by an LLM (or sent directly to `/v1/sql` and `run_readonly_sql`) against
databases that belong to tenants. The design goal is that **no input can modify data, read
outside the tenant's database or allow-list, or exhaust the server**. Defences, in order:

1. **SQL guard** (`sqlpilot.guard`): exactly one statement; only `SELECT`, `WITH ... SELECT` and
   set operations; writes/DDL anywhere in the parse tree, engine commands (`PRAGMA`, `ATTACH`,
   `COPY`, `SET`, transactions...), `SELECT ... INTO`, row locks, dangerous functions
   (`load_extension`, `pg_*`, `dblink`, `query_to_xml`, `nextval`...) and tables outside the
   tenant allow-list are rejected.
2. **Read-only execution** (`sqlpilot.db`): SQLite is opened `mode=ro` with `query_only` and an
   authorizer that only allows reads; Postgres queries run in a `READ ONLY` transaction with a
   `statement_timeout` and are always rolled back.
3. **Limits**: per-tenant row caps, time limits and input size limits.
4. **Least privilege (your part)**: point each tenant at a database role with `SELECT` grants
   only (see `docker/initdb/02-readonly-role.sh`), and never at a superuser.
5. **Authentication**: API keys are stored as SHA-256 hashes and compared in constant time; keys
   are scoped to one tenant; conversation memory and stored examples are tenant-scoped.

## Known limitations

- The guard parses SQL with sqlglot; dialect features sqlglot cannot parse are rejected, not
  allowed, but a parser bug could still misclassify a statement. Layer 2 exists for that reason.
- Expensive read queries are bounded by the time limit, not by a cost estimate.
- Query results are sent to the configured LLM for summarisation. Use a local model (the default
  is llama.cpp) or disable summaries (`SQLPILOT_LLM_SUMMARY=false`) for sensitive data.
- The web UI keeps the API key in `localStorage`; serve it only over HTTPS in production.
- There is no built-in rate limiting; put the API behind a gateway for internet exposure.
