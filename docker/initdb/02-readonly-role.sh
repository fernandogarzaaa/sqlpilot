#!/bin/sh
# Creates a SELECT-only role for sqlpilot on the chinook database (defence in depth on top of
# sqlpilot's guard and READ ONLY transactions).
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname chinook <<SQL
CREATE ROLE sqlpilot_reader LOGIN PASSWORD '${SQLPILOT_READER_PASSWORD:-reader}';
GRANT CONNECT ON DATABASE chinook TO sqlpilot_reader;
GRANT USAGE ON SCHEMA public TO sqlpilot_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO sqlpilot_reader;
ALTER ROLE sqlpilot_reader SET default_transaction_read_only = on;
ALTER ROLE sqlpilot_reader SET statement_timeout = '15s';
SQL
