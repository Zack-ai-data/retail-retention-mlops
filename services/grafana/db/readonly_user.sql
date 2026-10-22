-- A read-only login for Grafana, with access to a fixed list of tables and columns.
--
-- Run by the grafana-db-init service (docker compose --profile monitoring up -d grafana). Postgres
-- only runs sql/init on a brand-new volume, so this is a separate, repeatable step: it can be run
-- any number of times, it moves the password to whatever .env holds now, and it grants what exists
-- and reports what does not yet (see grant_wanted.sql).
--
-- Least privilege, on purpose:
--   * the role cannot write: it has no write privilege anywhere, and every session it opens is
--     read-only by default;
--   * it is granted the objects listed below, never "ALL TABLES", so a table added later is
--     invisible to it until it is added here;
--   * in the Airflow and MLflow databases it is granted named COLUMNS only, so it cannot read what
--     is stored beside them (connections, variables, run parameters, task arguments, log paths);
--   * in the warehouse it reads only the aggregate monitoring tables and the scoring-run log,
--     never a customer-level table.
-- To show something new on a dashboard, add exactly what it reads to the list below.

\set ON_ERROR_STOP on
\getenv ro_password GRAFANA_DB_PASSWORD
\getenv warehouse_db WAREHOUSE_DB

SELECT 'CREATE ROLE grafana_ro LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION'
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'grafana_ro') \gexec

SELECT format('ALTER ROLE grafana_ro WITH PASSWORD %L', :'ro_password') \gexec
ALTER ROLE grafana_ro CONNECTION LIMIT 10;
ALTER ROLE grafana_ro SET default_transaction_read_only = on;

-- The warehouse: when each scoring run happened and with which model versions, and the aggregate
-- tables the monitor stage writes (counts and rates, no customer rows).
\connect :"warehouse_db"
CREATE TEMP TABLE wanted (tbl text, cols text);
INSERT INTO wanted VALUES
    ('ml.scoring_runs', NULL),
    ('ml.monitor_snapshot_summary', NULL),
    ('ml.monitor_score_distribution', NULL),
    ('ml.monitor_segment_sizes', NULL),
    ('ml.monitor_model_versions', NULL),
    ('ml.monitor_model_metrics', NULL),
    ('ml.monitor_backtest', NULL);
\ir grant_wanted.sql

-- Airflow's database: whether the DAGs ran, and when. Not conf, executor_config, next_kwargs,
-- hostnames or anything in the connection, variable or log tables.
\connect airflow
CREATE TEMP TABLE wanted (tbl text, cols text);
INSERT INTO wanted VALUES
    ('dag', 'dag_id, is_paused, is_stale, has_import_errors, timetable_summary, last_parsed_time'),
    ('dag_run', 'dag_id, run_id, run_type, state, queued_at, logical_date, start_date, end_date'),
    ('task_instance', 'dag_id, task_id, run_id, map_index, state, try_number, start_date, end_date, duration'),
    ('asset_event', 'id, asset_id, source_dag_id, source_task_id, source_run_id, timestamp');
\ir grant_wanted.sql

-- MLflow's database: which model versions exist, which is champion, and what the gate decided.
-- Not the artifact locations, run sources or user names.
\connect mlflow
CREATE TEMP TABLE wanted (tbl text, cols text);
INSERT INTO wanted VALUES
    ('registered_models', 'name, creation_time, last_updated_time'),
    ('model_versions', 'name, version, creation_time, last_updated_time, current_stage, status'),
    ('model_version_tags', 'name, version, key, value'),
    ('registered_model_aliases', 'name, alias, version');
\ir grant_wanted.sql
