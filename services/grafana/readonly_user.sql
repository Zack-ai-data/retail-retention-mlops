-- A read-only login for Grafana, with access to a fixed list of tables and columns.
--
-- Run by the grafana-db-init service (docker compose --profile monitoring up -d grafana). Postgres
-- only runs sql/init on a brand-new volume, so this is a separate, repeatable step: it can be run
-- any number of times and also moves the password to whatever .env holds now.
--
-- Least privilege, on purpose:
--   * the role cannot write: it has no write privilege anywhere, and every session it opens is
--     read-only by default;
--   * it is granted named objects, never "ALL TABLES", so a table added later is invisible to it
--     until it is added here;
--   * in the Airflow and MLflow databases it is granted named COLUMNS only, so it cannot read what
--     is stored beside them (connections, variables, run parameters, task arguments, log paths);
--   * the dashboards show aggregates, so in the warehouse it never needs customer-level tables.
-- To show something new on a dashboard, add exactly what it reads below.

\set ON_ERROR_STOP on
\getenv ro_password GRAFANA_DB_PASSWORD
\getenv warehouse_db WAREHOUSE_DB

SELECT 'CREATE ROLE grafana_ro LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION'
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'grafana_ro') \gexec

SELECT format('ALTER ROLE grafana_ro WITH PASSWORD %L', :'ro_password') \gexec
ALTER ROLE grafana_ro CONNECTION LIMIT 10;
ALTER ROLE grafana_ro SET default_transaction_read_only = on;

-- The warehouse: scoring runs (when, and with which model versions).
\connect :"warehouse_db"
GRANT USAGE ON SCHEMA ml TO grafana_ro;
GRANT SELECT ON ml.scoring_runs TO grafana_ro;

-- Airflow's database: whether the DAGs ran, and when. Not conf, executor_config, next_kwargs,
-- hostnames or anything in the connection, variable or log tables.
\connect airflow
GRANT USAGE ON SCHEMA public TO grafana_ro;
GRANT SELECT (dag_id, is_paused, is_stale, has_import_errors, timetable_summary, last_parsed_time)
    ON dag TO grafana_ro;
GRANT SELECT (dag_id, run_id, run_type, state, queued_at, logical_date, start_date, end_date)
    ON dag_run TO grafana_ro;
GRANT SELECT (dag_id, task_id, run_id, map_index, state, try_number, start_date, end_date, duration)
    ON task_instance TO grafana_ro;
GRANT SELECT (id, asset_id, source_dag_id, source_task_id, source_run_id, timestamp)
    ON asset_event TO grafana_ro;

-- MLflow's database: which model versions exist, which is champion, and what the gate decided.
-- Not the artifact locations, run sources or user names.
\connect mlflow
GRANT USAGE ON SCHEMA public TO grafana_ro;
GRANT SELECT (name, creation_time, last_updated_time) ON registered_models TO grafana_ro;
GRANT SELECT (name, version, creation_time, last_updated_time, current_stage, status)
    ON model_versions TO grafana_ro;
GRANT SELECT (name, version, key, value) ON model_version_tags TO grafana_ro;
GRANT SELECT (name, alias, version) ON registered_model_aliases TO grafana_ro;
