-- Runs once, only when the Postgres data volume is first created
-- (mounted into /docker-entrypoint-initdb.d). The main warehouse database
-- is created by POSTGRES_DB; these are reserved for later MLOps tools so
-- their metadata stays separate from the retail data.
CREATE DATABASE mlflow;
CREATE DATABASE airflow;
