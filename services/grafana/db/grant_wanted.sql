-- Included by readonly_user.sql (\ir) once per database, after it has listed what it wants in the
-- temporary table `wanted` (tbl, cols). This is the only place a privilege is ever granted, and it
-- can only grant two things: USAGE on the schema of a listed table, and SELECT on a listed table
-- (or on its listed columns).
--
-- A listed table that does not exist yet is skipped and reported, not an error: on a new machine
-- the Airflow and MLflow tables appear only once those services have run, and the warehouse's
-- monitoring tables only once the pipeline has. Run the grafana-db-init job again after they exist.
-- A listed COLUMN that does not exist is an error, so a renamed column is noticed.

SELECT DISTINCT format(
    'GRANT USAGE ON SCHEMA %I TO grafana_ro',
    CASE WHEN position('.' IN tbl) > 0 THEN split_part(tbl, '.', 1) ELSE 'public' END
)
FROM wanted WHERE to_regclass(tbl) IS NOT NULL \gexec

SELECT format(
    'GRANT SELECT %s ON %s TO grafana_ro',
    CASE WHEN cols IS NULL THEN '' ELSE '(' || cols || ')' END,
    to_regclass(tbl)
)
FROM wanted WHERE to_regclass(tbl) IS NOT NULL \gexec

SELECT format('skipped %s: it does not exist yet. Run this job again once it does.', tbl) AS note
FROM wanted WHERE to_regclass(tbl) IS NULL;
