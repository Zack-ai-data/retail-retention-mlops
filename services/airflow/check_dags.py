"""Load every DAG the way Airflow does and fail if any cannot be imported.

A DAG with a mistake (a typo in a stage name, a bad import) is silently left out of Airflow and
only shows up as an import error in its interface. This script makes that a failing check. Run it
in the Airflow image with the DAGs folder mounted (nothing runs it automatically yet):

    docker run --rm -v "$PWD/services/airflow/dags:/opt/airflow/dags:ro" \\
        -v "$PWD/services/airflow/check_dags.py:/check_dags.py:ro" retail-airflow:ci \\
        python /check_dags.py
"""

import os
import sys

DAGS_FOLDER = "/opt/airflow/dags"

# Airflow's bundled example DAGs are controlled by configuration; leave them out of this check
os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "false"
# Airflow puts the DAGs folder on the import path so DAGs can share a helper module
# (pipeline_common.py); a standalone check has to do the same
sys.path.insert(0, DAGS_FOLDER)

from airflow.dag_processing.dagbag import DagBag  # noqa: E402  (after the setting above)

EXPECTED = {"pipeline_healthcheck", "retail_etl"}

bag = DagBag(DAGS_FOLDER)

problems = [f"{path}: {error}" for path, error in bag.import_errors.items()]
missing = EXPECTED - set(bag.dag_ids)
unexpected = set(bag.dag_ids) - EXPECTED
if missing:
    problems.append(f"DAGs that should exist were not loaded: {sorted(missing)}")
if unexpected:
    problems.append(
        f"DAGs not listed in check_dags.py (add them to EXPECTED): {sorted(unexpected)}"
    )

for dag_id in sorted(bag.dag_ids):
    dag = bag.dags[dag_id]
    print(f"{dag_id}: {len(dag.tasks)} tasks ({', '.join(t.task_id for t in dag.tasks)})")

if problems:
    print("\nProblems:")
    for problem in problems:
        print(" -", problem)
    sys.exit(1)
print("All DAGs loaded without errors.")
