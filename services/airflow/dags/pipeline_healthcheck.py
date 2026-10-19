"""Does the Airflow stack reach everything the pipeline needs? Run it by hand after any change.

It does no data work. It proves, through Airflow's Celery worker, that the pipeline's own Python
environment works, the pipeline's command line starts, and the database and the MLflow server are
reachable from where the tasks run. If this DAG is green, the real pipeline DAGs can run.
"""

from datetime import UTC, datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

# The pipeline runs in its own environment (see services/airflow/Dockerfile)
PIPELINE = "cd $PIPELINE_HOME && $PIPELINE_PYTHON"

with DAG(
    dag_id="pipeline_healthcheck",
    description="Checks that tasks can run the pipeline, and reach Postgres and MLflow",
    schedule=None,  # only when started by hand
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    tags=["health"],
    doc_md=__doc__,
) as dag:
    environment = BashOperator(
        task_id="pipeline_environment",
        bash_command=(
            f'{PIPELINE} -c "import sys, sklearn, pandas; print(sys.version.split()[0], '
            "'scikit-learn', sklearn.__version__, 'pandas', pandas.__version__)\""
        ),
    )

    command_line = BashOperator(
        task_id="pipeline_command_line",
        bash_command=f"{PIPELINE} -m src.main --help",
    )

    database = BashOperator(
        task_id="database_reachable",
        bash_command=(
            f'{PIPELINE} -c "from src import database; c = database.get_connection(retries=1); '
            "print('database ok:', c.execute('select 1').fetchone()[0]); c.close()\""
        ),
    )

    mlflow = BashOperator(
        task_id="mlflow_reachable",
        bash_command=(
            f"{PIPELINE} -c \"from src import tracking; tracking.configure(); print('mlflow ok')\""
        ),
    )

    environment >> command_line >> [database, mlflow]
