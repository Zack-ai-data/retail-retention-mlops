"""Pieces shared by the pipeline DAGs.

Every task runs one stage of the pipeline (`python -m src.main --stage <name>`) in the pipeline's
own Python environment, which holds the project's locked library versions (see
services/airflow/Dockerfile). The stages are the same ones you can run by hand with
`docker compose run --rm app python -m src.main --stage <name>`.
"""

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Asset

# The pipeline runs in its own environment, not in Airflow's
PIPELINE = "cd $PIPELINE_HOME && $PIPELINE_PYTHON"

STAGES = ("load", "transform", "train", "score", "plan")

# The feature table the models are trained from. The ETL DAG reports that it has refreshed it, and
# later DAGs can start when it has, instead of waiting for a clock.
CUSTOMER_SNAPSHOTS = Asset(
    name="customer_snapshots",
    uri="postgres://postgres:5432/retail_dw/marts/customer_snapshots",
)


def stage(task_id: str, name: str, **kwargs) -> BashOperator:
    """A task that runs one pipeline stage. Extra keyword arguments go to the task (retries,
    timeouts, outlets). An unknown stage name is an error when the DAG is loaded, not when it runs.
    """
    if name not in STAGES:
        raise ValueError(f"Unknown pipeline stage {name!r}; the stages are {', '.join(STAGES)}")
    return BashOperator(
        task_id=task_id, bash_command=f"{PIPELINE} -m src.main --stage {name}", **kwargs
    )
