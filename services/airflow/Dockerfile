# Airflow with the pipeline inside it.
#
# Two separate Python environments live in this image, on purpose:
#   * Airflow's own, with the library versions Airflow tests together (its constraints file).
#   * /opt/pipeline-venv, built from the project's hashed lock file, which holds the pipeline.
# Tasks run `/opt/pipeline-venv/bin/python -m src.main --stage ...`, so the pipeline always runs on
# exactly the library versions the tests ran with, and neither environment can break the other.
# Airflow 3.3.2 pins pandas 3.0.5 and numpy 2.5.3; the project is tested on 3.0.6 and 2.5.4.
#
# Build from the project folder (the compose file does): the context must contain src/ and sql/.

FROM apache/airflow:3.3.2-python3.12

USER root
RUN mkdir -p /opt/pipeline /opt/pipeline-venv \
    && chown -R airflow:0 /opt/pipeline /opt/pipeline-venv
USER airflow

# The pipeline's environment, installed from the same lock file as every other image
COPY --chown=airflow:0 requirements.lock /opt/pipeline/requirements.lock
RUN python -m venv /opt/pipeline-venv \
    && /opt/pipeline-venv/bin/pip install --no-cache-dir --require-hashes \
        -r /opt/pipeline/requirements.lock

# The pipeline code, as the other images carry it
COPY --chown=airflow:0 src /opt/pipeline/src
COPY --chown=airflow:0 sql /opt/pipeline/sql

# Builds Airflow's database URLs from the POSTGRES_* variables (see the file)
COPY --chown=airflow:0 services/airflow/db_url.py /opt/airflow/db_url.py

# Pass the commit so training runs started by Airflow record which code made them
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA} \
    PIPELINE_HOME=/opt/pipeline \
    PIPELINE_PYTHON=/opt/pipeline-venv/bin/python
