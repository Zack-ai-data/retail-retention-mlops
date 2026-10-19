"""Print the URL of Airflow's metadata database, built from the project's POSTGRES_* variables.

Airflow reads a database URL from its configuration. Building it here, instead of writing the
password into the compose file, has two benefits: the password stays only in .env, and it is
URL-quoted, so a password containing characters such as @ / : # still works.

    python db_url.py            postgresql+psycopg2://user:password@host:5432/airflow
    python db_url.py --celery   db+postgresql+psycopg2://...   (Celery's result backend form)

Airflow runs this through its `..._CMD` settings and uses the output exactly as printed, so the
URL is written with no trailing newline (a newline would become part of the database name). It
only prints the URL; it never logs it.
"""

import os
import sys
from urllib.parse import quote

DATABASE = "airflow"


def database_url(celery: bool = False) -> str:
    user = quote(os.environ["POSTGRES_USER"], safe="")
    password = quote(os.environ["POSTGRES_PASSWORD"], safe="")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{DATABASE}"
    return f"db+{url}" if celery else url


if __name__ == "__main__":
    sys.stdout.write(database_url(celery="--celery" in sys.argv))
