"""Start the MLflow tracking server with Postgres as its metadata store.

The database URL is built here (not in docker-compose.yml) so that a password
containing characters such as @ : / is URL-encoded correctly.
"""

import os
from urllib.parse import quote

user = quote(os.environ["POSTGRES_USER"], safe="")
password = quote(os.environ["POSTGRES_PASSWORD"], safe="")
host = os.environ.get("POSTGRES_HOST", "postgres")
port = os.environ.get("POSTGRES_PORT", "5432")
database = os.environ.get("MLFLOW_DB", "mlflow")

# Name the driver explicitly: newer SQLAlchemy versions changed the default for "postgresql://"
backend_store_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

os.execvp(
    "mlflow",
    [
        "mlflow",
        "server",
        "--host",
        "0.0.0.0",
        "--port",
        "5000",
        "--backend-store-uri",
        backend_store_uri,
        # Clients upload artifacts through the server over HTTP, so no shared volume is needed
        "--serve-artifacts",
        "--artifacts-destination",
        "/mlartifacts",
        # DNS-rebinding protection stays on; allow the app container (mlflow) and the browser
        # (localhost)
        "--allowed-hosts",
        "mlflow,mlflow:*,localhost,localhost:*,127.0.0.1,127.0.0.1:*",
    ],
)
