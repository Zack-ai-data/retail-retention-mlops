# Settings and constants for the retail retention MLOps project
# (could name this file settings.py or constants.py)
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source data. Overridden by DATA_PATH inside Docker (mounted at /data).
DATA_PATH = Path(os.getenv("DATA_PATH", PROJECT_ROOT / "data" / "online_retail_II.xlsx"))
SCHEMA_DIR = PROJECT_ROOT / "sql" / "schema"
TRANSFORM_DIR = PROJECT_ROOT / "sql" / "transform"

# PostgreSQL connection. Host defaults to localhost for running outside Docker;
# docker-compose sets POSTGRES_HOST=postgres for the app container.
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "")
POSTGRES_CONNECT_TIMEOUT = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5"))  # seconds per attempt

# MLflow tracking server (http://mlflow:5000 inside Docker, http://localhost:5000 from the host)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = "retail-retention"
RANDOM_STATE = 42

# Registered model names (MLflow model registry)
CHURN_MODEL_NAME = "retail-churn-classifier"
VALUE_MODEL_NAME = "retail-value-regressor"
SEGMENT_MODEL_NAME = "retail-rfm-segments"

# Models are stored with cloudpickle: only load models from this project's own registry.
SERIALIZATION = "cloudpickle"

# Business assumptions for ranking customers by margin at risk (docs/problem_statement.md).
# There is no data on either figure; they are stated assumptions, and the write-up shows how the
# outreach list changes as they vary.
MARGIN = 0.30  # share of a customer's spend that is profit
CONTACT_COST_GBP = 5.0  # cost of contacting one customer

# Raw layer
RAW_TABLE = "raw.online_retail"

# Excel column name -> database column name
COLUMN_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}

# Order of the columns written to RAW_TABLE (loaded_at is filled by the database)
RAW_COLUMNS = ["source_sheet", "source_row", *COLUMN_MAP.values()]
