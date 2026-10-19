"""The feature contract between the data warehouse and the models.

Training and (later) the API both build model inputs with prepare_features(), so a
model always sees columns computed the same way.
"""

import numpy as np
import pandas as pd
import psycopg

from src import database

# Columns of marts.customer_snapshots used as model inputs (all known at the snapshot date).
# History is measured over a fixed window (365 days) so every snapshot is described the same
# way, however much data exists before it; history_days_available says how much of the window
# is actually covered. season_sin / season_cos encode the time of year.
BASE_FEATURES = [
    "recency_days",
    "tenure_days_capped",
    "history_days_available",
    "frequency_365d",
    "monetary_365d",
    "avg_order_value_365d",
    "total_units_365d",
    "distinct_products_365d",
    "avg_days_between_orders_365d",
    "orders_last_90d",
    "orders_prev_90d",
    "spend_last_90d",
    "return_invoices_365d",
    "return_value_365d",
    "return_rate_365d",
    "is_uk",
    "season_sin",
    "season_cos",
]
DERIVED_FEATURES = ["recency_ratio", "orders_trend"]
FEATURE_COLUMNS = BASE_FEATURES + DERIVED_FEATURES

# Lifetime columns that are not model inputs. Customer segmentation (RFM) reads frequency and
# monetary; the BG/NBD baseline reads frequency, recency_days and tenure_days.
LIFETIME_COLUMNS = ["frequency", "monetary", "tenure_days"]

# Customers a retention campaign would target: bought within this many days of the snapshot.
# marts.customer_snapshots.is_actionable uses the same threshold (a test checks they agree).
ACTIONABLE_RECENCY_DAYS = 180

# Length of the label window in days. marts.snapshot_dates.horizon_days holds the same value
# (a test checks they agree); the backtest uses it to drop training snapshots whose label window
# overlaps the period being evaluated.
HORIZON_DAYS = 90

# Targets. They describe the future of a snapshot and must never be used as inputs.
CHURN_TARGET = "label_churned"
VALUE_TARGET = "label_future_spend"

_NUMERIC_SELECT = ", ".join(
    f"{column}::float8 AS {column}"
    for column in [*BASE_FEATURES, *LIFETIME_COLUMNS]
    if column != "is_uk"
)

_SELECT = f"""
    SELECT snapshot_date, split, customer_id,
           {_NUMERIC_SELECT},
           is_uk::int::float8 AS is_uk,
           is_actionable,
           label_churned::int::float8 AS {CHURN_TARGET},
           label_future_spend::float8 AS {VALUE_TARGET}
    FROM marts.customer_snapshots
"""
_SNAPSHOT_QUERY = _SELECT + " ORDER BY snapshot_date, customer_id"


def load_snapshots(conn: psycopg.Connection) -> pd.DataFrame:
    """Load every row of marts.customer_snapshots with numeric columns as floats.

    The score snapshot has no labels, so its target columns are NaN.
    """
    return database.read_dataframe(conn, _SNAPSHOT_QUERY)


def load_score_snapshot(conn: psycopg.Connection) -> pd.DataFrame:
    """Every customer as of the latest snapshot (the one with no labels yet), ready to be scored."""
    return database.read_dataframe(conn, _SELECT + " WHERE split = 'score' ORDER BY customer_id")


def load_score_customer(conn: psycopg.Connection, customer_id: int) -> pd.DataFrame:
    """One customer as of the latest snapshot: one row, or none if they are not in it."""
    return database.read_dataframe(
        conn, _SELECT + " WHERE split = 'score' AND customer_id = %s", (customer_id,)
    )


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model input columns (FEATURE_COLUMNS) as floats, adding the derived ones.

    recency_ratio: how overdue the customer is relative to their own buying rhythm
        (days since last order / average days between orders over the last 365 days);
        NaN when the customer has fewer than two orders in that window.
    orders_trend: orders in the last 90 days minus orders in the 90 days before that.
    """
    missing = set(BASE_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    out = df[BASE_FEATURES].astype("float64").copy()
    out["recency_ratio"] = (out["recency_days"] / out["avg_days_between_orders_365d"]).replace(
        [np.inf, -np.inf], np.nan
    )
    out["orders_trend"] = out["orders_last_90d"] - out["orders_prev_90d"]
    return out[FEATURE_COLUMNS]
