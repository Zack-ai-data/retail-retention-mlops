from decimal import Decimal

import pandas as pd

from src import config, database
from src.extractor import normalise


def test_run_sql_dir_is_idempotent(scratch_conn):
    # scratch_conn already applied the schema once; applying it again must not fail
    database.run_sql_dir(scratch_conn, config.SCHEMA_DIR)

    schemas = {
        row[0]
        for row in scratch_conn.execute(
            "SELECT schema_name FROM information_schema.schemata"
        ).fetchall()
    }
    assert {"raw", "staging", "marts", "ml"} <= schemas


def test_copy_dataframe_round_trip(scratch_conn):
    sheet = pd.DataFrame(
        {
            "Invoice": ["489434", "C489435"],
            "StockCode": ["85048", "79323P"],
            "Description": ['BALL, "LIGHTS"', None],
            "Quantity": [12, -1],
            "InvoiceDate": pd.to_datetime(["2009-12-01 07:45", "2009-12-01 09:00"]),
            "Price": [6.95, 2.5],
            "Customer ID": [13085.0, None],
            "Country": ["United Kingdom", "France"],
        }
    )
    df = normalise(sheet, "Year 2009-2010")

    loaded = database.copy_dataframe(scratch_conn, df, config.RAW_TABLE)
    scratch_conn.commit()

    assert loaded == 2
    rows = scratch_conn.execute(
        "SELECT invoice, description, quantity, price, customer_id, source_row "
        "FROM raw.online_retail ORDER BY source_row"
    ).fetchall()
    assert rows[0] == ("489434", 'BALL, "LIGHTS"', 12, Decimal("6.95"), 13085, 2)
    assert rows[1][1] is None  # missing description stays NULL
    assert rows[1][4] is None  # missing customer id stays NULL
