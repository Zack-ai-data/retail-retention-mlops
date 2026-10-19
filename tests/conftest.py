import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, database, tracking
from src.errors import PipelineError

SHEET_1 = "Year 2009-2010"
SHEET_2 = "Year 2010-2011"


def pytest_addoption(parser):
    parser.addoption(
        "--require-warehouse",
        action="store_true",
        default=False,
        help="fail (instead of skip) the regression tests that need the real loaded data",
    )


@pytest.fixture
def scratch_conn():
    """Connection to a throwaway database with the schema applied; dropped afterwards.

    Skips the test when PostgreSQL is not reachable (e.g. the container is not running).
    """
    try:
        admin = database.get_connection(dbname="postgres", retries=1, autocommit=True)
    except PipelineError as exc:
        pytest.skip(f"PostgreSQL not available: {exc}")

    name = f"test_{uuid.uuid4().hex[:8]}"
    admin.execute(f'CREATE DATABASE "{name}"')
    conn = database.get_connection(dbname=name, retries=1)
    try:
        database.run_sql_dir(conn, config.SCHEMA_DIR)
        yield conn
    finally:
        conn.close()
        admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.close()


@pytest.fixture
def insert_raw(scratch_conn):
    """Insert rows into raw.online_retail.

    Each row is (sheet, invoice, stock_code, description, quantity, 'YYYY-MM-DD HH:MM',
    price, customer_id, country). source_row is numbered automatically per sheet.
    """
    counters: dict[str, int] = {}

    def _insert(rows):
        for sheet, invoice, stock, desc, qty, when, price, customer, country in rows:
            counters[sheet] = counters.get(sheet, 1) + 1  # Excel row 1 is the header
            scratch_conn.execute(
                """
                INSERT INTO raw.online_retail
                    (source_sheet, source_row, invoice, stock_code, description, quantity,
                     invoice_date, price, customer_id, country)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    sheet,
                    counters[sheet],
                    invoice,
                    stock,
                    desc,
                    qty,
                    datetime.strptime(when, "%Y-%m-%d %H:%M"),
                    price,
                    customer,
                    country,
                ),
            )
        scratch_conn.commit()

    return _insert


EXCEL_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


def build_sample_orders(seed: int = 7) -> pd.DataFrame:
    """About 6,000 synthetic order lines shaped like the real file.

    Covers 2009-12-01 to 2011-12-09 so every snapshot and label window exists. Includes
    cancellations, postage lines and invoices without a customer id. Deterministic for a seed.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2009-12-01")
    n_days = (pd.Timestamp("2011-12-09") - start).days
    products = [
        (str(85000 + i), f"ITEM {i}", float(price))
        for i, price in enumerate(rng.uniform(0.5, 12, 40).round(2))
    ]

    rows: list[tuple] = []
    invoice_number = 489000

    def add_invoice(when, customer, lines, cancellation=False):
        nonlocal invoice_number
        invoice_number += 1
        invoice = f"C{invoice_number}" if cancellation else str(invoice_number)
        for code, description, quantity, price in lines:
            rows.append(
                (invoice, code, description, quantity, when, price, customer, "United Kingdom")
            )

    for customer in range(12000, 12400):
        first_day = int(rng.integers(0, n_days - 120))
        order_days = np.sort(rng.integers(first_day, n_days + 1, 1 + int(rng.poisson(5))))
        for day in order_days:
            when = start + pd.Timedelta(
                days=int(day), hours=int(rng.integers(8, 18)), minutes=int(rng.integers(0, 60))
            )
            known = None if rng.random() < 0.15 else float(customer)  # guest checkout
            chosen = rng.choice(len(products), int(rng.integers(1, 5)), replace=False)
            lines = [
                (products[i][0], products[i][1], int(rng.integers(1, 13)), products[i][2])
                for i in chosen
            ]
            if rng.random() < 0.1:
                lines.append(("POST", "POSTAGE", 1, 18.0))
            add_invoice(when, known, lines)
            if rng.random() < 0.04:  # a cancellation of the first product line
                code, description, quantity, price = lines[0]
                add_invoice(
                    when + pd.Timedelta(days=1),
                    known,
                    [(code, description, -quantity, price)],
                    cancellation=True,
                )
    # the real data ends on 2011-12-09; make sure the last day is present
    add_invoice(
        pd.Timestamp("2011-12-09 12:00"),
        12000.0,
        [(products[0][0], products[0][1], 2, products[0][2])],
    )
    return pd.DataFrame(rows, columns=EXCEL_COLUMNS)


@pytest.fixture
def sample_workbook(tmp_path) -> Path:
    """A synthetic Excel workbook with the real file's two-sheet layout, including the overlap.

    Rows from 2010-12-01 to 2010-12-09 appear identically in both sheets, as in the source data.
    """
    orders = build_sample_orders()
    overlap_start, overlap_end = pd.Timestamp("2010-12-01"), pd.Timestamp("2010-12-10")
    sheet_1 = orders[orders["InvoiceDate"] < overlap_end]
    sheet_2 = orders[orders["InvoiceDate"] >= overlap_start]
    path = tmp_path / "online_retail_sample.xlsx"
    with pd.ExcelWriter(path) as writer:
        sheet_1.to_excel(writer, sheet_name=SHEET_1, index=False)
        sheet_2.to_excel(writer, sheet_name=SHEET_2, index=False)
    return path


@pytest.fixture
def mlflow_client(tmp_path):
    """MLflow client on a throwaway SQLite store (no tracking server needed)."""
    return tracking.configure(
        f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_location=(tmp_path / "artifacts").as_uri()
    )
