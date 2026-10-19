import pandas as pd
import pytest

from src.config import COLUMN_MAP, RAW_COLUMNS
from src.extractor import normalise


def make_sheet() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Invoice": ["489434", 489435, "C489436"],
            "StockCode": [85048, "79323P", "POST"],
            "Description": ["BALL", None, "POSTAGE"],
            "Quantity": [12, 6, -1],
            "InvoiceDate": pd.to_datetime(
                ["2009-12-01 07:45", "2009-12-01 07:46", "2009-12-01 09:00"]
            ),
            "Price": [6.95, 2.5, 18.0],
            "Customer ID": [13085.0, None, 12345.0],
            "Country": ["United Kingdom"] * 3,
        }
    )


def test_normalise_columns_and_order():
    out = normalise(make_sheet(), "Year 2009-2010")
    assert list(out.columns) == RAW_COLUMNS


def test_normalise_lineage_columns():
    out = normalise(make_sheet(), "Year 2009-2010")
    assert (out["source_sheet"] == "Year 2009-2010").all()
    assert out["source_row"].tolist() == [2, 3, 4]  # Excel row numbers, header is row 1


def test_normalise_types():
    out = normalise(make_sheet(), "s")
    assert out["invoice"].tolist() == ["489434", "489435", "C489436"]
    assert out["stock_code"].tolist() == ["85048", "79323P", "POST"]
    assert str(out["customer_id"].dtype) == "Int64"
    assert pd.isna(out["customer_id"].iloc[1])
    assert pd.isna(out["description"].iloc[1])


def test_normalise_does_not_modify_input():
    df = make_sheet()
    normalise(df, "s")
    assert list(df.columns) == list(COLUMN_MAP)


def test_normalise_rejects_missing_columns():
    with pytest.raises(ValueError, match="Country"):
        normalise(make_sheet().drop(columns=["Country"]), "s")
