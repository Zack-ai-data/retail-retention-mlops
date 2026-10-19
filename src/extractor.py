import zipfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException

from src.config import COLUMN_MAP, RAW_COLUMNS
from src.errors import DataFileError

# Excel row number of the first data row (row 1 is the header)
FIRST_DATA_ROW = 2


def normalise(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Rename Excel columns to database names and add lineage columns."""
    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(f"Sheet {sheet_name!r} is missing columns: {sorted(missing)}")

    out = df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())].copy()
    # Mixed int/str cells in the Excel file; keep them all as text
    out["invoice"] = out["invoice"].astype(str)
    out["stock_code"] = out["stock_code"].astype(str)
    # Read as float because of missing values; nullable integer keeps NULLs
    out["customer_id"] = out["customer_id"].astype("Int64")
    out.insert(0, "source_sheet", sheet_name)
    out.insert(1, "source_row", range(FIRST_DATA_ROW, FIRST_DATA_ROW + len(out)))
    return out[RAW_COLUMNS]


def extract(path: Path) -> Iterator[pd.DataFrame]:
    """Yield one normalised DataFrame per sheet (one at a time to limit memory).

    Raises DataFileError when the file is missing, is not a readable Excel workbook, or a sheet
    lacks the expected columns.
    """
    if not path.is_file():
        raise DataFileError(
            f"Source file not found: {path}. Download online_retail_II.xlsx from the UCI "
            "repository (see README) and place it in the data/ folder."
        )
    try:
        workbook = pd.ExcelFile(path)
    except (zipfile.BadZipFile, InvalidFileException, ValueError, OSError) as exc:
        raise DataFileError(f"{path} is not a readable Excel workbook: {exc}") from exc

    with workbook:
        for sheet_name in workbook.sheet_names:
            try:
                yield normalise(workbook.parse(sheet_name), str(sheet_name))
            except ValueError as exc:
                raise DataFileError(f"{path}: {exc}") from exc
