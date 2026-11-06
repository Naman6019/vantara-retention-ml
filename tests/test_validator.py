import numpy as np
import pandas as pd
import pytest

from src.data.validator import REQUIRED_COLUMNS, validate_raw_data


def _valid_frame(n_rows: int = 600_000, n_customers: int = 5_000, cust_null_pct: float = 0.10):
    base = {
        "invoice": np.arange(n_rows).astype(str),
        "stock_code": ["SKU"] * n_rows,
        "description": ["ITEM"] * n_rows,
        "quantity": np.ones(n_rows),
        "invoice_date": pd.to_datetime(["2010-06-15"] * n_rows),
        "price": np.full(n_rows, 1.5),
        "customer_id": np.arange(n_rows, dtype=float) % n_customers,
        "country": ["United Kingdom"] * n_rows,
    }
    df = pd.DataFrame(base)
    n_null = int(n_rows * cust_null_pct)
    if n_null:
        df.loc[: n_null - 1, "customer_id"] = np.nan
    return df


def test_validate_raw_data_passes():
    df = _valid_frame()
    passed, report = validate_raw_data(df)
    assert passed is True
    assert report["total_records"] == 600_000
    assert report["unique_customers"] == 5_000
    assert report["checks_failed"] == []
    assert len(report["checks_passed"]) >= 5


def test_validate_raw_data_null_critical_column_raises():
    df = _valid_frame()
    df.loc[0, "price"] = np.nan
    with pytest.raises(ValueError, match="Integrity check failed: Column 'price'"):
        validate_raw_data(df)


def test_validate_raw_data_excessive_customer_nulls_raises():
    df = _valid_frame(cust_null_pct=0.50)
    with pytest.raises(ValueError, match="Customer ID missing rate excessive"):
        validate_raw_data(df)


def test_validate_raw_data_low_unique_customers_raises():
    df = _valid_frame(n_customers=100)
    with pytest.raises(ValueError, match="Unique customer count low"):
        validate_raw_data(df)


def test_validate_raw_data_date_range_warning():
    df = _valid_frame()
    df["invoice_date"] = pd.to_datetime(["2005-01-01"] * len(df))
    passed, report = validate_raw_data(df)
    assert passed is True
    assert report["warnings"]
    assert "2005" in report["warnings"][0]


def test_required_columns_constant():
    assert REQUIRED_COLUMNS == [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "price",
        "customer_id",
        "country",
    ]
