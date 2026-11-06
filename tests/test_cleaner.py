import pandas as pd

from src.data.cleaner import (
    clean_missing_customers,
    clean_outliers_and_prices,
    deduplicate_transactions,
    filter_admin_codes,
    flag_returns_and_cancellations,
)


def test_clean_missing_customers():
    df = pd.DataFrame({
        "customer_id": [12345.0, None, 12346.0],
        "quantity": [1, 2, 3],
    })
    # Dropping
    df_dropped = clean_missing_customers(df, drop=True)
    assert len(df_dropped) == 2
    assert df_dropped["customer_id"].isna().sum() == 0

    # Flagging
    df_flagged = clean_missing_customers(df, drop=False)
    assert len(df_flagged) == 3
    assert "is_guest_customer" in df_flagged.columns
    assert df_flagged["is_guest_customer"].sum() == 1


def test_deduplicate_transactions():
    df = pd.DataFrame({
        "invoice": ["A", "A", "B"],
        "stock_code": ["1", "1", "2"],
        "quantity": [10, 10, 5],
    })
    df_dedup = deduplicate_transactions(df)
    assert len(df_dedup) == 2


def test_filter_admin_codes():
    df = pd.DataFrame({
        "stock_code": ["POST", "85123A", "D", "M", "22423"],
        "price": [10.0, 2.5, 5.0, 1.0, 3.0],
    })
    df_filtered = filter_admin_codes(df, excluded_codes=["POST", "D", "M"])
    assert len(df_filtered) == 2
    assert set(df_filtered["stock_code"]) == {"85123A", "22423"}


def test_flag_returns_and_cancellations():
    df = pd.DataFrame({
        "invoice": ["C12345", "54321", "54322"],
        "quantity": [-5, 10, -2],
    })
    df_flagged = flag_returns_and_cancellations(df)
    assert "is_cancellation" in df_flagged.columns
    assert df_flagged["is_cancellation"].tolist() == [True, False, True]


def test_clean_outliers_and_prices():
    df = pd.DataFrame({
        "quantity": [5, 20000, 10],
        "price": [2.5, 1.0, -0.5],
    })
    df_clean = clean_outliers_and_prices(df, qty_max=10000, price_min=0.001)
    assert len(df_clean) == 1
    assert df_clean.iloc[0]["quantity"] == 5
