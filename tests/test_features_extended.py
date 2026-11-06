import numpy as np
import pandas as pd
import pytest

from src.features.builder import (
    build_customer_feature_table,
    compute_behavioral_dynamics,
    compute_risk_and_preference_features,
)


@pytest.fixture
def transactions():
    cutoff = pd.Timestamp("2011-09-01")
    rows = []

    def add(cid, invoice, date, qty, price, country, cancellation=False):
        rows.append(
            {
                "invoice": invoice,
                "stock_code": "SKU-" + str(qty),
                "description": "ITEM",
                "quantity": qty,
                "price": price,
                "customer_id": float(cid),
                "invoice_date": pd.Timestamp(date),
                "line_total": qty * price,
                "is_cancellation": cancellation,
                "country": country,
            }
        )

    # Customer 1: regular, retained, some returns, Q4 spend
    add(1, "1001", "2011-06-01", 4, 10.0, "United Kingdom")
    add(1, "1002", "2011-07-01", 2, 25.0, "United Kingdom")
    add(1, "1003", "2011-08-01", -1, 10.0, "United Kingdom")
    add(1, "1004", "2011-08-20", 5, 3.0, "United Kingdom")
    add(1, "1005", "2011-10-05", 6, 10.0, "United Kingdom")

    # Customer 2: churned, active earlier (still inside the 365-day lookback)
    add(2, "2001", "2011-01-10", 20, 20.0, "Germany")
    add(2, "2002", "2011-03-15", 10, 15.0, "Germany")

    # Customer 3: loyal UK buyer with cancellations and heavy Q4
    add(3, "3000", "2010-11-05", 5, 12.0, "United Kingdom")
    add(3, "3001", "2011-05-01", 8, 12.0, "United Kingdom")
    add(3, "3002", "2011-07-15", 6, 12.0, "United Kingdom")
    add(3, "3003", "2011-08-10", 6, 12.0, "United Kingdom")
    add(3, "3004", "2011-11-20", 6, 12.0, "United Kingdom")
    add(3, "C900", "2011-08-25", -3, 12.0, "United Kingdom", cancellation=True)

    # Customer 4: joins after cutoff (should be excluded from cohort)
    add(4, "4001", "2011-10-01", 1, 50.0, "France")

    return pd.DataFrame(rows), cutoff


def test_compute_behavioral_dynamics(transactions):
    df, cutoff = transactions
    df_obs = df[df["invoice_date"] < cutoff].copy()
    dyn = compute_behavioral_dynamics(df_obs, cutoff)

    assert "interpurchase_interval_mean" in dyn.columns
    assert "interpurchase_interval_std" in dyn.columns
    assert "purchase_frequency_trend" in dyn.columns

    # Return line (qty < 0) is excluded, so gaps are 30 and 50 days
    c1 = dyn[dyn["customer_id"] == 1.0].iloc[0]
    assert c1["interpurchase_interval_mean"] == pytest.approx(
        np.mean([30.0, 50.0]), abs=0.5
    )

    # Trend is bounded between -1 and 1
    assert (dyn["purchase_frequency_trend"].between(-1.0, 1.0)).all()


def test_compute_risk_and_preference_features(transactions):
    df, cutoff = transactions
    df_obs = df[df["invoice_date"] < cutoff].copy()
    risk = compute_risk_and_preference_features(df_obs)

    c1 = risk[risk["customer_id"] == 1.0].iloc[0]
    # 1 unit returned, 11 units bought -> 1 / (11 + 1) = 0.0833
    assert c1["return_rate"] == pytest.approx(1 / 12, abs=1e-3)
    assert c1["is_uk_customer"] == 1
    assert 0.0 <= c1["discount_sensitivity"] <= 1.0
    assert 0.0 <= c1["seasonal_concentration"] <= 1.0

    c3 = risk[risk["customer_id"] == 3.0].iloc[0]
    assert c3["cancellation_rate"] > 0
    assert c3["seasonal_concentration"] > 0

    c2 = risk[risk["customer_id"] == 2.0].iloc[0]
    assert c2["is_uk_customer"] == 0


def test_build_customer_feature_table_no_save(transactions):
    df, cutoff = transactions
    features = build_customer_feature_table(
        df, cutoff_date_str="2011-09-01", save_processed=False
    )

    # Customer 4 joins after cutoff -> excluded
    assert 4.0 not in set(features["customer_id"])
    assert {"is_churned", "future_clv", "engagement_score"}.issubset(features.columns)
    assert features["is_churned"].isin([0, 1]).all()

    # Customer 1 purchased after cutoff -> retained
    c1 = features[features["customer_id"] == 1.0].iloc[0]
    assert c1["is_churned"] == 0
    assert c1["future_clv"] == 60.0

    # Customer 2 did not purchase after cutoff -> churned
    c2 = features[features["customer_id"] == 2.0].iloc[0]
    assert c2["is_churned"] == 1
    assert c2["future_clv"] == 0.0


def test_build_customer_feature_table_saves(transactions, monkeypatch, tmp_path):
    df, _ = transactions
    from src.features import builder

    monkeypatch.setattr(builder, "get_project_root", lambda: tmp_path)
    out = build_customer_feature_table(
        df, cutoff_date_str="2011-09-01", save_processed=True
    )
    saved = tmp_path / "data/processed/customer_features.parquet"
    assert saved.exists()
    assert len(pd.read_parquet(saved)) == len(out)
