import pandas as pd
import pytest

from src.features.builder import (
    compute_engagement_score,
    compute_rfm_and_spend_features,
    compute_targets,
    extract_temporal_windows,
)


@pytest.fixture
def mock_transactions():
    cutoff = pd.Timestamp("2011-09-01")
    data = [
        # Customer 1: active before cutoff, also active after cutoff (retained, is_churned=0)
        {"invoice": "1001", "stock_code": "A", "description": "ITEM A", "quantity": 2, "price": 10.0, "customer_id": 1.0, "invoice_date": pd.Timestamp("2011-06-01"), "line_total": 20.0, "is_cancellation": False, "country": "United Kingdom"},
        {"invoice": "1002", "stock_code": "B", "description": "ITEM B", "quantity": 1, "price": 15.0, "customer_id": 1.0, "invoice_date": pd.Timestamp("2011-08-15"), "line_total": 15.0, "is_cancellation": False, "country": "United Kingdom"},
        {"invoice": "1003", "stock_code": "C", "description": "ITEM C", "quantity": 3, "price": 10.0, "customer_id": 1.0, "invoice_date": pd.Timestamp("2011-10-01"), "line_total": 30.0, "is_cancellation": False, "country": "United Kingdom"},

        # Customer 2: active before cutoff, inactive after cutoff (churned, is_churned=1)
        {"invoice": "2001", "stock_code": "A", "description": "ITEM A", "quantity": 5, "price": 20.0, "customer_id": 2.0, "invoice_date": pd.Timestamp("2011-01-10"), "line_total": 100.0, "is_cancellation": False, "country": "Germany"},

        # Customer 3: joins strictly after cutoff (new customer, should NOT be evaluated at cutoff)
        {"invoice": "3001", "stock_code": "D", "description": "ITEM D", "quantity": 1, "price": 50.0, "customer_id": 3.0, "invoice_date": pd.Timestamp("2011-11-01"), "line_total": 50.0, "is_cancellation": False, "country": "France"},
    ]
    return pd.DataFrame(data), cutoff


def test_temporal_window_split_no_leakage(mock_transactions):
    df, cutoff = mock_transactions
    df_obs, df_pred = extract_temporal_windows(df, cutoff)

    # Verification: Observation window strictly before cutoff
    assert (df_obs["invoice_date"] < cutoff).all(), "Leakage detected: obs records on or after cutoff!"
    # Verification: Prediction window strictly on or after cutoff
    assert (df_pred["invoice_date"] >= cutoff).all(), "Leakage detected: pred records before cutoff!"


def test_leak_free_target_calculation(mock_transactions):
    df, cutoff = mock_transactions
    df_obs, df_pred = extract_temporal_windows(df, cutoff)
    # Pass the explicit cohort of active customers (both last purchased within 365 days).
    targets = compute_targets(df_obs, df_pred, cohort={1.0, 2.0})

    # Customer 3 should not be in target evaluation cohort as they weren't historical customers
    assert set(targets["customer_id"]) == {1.0, 2.0}

    # Customer 1 purchased after cutoff -> not churned
    c1 = targets[targets["customer_id"] == 1.0].iloc[0]
    assert c1["is_churned"] == 0
    assert c1["future_clv"] == 30.0

    # Customer 2 did not purchase after cutoff -> churned
    c2 = targets[targets["customer_id"] == 2.0].iloc[0]
    assert c2["is_churned"] == 1
    assert c2["future_clv"] == 0.0


def test_rfm_calculation(mock_transactions):
    df, cutoff = mock_transactions
    df_obs, _ = extract_temporal_windows(df, cutoff)
    rfm = compute_rfm_and_spend_features(df_obs, cutoff)

    # Customer 1: invoices 1001 & 1002 (frequency=2, monetary=35.0)
    c1 = rfm[rfm["customer_id"] == 1.0].iloc[0]
    assert c1["frequency"] == 2
    assert c1["monetary_total"] == 35.0
    assert c1["recency"] == pytest.approx((cutoff - pd.Timestamp("2011-08-15")).total_seconds() / 86400.0)


def test_engagement_score_bounds():
    df = pd.DataFrame({
        "recency": [10.0, 50.0, 200.0],
        "frequency": [15, 5, 1],
        "monetary_total": [5000.0, 500.0, 50.0],
    })
    scores = compute_engagement_score(df)
    assert (scores >= 0).all()
    assert (scores <= 100).all()
    # Most active customer should have highest engagement
    assert scores.iloc[0] > scores.iloc[2]


# ---------------------------------------------------------------------------
# Task 1 additions: 90-day label window, active cohort, engagement reference
# ---------------------------------------------------------------------------

def test_label_window_excludes_transaction_at_t_plus_window():
    """A purchase on T+90 is *outside* the 90-day window [T, T+90) and is ignored."""
    from src.features.builder import build_customer_feature_table

    cutoff = pd.Timestamp("2011-09-01")
    window = 90  # window end = 2011-11-30
    rows = [
        # active customer with one historical order
        {"invoice": "1", "stock_code": "A", "description": "A", "quantity": 2, "price": 10.0,
         "customer_id": 1.0, "invoice_date": pd.Timestamp("2011-08-01"), "line_total": 20.0,
         "is_cancellation": False, "country": "United Kingdom"},
        # purchase exactly at T+90 -> excluded from the label window
        {"invoice": "2", "stock_code": "A", "description": "A", "quantity": 3, "price": 10.0,
         "customer_id": 1.0, "invoice_date": cutoff + pd.Timedelta(days=window), "line_total": 30.0,
         "is_cancellation": False, "country": "United Kingdom"},
    ]
    df = pd.DataFrame(rows)
    features = build_customer_feature_table(
        df, cutoff_date_str="2011-09-01", save_processed=False, churn_window_days=window
    )
    c1 = features[features["customer_id"] == 1.0].iloc[0]
    # The retained purchase is excluded, so the customer is labelled churned with 0 value.
    assert c1["is_churned"] == 1
    assert c1["future_clv"] == 0.0


def test_inactive_customer_excluded_from_cohort():
    """A customer with no purchase in the lookback window is dropped (plan D2)."""
    from src.features.builder import build_customer_feature_table

    cutoff = pd.Timestamp("2011-09-01")
    rows = [
        # last purchase 400 days before cutoff -> outside the 365-day lookback
        {"invoice": "1", "stock_code": "A", "description": "A", "quantity": 1, "price": 10.0,
         "customer_id": 1.0, "invoice_date": cutoff - pd.Timedelta(days=400), "line_total": 10.0,
         "is_cancellation": False, "country": "United Kingdom"},
        # recent purchase -> in the cohort
        {"invoice": "2", "stock_code": "A", "description": "A", "quantity": 1, "price": 10.0,
         "customer_id": 2.0, "invoice_date": cutoff - pd.Timedelta(days=30), "line_total": 10.0,
         "is_cancellation": False, "country": "United Kingdom"},
    ]
    df = pd.DataFrame(rows)
    features = build_customer_feature_table(
        df, cutoff_date_str="2011-09-01", save_processed=False, active_lookback_days=365
    )
    assert set(features["customer_id"]) == {2.0}


def test_engagement_reference_is_reproducible():
    """Scoring one row against a saved reference equals scoring it in a batch (D10)."""
    from src.features import engagement

    batch = pd.DataFrame({
        "recency": [5.0, 60.0, 300.0],
        "frequency": [20, 6, 1],
        "monetary_total": [8000.0, 900.0, 40.0],
    })
    ref = engagement.fit_reference(batch)
    batch_scores = engagement.score(batch, ref)
    single = engagement.score(batch.iloc[[1]], ref)
    assert single.iloc[0] == pytest.approx(batch_scores.iloc[1])
