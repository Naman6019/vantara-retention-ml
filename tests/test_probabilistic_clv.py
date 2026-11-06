import numpy as np
import pandas as pd

from src.models.probabilistic_clv import (
    build_rfm_summary,
    deserialize_models,
    fit_models,
    predict,
    serialize_models,
)


def _synthetic_transactions(seed: int = 0) -> pd.DataFrame:
    """Simulate repeat-purchase behaviour so BG/NBD can recover the structure."""
    rng = np.random.RandomState(seed)
    cutoff = pd.Timestamp("2011-09-01")
    rows = []
    for cid in range(1, 121):
        first = cutoff - pd.Timedelta(days=int(rng.uniform(200, 700)))
        n_orders = rng.poisson(4) + 1
        date = first
        for k in range(n_orders):
            date = date + pd.Timedelta(days=int(rng.uniform(20, 90)))
            if date >= cutoff:
                break
            rows.append({
                "customer_id": float(cid),
                "invoice": f"{cid}-{k}",
                "stock_code": "SKU",
                "quantity": 1,
                "price": float(rng.uniform(5, 40)),
                "invoice_date": date,
                "line_total": float(rng.uniform(5, 40)),
                "is_cancellation": False,
                "country": "United Kingdom",
            })
    return pd.DataFrame(rows)


def test_build_rfm_summary_shapes():
    df = _synthetic_transactions()
    summary = build_rfm_summary(df, pd.Timestamp("2011-09-01"))
    assert {"frequency", "recency", "T", "monetary_value"}.issubset(summary.columns)
    assert (summary["T"] > 0).all()
    # recency must be strictly below T for a valid likelihood
    assert (summary["recency"] < summary["T"]).all()
    assert (summary["frequency"] >= 0).all()


def test_fit_and_predict_returns_value_and_palive():
    df = _synthetic_transactions()
    summary = build_rfm_summary(df, pd.Timestamp("2011-09-01"))
    models = fit_models(summary)
    preds = predict(models, summary, horizon_days=90)

    assert len(preds) == len(summary)
    assert (preds["probabilistic_value"] >= 0).all()
    assert (preds["p_alive"] >= 0).all() and (preds["p_alive"] <= 1).all()
    assert np.isfinite(preds["expected_purchases_90d"]).all()
    # Repeat buyers should be predicted to buy more than hard zeros on average.
    assert preds["expected_purchases_90d"].mean() > 0


def test_serialize_deserialize_roundtrip_preserves_predictions():
    df = _synthetic_transactions()
    summary = build_rfm_summary(df, pd.Timestamp("2011-09-01"))
    models = fit_models(summary)
    before = predict(models, summary, horizon_days=90)["probabilistic_value"].to_numpy()

    payload = serialize_models(models)
    rebuilt = deserialize_models(payload)
    after = predict(rebuilt, summary, horizon_days=90)["probabilistic_value"].to_numpy()

    assert np.allclose(before, after, rtol=1e-6)
