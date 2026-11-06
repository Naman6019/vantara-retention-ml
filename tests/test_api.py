import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _payload(**overrides):
    payload = {
        "customer_id": 12345.0,
        "frequency": 8,
        "total_items_bought": 250,
        "unique_skus_bought": 45,
        "monetary_total": 1250.50,
        "recency": 14.5,
        "customer_age_days": 320.0,
        "monetary_avg": 156.31,
        "avg_basket_size": 31.25,
        "interpurchase_interval_mean": 28.5,
        "interpurchase_interval_std": 12.1,
        "purchase_frequency_trend": 0.25,
        "return_rate": 0.02,
        "cancellation_rate": 0.0,
        "seasonal_concentration": 0.35,
        "discount_sensitivity": 0.15,
        "is_uk_customer": 1,
        "engagement_score": 82.5,
        "spend_last_90d": 400.0,
    }
    payload.update(overrides)
    return payload


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metadata_endpoint(client):
    r = client.get("/metadata")
    assert r.status_code == 200
    data = r.json()
    assert "churn_model" in data
    assert 0.0 <= data["test_roc_auc"] <= 1.0
    assert 0.0 <= data["decision_threshold"] <= 1.0
    assert data["version"]


def test_predict_single_customer_valid(client):
    r = client.post("/predict", json=_payload())
    assert r.status_code == 200, r.text
    data = r.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["predicted_future_clv"] >= 0
    assert data["value_at_risk"] == pytest.approx(
        data["churn_probability"] * data["predicted_future_clv"], abs=0.05
    )
    assert data["risk_tier"] in {"Low", "Moderate", "High"}
    assert data["recommended_action"]
    assert data["recommendation_reasons"]
    assert len(data["top_risk_drivers"]) > 0


def test_predict_validation_error(client):
    r = client.post("/predict", json={"frequency": -5, "monetary_total": 100.0})
    assert r.status_code == 422


def test_at_risk_sorted_descending(client):
    r = client.get("/customers/at-risk?limit=10")
    assert r.status_code == 200
    rows = r.json()
    vals = [row["value_at_risk"] for row in rows]
    assert vals == sorted(vals, reverse=True)


def test_customer_360_unknown_returns_404(client):
    r = client.get("/customers/999999999999")
    assert r.status_code == 404


def test_segments_endpoint(client):
    r = client.get("/segments")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_batch_score_missing_columns_returns_400(client):
    bad_csv = "recency,frequency\n1,2\n"
    r = client.post("/batch-score", files={"file": ("bad.csv", bad_csv, "text/csv")})
    assert r.status_code == 400
    assert "missing required feature columns" in r.json()["detail"].lower()


def test_batch_score_valid_csv(client):
    from src.models.model_card import read_model_card

    cols = read_model_card()["feature_cols"]
    df = pd.DataFrame(np.random.RandomState(0).uniform(1, 100, (3, len(cols))), columns=cols)
    df["customer_id"] = [1, 2, 3]
    csv = df.to_csv(index=False)
    r = client.post("/batch-score", files={"file": ("ok.csv", csv, "text/csv")})
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_metrics_endpoint(client):
    client.get("/health")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text


def test_customer_latency_smoke(client):
    import time

    # Warm up.
    client.get("/health")
    t0 = time.perf_counter()
    for _ in range(30):
        client.get("/customers/12347")
    p95 = (time.perf_counter() - t0) / 30 * 1000
    assert p95 < 400
