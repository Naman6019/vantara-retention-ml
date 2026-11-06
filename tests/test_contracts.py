import pandas as pd
import pytest

from src.data.contracts import FEATURE_SCHEMA, validate_features, validate_raw
from src.pipeline import main as pipeline_main


def _valid_features():
    return pd.DataFrame({
        "customer_id": [1.0, 2.0],
        "frequency": [2, 3],
        "recency": [10.0, 20.0],
        "monetary_total": [100.0, 200.0],
        "engagement_score": [50.0, 60.0],
        "return_rate": [0.1, 0.2],
        "cancellation_rate": [0.0, 0.0],
        "discount_sensitivity": [0.3, 0.4],
        "is_uk_customer": [1, 0],
        "is_churned": [0, 1],
        "future_clv": [0.0, 500.0],
    })


def test_validate_features_accepts_good_data():
    out = validate_features(_valid_features())
    assert len(out) == 2


def test_validate_features_rejects_bad_range(tmp_path, monkeypatch):
    from src.data import contracts

    monkeypatch.setattr(contracts, "_quarantine", lambda records, root=None: tmp_path / "q.json")
    bad = _valid_features().assign(return_rate=[2.0, 0.2])
    with pytest.raises(ValueError, match="Feature contract violated"):
        contracts.validate_features(bad)


def test_validate_raw_accepts_minimal_frame():
    df = pd.DataFrame({
        "invoice": ["1", "2"],
        "stock_code": ["A", "B"],
        "quantity": [1.0, 2.0],
        "invoice_date": pd.to_datetime(["2011-01-01", "2011-01-02"]),
        "price": [1.0, 2.0],
        "customer_id": [1.0, 2.0],
        "country": ["United Kingdom", "France"],
    })
    assert len(validate_raw(df)) == 2


def test_feature_schema_is_not_strict():
    assert FEATURE_SCHEMA.strict is False


def test_pipeline_bad_stage_exits_nonzero():
    with pytest.raises(SystemExit):
        pipeline_main(["not-a-stage"])
