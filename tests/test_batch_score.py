import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.serving import db
from src.serving.batch_score import score_customers
from src.serving.models import Base, Customer, Prediction, Recommendation, Segment


@pytest.fixture
def sqlite_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def test_score_customers_shapes_and_value_at_risk():
    from src.models.data_split import FEATURE_COLS

    rng = np.random.RandomState(0)
    n = 20
    df = pd.DataFrame({c: rng.uniform(1, 100, n) for c in FEATURE_COLS})
    df["customer_id"] = np.arange(1, n + 1, dtype=float)
    df["spend_last_90d"] = rng.uniform(0, 500, n)
    df["return_rate"] = rng.uniform(0, 0.3, n)
    df["engagement_score"] = rng.uniform(0, 100, n)
    df["is_uk_customer"] = rng.randint(0, 2, n)

    results = score_customers(df)

    assert len(results) == n
    # value_at_risk == calibrated_probability * predicted_value
    expected = (results["calibrated_probability"] * results["predicted_value"]).round(2)
    assert np.allclose(results["value_at_risk"], expected, atol=0.02)
    assert results["action"].notna().all()
    assert results["risk_tier"].isin(["Low", "Moderate", "High"]).all()


def test_persist_results_writes_rows(sqlite_session):
    from src.models.data_split import FEATURE_COLS
    from src.serving.batch_score import persist_results

    rng = np.random.RandomState(1)
    n = 5
    df = pd.DataFrame({c: rng.uniform(1, 50, n) for c in FEATURE_COLS})
    df["customer_id"] = np.arange(100, 100 + n, dtype=float)
    df["spend_last_90d"] = rng.uniform(0, 200, n)
    df["return_rate"] = rng.uniform(0, 0.2, n)
    df["engagement_score"] = rng.uniform(0, 100, n)
    df["is_uk_customer"] = 1

    results = score_customers(df)
    written = persist_results(results, sqlite_session)
    sqlite_session.commit()

    assert written == n
    assert sqlite_session.query(Prediction).count() == n
    assert sqlite_session.query(Segment).count() == n
    assert sqlite_session.query(Recommendation).count() == n
    assert sqlite_session.query(Customer).count() == n


def test_db_ensure_engine_falls_back_to_sqlite(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db.reset_engine()
    engine = db.ensure_engine()
    assert engine is not None
    assert str(engine.url).startswith("sqlite")
    db.reset_engine()


def test_db_init_engine_with_url(tmp_path):
    db.reset_engine()
    engine = db.init_engine(f"sqlite:///{tmp_path / 'x.db'}")
    assert engine is not None
    with db.get_session() as session:
        session.add(Customer(customer_id=1.0))
    assert engine is not None
    db.reset_engine()
