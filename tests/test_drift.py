import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.monitoring.drift import compute_drift, persist_drift, psi, psi_from_edges
from src.serving.models import Base, DriftReport


def test_psi_identical_distributions_is_zero():
    ref = np.array([10, 20, 30, 40])
    assert psi(ref, ref) == 0.0


def test_psi_shifted_distribution_exceeds_threshold():
    ref = np.array([50, 50, 0, 0])
    cur = np.array([0, 0, 50, 50])
    assert psi(ref, cur) > 0.2


def test_psi_from_edges_shift():
    rng = np.random.RandomState(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(2, 1, 5000)  # shifted by 2 sigma
    edges = np.quantile(ref, np.linspace(0, 1, 11))
    assert psi_from_edges(ref, cur, edges) > 0.2
    assert psi_from_edges(ref, ref, edges) < 1e-6


def test_compute_drift_flags_shifted_feature():
    rng = np.random.RandomState(1)
    ref_values = rng.normal(0, 1, 2000)
    edges = np.quantile(ref_values, np.linspace(0, 1, 11)).tolist()
    reference = {"recency": {"edges": edges}}

    stable = pd.DataFrame({"recency": rng.normal(0, 1, 500)})
    shifted = pd.DataFrame({"recency": rng.normal(3, 1, 500)})

    rows_stable = compute_drift(stable, reference, threshold=0.2)
    rows_shift = compute_drift(shifted, reference, threshold=0.2)
    assert rows_stable[0]["drifted"] == 0
    assert rows_shift[0]["drifted"] == 1
    assert rows_shift[0]["psi"] > rows_stable[0]["psi"]


def test_persist_drift_writes_rows(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'drift.db'}", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    session = Session()

    rows = [{"feature": "recency", "psi": 0.31, "drifted": 1}]
    written = persist_drift(rows, batch_date="2011-12-09", session=session)
    session.commit()

    assert written == 1
    assert session.query(DriftReport).count() == 1
    session.close()
    engine.dispose()
