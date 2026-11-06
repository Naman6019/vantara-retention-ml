import numpy as np
import pandas as pd

from src.segmentation import clusterer
from src.segmentation.clusterer import (
    SEGMENTATION_FEATURES,
    assign_business_personas,
    evaluate_kmeans_k_range,
    run_customer_segmentation,
)


def _segmentation_frame(n: int = 200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "recency": rng.uniform(0, 300, n),
            "frequency": rng.randint(1, 20, n),
            "monetary_total": rng.uniform(50, 5000, n),
            "return_rate": rng.uniform(0, 0.4, n),
            "engagement_score": rng.uniform(0, 100, n),
            "customer_id": np.arange(n, dtype=float),
            "is_churned": rng.randint(0, 2, n),
            "future_clv": rng.uniform(0, 1000, n),
        }
    )
    return df


def test_evaluate_kmeans_k_range_returns_best_and_metrics():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(0, 0.5, (30, 5)), rng.normal(6, 0.5, (30, 5))])
    best_k, metrics = evaluate_kmeans_k_range(X, k_min=2, k_max=4, seed=1)
    assert best_k in [2, 3, 4]
    assert len(metrics) == 3
    assert set(metrics.columns) == {"k", "inertia", "silhouette", "davies_bouldin"}


def test_assign_business_personas_all_branches():
    profiles = pd.DataFrame(
        [
            {"cluster": 0, "recency": 10.0, "frequency": 2, "monetary_total": 100.0, "return_rate": 0.0, "engagement_score": 80.0},
            {"cluster": 1, "recency": 30.0, "frequency": 5, "monetary_total": 800.0, "return_rate": 0.0, "engagement_score": 55.0},
            {"cluster": 2, "recency": 220.0, "frequency": 4, "monetary_total": 1500.0, "return_rate": 0.0, "engagement_score": 40.0},
            {"cluster": 3, "recency": 30.0, "frequency": 1, "monetary_total": 200.0, "return_rate": 0.0, "engagement_score": 30.0},
            {"cluster": 4, "recency": 250.0, "frequency": 1, "monetary_total": 100.0, "return_rate": 0.0, "engagement_score": 10.0},
            {"cluster": 5, "recency": 120.0, "frequency": 5, "monetary_total": 900.0, "return_rate": 0.0, "engagement_score": 50.0},
        ]
    )
    personas = assign_business_personas(profiles)
    # Highest monetary cluster is VIP; every cluster gets a distinct label.
    assert personas[2] == "High-Value Champions (VIP)"
    assert personas[1] == "Loyal & Active Frequent Buyers"
    assert personas[0] == "Recent Onboarding / New Prospects"
    assert personas[3].startswith("Recent Onboarding / New Prospects")
    assert personas[4] == "Dormant / Low-Engagement"
    assert personas[5] == "Steady Mid-Value Buyers"
    assert len(set(personas.values())) == len(personas)


def test_assign_business_personas_high_return_branch():
    profiles = pd.DataFrame(
        [
            {"cluster": 0, "recency": 100.0, "frequency": 5, "monetary_total": 5000.0, "return_rate": 0.05, "engagement_score": 70.0},
            {"cluster": 1, "recency": 60.0, "frequency": 2, "monetary_total": 900.0, "return_rate": 0.35, "engagement_score": 45.0},
        ]
    )
    personas = assign_business_personas(profiles)
    assert personas[0] == "High-Value Champions (VIP)"
    assert personas[1] == "High-Return Service-Recovery"


def test_run_customer_segmentation_with_save(monkeypatch, tmp_path):
    monkeypatch.setattr(clusterer, "get_project_root", lambda: tmp_path)
    (tmp_path / "models_artifacts").mkdir(parents=True)
    (tmp_path / "data/processed").mkdir(parents=True)
    df = _segmentation_frame()

    result = run_customer_segmentation(df, save_artifacts=True)

    assert set(result).issuperset(
        {
            "kmeans_model",
            "gmm_model",
            "personas",
            "kmeans_silhouette",
            "segmented_df",
        }
    )
    segmented = result["segmented_df"]
    assert "kmeans_cluster" in segmented.columns
    assert "gmm_cluster" in segmented.columns
    assert "segment_persona" in segmented.columns
    assert segmented["segment_persona"].notna().all()

    art = tmp_path / "models_artifacts"
    assert (art / "kmeans_model.joblib").exists()
    assert (art / "gmm_model.joblib").exists()
    assert (art / "segment_personas.joblib").exists()
    assert (art / "segmentation_scaler.joblib").exists()
    assert (tmp_path / "data/processed/customer_segments.parquet").exists()


def test_run_customer_segmentation_loads_from_parquet(monkeypatch, tmp_path):
    monkeypatch.setattr(clusterer, "get_project_root", lambda: tmp_path)
    proc = tmp_path / "data/processed"
    proc.mkdir(parents=True)
    _segmentation_frame(120).to_parquet(proc / "customer_features.parquet")

    result = run_customer_segmentation(df=None, save_artifacts=False)
    assert len(result["segmented_df"]) == 120


def test_segmentation_features_constant():
    assert SEGMENTATION_FEATURES == [
        "recency",
        "frequency",
        "monetary_total",
        "return_rate",
        "engagement_score",
    ]
