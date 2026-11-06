import numpy as np
import pandas as pd
import pytest

from src.models.churn_models import evaluate_model, initialize_models
from src.models.clv_regressor import evaluate_regressor, initialize_regressors
from src.models.data_split import FEATURE_COLS, split_data
from src.segmentation.clusterer import assign_business_personas, evaluate_kmeans_k_range


@pytest.fixture
def mock_dataset():
    np.random.seed(42)
    n = 120
    data = {col: np.random.uniform(1, 100, n) for col in FEATURE_COLS}
    data["customer_id"] = np.arange(1000, 1000 + n)
    data["is_churned"] = np.random.choice([0, 1], size=n, p=[0.45, 0.55])
    data["future_clv"] = np.where(data["is_churned"] == 1, 0.0, np.random.uniform(50, 1000, n))
    data["spend_last_90d"] = np.random.uniform(0, 800, n)
    # Stacked snapshots: two training cutoffs, one validation, one test.
    data["cutoff_date"] = pd.to_datetime(
        [pd.Timestamp("2011-01-01")] * 84
        + [pd.Timestamp("2011-06-01")] * 18
        + [pd.Timestamp("2011-09-01")] * 18
    )
    return pd.DataFrame(data)


def test_split_data_shapes(mock_dataset):
    data = split_data(mock_dataset, save_scaler=False)
    assert len(data["X_train"]) == 84  # 2 monthly training cutoffs
    assert len(data["X_val"]) == 18    # validation cutoff
    assert len(data["X_test"]) == 18   # test cutoff
    assert "X_train_scaled" in data
    assert data["X_train_scaled"].shape == data["X_train"].shape


def test_churn_model_evaluation(mock_dataset):
    data = split_data(mock_dataset, save_scaler=False)
    models = initialize_models(seed=42)

    # Test Logistic Regression on scaled data
    lr_item = models["Logistic Regression"]
    metrics = evaluate_model(
        model=lr_item["model"],
        X_train=data["X_train_scaled"],
        y_train=data["y_train"],
        X_test=data["X_test_scaled"],
        y_test=data["y_test"],
        cv_folds=3,
        seed=42,
    )
    assert "roc_auc" in metrics
    assert "recall" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_clv_regressor_evaluation(mock_dataset):
    data = split_data(mock_dataset, save_scaler=False)
    regressors = initialize_regressors(seed=42)

    # Test Ridge on the log1p pipeline
    ridge_item = regressors["Ridge Regression (log1p)"]
    train = data["X_train"].copy()
    test = data["X_test"].copy()
    train["spend_last_90d"] = data["aux_train"]
    test["spend_last_90d"] = data["aux_test"]
    metrics = evaluate_regressor(
        model=ridge_item["model"],
        X_train=train,
        y_train=data["clv_train"],
        X_test=test,
        y_test=data["clv_test"],
        use_log_transform=True,
    )
    assert "r2_score" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "value_capture_top_decile" in metrics


def test_segmentation_k_range():
    X_sample = np.random.randn(50, 5)
    best_k, k_df = evaluate_kmeans_k_range(X_sample, k_min=2, k_max=4)
    assert best_k in [2, 3, 4]
    assert len(k_df) == 3
    assert "silhouette" in k_df.columns


def test_assign_business_personas():
    mock_profiles = pd.DataFrame([
        {"cluster": 0, "recency": 15.0, "frequency": 12.0, "monetary_total": 4500.0, "engagement_score": 88.0},
        {"cluster": 1, "recency": 250.0, "frequency": 1.0, "monetary_total": 80.0, "engagement_score": 15.0},
    ])
    personas = assign_business_personas(mock_profiles)
    assert personas[0] == "High-Value Champions (VIP)"
    assert personas[1] == "Dormant / Low-Engagement"
