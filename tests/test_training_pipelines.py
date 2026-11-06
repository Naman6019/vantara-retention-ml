
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC

from src.models import calibration, churn_models, clv_regressor
from src.models.model_card import build_model_card, read_model_card, write_model_card


@pytest.fixture
def tiny_split(feature_table):
    from src.models.data_split import split_data

    return split_data(feature_table, save_scaler=False)


def test_initialize_models_contains_required_six():
    models = churn_models.initialize_models(seed=1)
    assert set(models) == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "KNN",
    }
    for item in models.values():
        assert "model" in item and "use_scaled" in item


def test_evaluate_model_decision_function_branch(tiny_split):
    clf = LinearSVC(max_iter=200)
    metrics = churn_models.evaluate_model(
        model=clf,
        X_train=tiny_split["X_train_scaled"],
        y_train=tiny_split["y_train"],
        X_test=tiny_split["X_test_scaled"],
        y_test=tiny_split["y_test"],
        cv_folds=2,
    )
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert "confusion_matrix" in metrics


def test_select_best_uses_validation_metric_only():
    val = {
        "A": {"roc_auc": 0.71},
        "B": {"roc_auc": 0.83},
        "C": {"roc_auc": 0.79},
    }
    assert churn_models.select_best(val) == "B"
    # Signature carries no test data at all (leakage guard).
    import inspect

    params = inspect.signature(churn_models.select_best).parameters
    assert "test" not in " ".join(params).lower()


def test_forward_chaining_cv_reports_score(tiny_split):
    model = LogisticRegression(max_iter=200)
    score = churn_models.forward_chaining_cv_auc(
        model, tiny_split["X_train_scaled"], tiny_split["y_train"], tiny_split["cutoff_train"], n_splits=2
    )
    assert 0.0 <= score <= 1.0


def test_tune_and_compare_churn_models_tiny(tiny_split):
    leaderboard, fitted = churn_models.tune_and_compare_churn_models(tiny_split, n_trials=1)
    assert isinstance(leaderboard, pd.DataFrame)
    assert {"Model", "Val ROC-AUC", "CV ROC-AUC (fwd-chain)"}.issubset(leaderboard.columns)
    assert len(fitted) == 6


def test_save_leaderboard_writes_csv(tiny_split, tmp_path):
    leaderboard, _ = churn_models.tune_and_compare_churn_models(tiny_split, n_trials=1)
    churn_models.save_leaderboard(leaderboard, tmp_path)
    assert (tmp_path / "churn_model_comparison.csv").exists()


def test_initialize_regressors_includes_baseline_tweedie_forest_ridge():
    regressors = clv_regressor.initialize_regressors(seed=1)
    assert set(regressors) == {
        "Spend Baseline (last 90d)",
        "Ridge Regression (log1p)",
        "LightGBM Tweedie",
        "Random Forest Regressor",
    }


def test_top_decile_value_capture_bounds():
    import numpy as np

    y_true = np.array([0.0, 0.0, 0.0, 100.0])
    y_pred = np.array([0.1, 0.1, 0.1, 1.0])
    capture = clv_regressor.top_decile_value_capture(y_true, y_pred, decile=0.25)
    assert capture == pytest.approx(1.0)


def test_evaluate_regressor_with_log_transform(tiny_split):
    metrics = clv_regressor.evaluate_regressor(
        model=Ridge(alpha=1.0),
        X_train=tiny_split["X_train_scaled"],
        y_train=tiny_split["clv_train"],
        X_test=tiny_split["X_test_scaled"],
        y_test=tiny_split["clv_test"],
        use_log_transform=True,
    )
    assert "r2_score" in metrics
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0
    assert "value_capture_top_decile" in metrics


def test_tune_and_compare_clv_regressors_tiny(tiny_split):
    leaderboard, fitted = clv_regressor.tune_and_compare_clv_regressors(tiny_split)
    assert isinstance(leaderboard, pd.DataFrame)
    assert {"Model", "Val MAE (£)", "Top-Decile Value Capture"}.issubset(leaderboard.columns)
    assert len(fitted) == 4


# ---------------------------------------------------------------------------
# Calibration (plan D4)
# ---------------------------------------------------------------------------

def test_choose_threshold_meets_recall_target():
    import numpy as np

    y = np.array([0, 0, 0, 1, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.55, 0.6, 0.75, 0.9])
    info = calibration.choose_threshold(y, probs, recall_target=0.75)
    assert info["recall"] >= 0.75
    # Highest threshold that still satisfies the recall floor.
    assert info["threshold"] > 0.3


def test_calibrator_fit_predict_and_tiers():
    import numpy as np

    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 300)
    probs = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.1, 300), 0, 1)
    cal = calibration.ProbabilityCalibrator(method="isotonic").fit(probs, y)
    out = cal.predict(probs)
    assert out.min() >= 0.0 and out.max() <= 1.0
    tiers = cal.tiers(out)
    assert set(tiers).issubset({"Low", "Moderate", "High"})
    # Calibrated probability should be informative about the label.
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, out) > 0.70


def test_calibrator_save_load_roundtrip(tmp_path):
    import numpy as np

    y = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    cal = calibration.ProbabilityCalibrator().fit(probs, y)
    path = tmp_path / "cal.joblib"
    cal.save(path)
    loaded = calibration.ProbabilityCalibrator.load(path)
    assert np.allclose(loaded.predict(probs), cal.predict(probs))


# ---------------------------------------------------------------------------
# Model card (plan D8)
# ---------------------------------------------------------------------------

def test_model_card_roundtrip(monkeypatch, tmp_path):
    from src.models import model_card as mc

    monkeypatch.setattr(mc, "get_project_root", lambda: tmp_path)
    card = build_model_card(
        churn_model_name="XGBoost",
        churn_model_path="models_artifacts/churn_best_model.joblib",
        clv_model_name="LightGBM Tweedie",
        clv_model_path="models_artifacts/clv_best_model.joblib",
        feature_cols=["recency", "frequency"],
        train_cutoffs=["2010-06-01"],
        val_cutoff="2011-06-01",
        test_cutoff="2011-09-01",
        threshold=0.42,
        calibrator_path="models_artifacts/churn_calibrator.joblib",
        validation_metrics={"roc_auc": 0.8},
        test_metrics={"roc_auc": 0.78},
    )
    path = write_model_card(card, tmp_path / "models_artifacts" / "model_card.json")
    loaded = read_model_card(path)
    assert loaded["churn_model"] == "XGBoost"
    assert loaded["decision_threshold"] == 0.42
    assert loaded["feature_cols"] == ["recency", "frequency"]
    assert loaded["test_metrics"]["roc_auc"] == 0.78


def test_read_model_card_missing_raises(monkeypatch, tmp_path):
    from src.models import model_card as mc

    monkeypatch.setattr(mc, "get_project_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        read_model_card()
