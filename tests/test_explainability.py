import matplotlib

matplotlib.use("Agg")

import joblib
import pytest
import xgboost as xgb

from src.explainability import explainer as explainer_mod
from src.explainability.explainer import ModelExplainer, run_explainability_suite


@pytest.fixture
def trained_model(split_result):
    model = xgb.XGBClassifier(
        n_estimators=15, max_depth=3, learning_rate=0.1, random_state=0, n_jobs=1
    )
    model.fit(split_result["X_train"], split_result["y_train"])
    return model


@pytest.fixture
def explainer(monkeypatch, tmp_path, split_result, trained_model):
    art = tmp_path / "models_artifacts"
    art.mkdir(parents=True)
    joblib.dump(trained_model, art / "churn_xgboost.joblib")

    monkeypatch.setattr(explainer_mod, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(explainer_mod, "split_data", lambda: split_result)
    return ModelExplainer()


def test_explainer_loads_model(explainer, trained_model):
    assert type(explainer.model).__name__ == "XGBClassifier"
    assert explainer.feature_names
    assert len(explainer.X_train) > 0


def test_generate_global_shap_summary(explainer, tmp_path, monkeypatch):
    monkeypatch.setattr(explainer_mod, "get_project_root", lambda: tmp_path)
    path = explainer.generate_global_shap_summary()
    assert path.exists()
    assert path.name == "shap_global_summary.png"


def test_explain_single_customer_shap(explainer):
    result = explainer.explain_single_customer_shap(0)
    assert "base_value" in result
    assert len(result["top_drivers"]) == 6
    assert len(result["all_impacts"]) == len(explainer.feature_names)
    for driver in result["top_drivers"]:
        assert driver["direction"] in {"Increases Churn", "Decreases Churn"}
    impacts = [abs(d["shap_impact"]) for d in result["top_drivers"]]
    assert impacts == sorted(impacts, reverse=True)


def test_explain_single_customer_lime(explainer):
    exp = explainer.explain_single_customer_lime(1)
    assert isinstance(exp, list)
    assert len(exp) > 0
    rule, weight = exp[0]
    assert isinstance(rule, str)
    assert isinstance(weight, float)


def test_generate_partial_dependence_plots(explainer, tmp_path, monkeypatch):
    monkeypatch.setattr(explainer_mod, "get_project_root", lambda: tmp_path)
    path = explainer.generate_partial_dependence_plots(features=["recency", "frequency"])
    assert path.exists()
    assert path.name == "pdp_plots.png"


def test_translate_to_narrative_high_risk():
    shap_exp = {
        "top_drivers": [
            {"feature": "recency", "value": 300.0, "shap_impact": 0.8, "direction": "Increases Churn"},
            {"feature": "engagement_score", "value": 5.0, "shap_impact": 0.5, "direction": "Increases Churn"},
        ]
    }
    narrative = ModelExplainer.translate_to_narrative(None, shap_exp, 0.9)
    assert "High Churn Risk" in narrative
    assert "Primary churn warning signs" in narrative


def test_translate_to_narrative_low_risk_with_mitigators():
    shap_exp = {
        "top_drivers": [
            {"feature": "recency", "value": 2.0, "shap_impact": -0.9, "direction": "Decreases Churn"},
            {"feature": "frequency", "value": 20.0, "shap_impact": -0.4, "direction": "Decreases Churn"},
        ]
    }
    narrative = ModelExplainer.translate_to_narrative(None, shap_exp, 0.05)
    assert "Low Churn Risk" in narrative
    assert "Retention mitigating factors" in narrative


def test_run_explainability_suite(monkeypatch, tmp_path, split_result, trained_model):
    art = tmp_path / "models_artifacts"
    art.mkdir(parents=True)
    joblib.dump(trained_model, art / "churn_xgboost.joblib")
    monkeypatch.setattr(explainer_mod, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(explainer_mod, "split_data", lambda: split_result)

    result = run_explainability_suite()

    assert "global_shap_plot" in result
    assert "pdp_plot" in result
    assert set(result["representative_personas"]) == {
        "Low-Risk Customer",
        "High-Risk Customer",
        "Borderline Customer",
    }
    assert len(result["lime_borderline_explanation"]) > 0
    for details in result["representative_personas"].values():
        assert 0.0 <= details["churn_prob"] <= 1.0
        assert details["narrative"]
