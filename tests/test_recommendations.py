from src.recommendations import engine
from src.recommendations.engine import recommend, value_tier_from_value


def _shap(*features):
    return [{"feature": f} for f in features]


def test_high_value_high_risk_recency_rule():
    row = {"risk_tier": "High", "value_tier": "high", "monetary_total": 5000.0, "predicted_value": 900.0, "churn_probability": 0.8}
    rec = recommend(row, shap_top=_shap("recency", "frequency"))
    assert rec["action"] == "Account manager call within 7 days"
    assert rec["rule_id"] == "high_value_high_risk_recency"
    assert rec["priority_score"] == 720.0


def test_high_value_high_risk_without_recency_driver():
    row = {"risk_tier": "High", "value_tier": "high", "monetary_total": 5000.0, "predicted_value": 900.0, "churn_probability": 0.8}
    rec = recommend(row, shap_top=_shap("frequency", "monetary_total"))
    assert rec["rule_id"] == "high_value_high_risk"


def test_return_rate_driver_rule():
    row = {"risk_tier": "Moderate", "value_tier": "mid", "predicted_value": 300.0, "churn_probability": 0.5}
    rec = recommend(row, shap_top=_shap("return_rate"))
    assert rec["action"] == "Service-recovery outreach, no discount"


def test_discount_sensitive_mid_value_rule():
    row = {"risk_tier": "Low", "value_tier": "mid", "predicted_value": 250.0, "churn_probability": 0.2}
    rec = recommend(row, shap_top=_shap("discount_sensitivity"))
    assert rec["action"] == "Targeted 10% voucher"


def test_low_value_high_risk_rule():
    row = {"risk_tier": "High", "value_tier": "low", "predicted_value": 20.0, "churn_probability": 0.9}
    rec = recommend(row, shap_top=_shap("frequency"))
    assert rec["action"] == "Automated win-back email"


def test_low_risk_high_value_rule():
    row = {"risk_tier": "Low", "value_tier": "high", "predicted_value": 1500.0, "churn_probability": 0.05}
    rec = recommend(row, shap_top=_shap("recency"))
    assert rec["action"] == "Loyalty / early-access offer, no discount"


def test_fallback_rule_matched():
    row = {"risk_tier": "Low", "value_tier": "low", "predicted_value": 10.0, "churn_probability": 0.1}
    rec = recommend(row, shap_top=[])
    assert rec["rule_id"] == "fallback"
    assert rec["reasons"]


def test_every_customer_gets_one_action_and_reason():
    rows = [
        {"customer_id": 1, "risk_tier": "High", "value_tier": "high", "predicted_value": 900.0, "churn_probability": 0.8},
        {"customer_id": 2, "risk_tier": "Low", "value_tier": "low", "predicted_value": 5.0, "churn_probability": 0.1},
    ]
    out = engine.recommend_batch(rows, shap_lookup={1: _shap("recency")})
    assert len(out) == 2
    assert all(r["action"] and r["reasons"] for r in out)


def test_value_tier_from_value():
    assert value_tier_from_value(5.0, q_low=100.0, q_high=500.0) == "low"
    assert value_tier_from_value(200.0, q_low=100.0, q_high=500.0) == "mid"
    assert value_tier_from_value(900.0, q_low=100.0, q_high=500.0) == "high"


def test_load_rules_has_fallback():
    rules = engine.load_rules()
    ids = [r["id"] for r in rules["rules"]]
    assert "fallback" in ids
