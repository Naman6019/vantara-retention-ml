"""Rule-based retention recommendation engine (plan D5).

There is no treatment/outcome history in the dataset, so we cannot *learn* which
intervention works (uplift modelling). Instead we map a customer's (value tier,
risk tier, top SHAP drivers, segment) onto an action using an editable rule
table, and every recommendation cites the trigger that fired it.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("recommendations")

DEFAULT_RULES_PATH = "config/recommendations.yaml"


def load_rules(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the rule table from YAML (marketing can edit this freely)."""
    root = get_project_root()
    config = load_config()
    if path is None:
        rel = config.get("recommendations", {}).get("rules_file", DEFAULT_RULES_PATH)
        path = root / rel
    if not path.exists():
        raise FileNotFoundError(f"Recommendation rules not found at: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def value_tier_from_value(value: float, q_low: float, q_high: float) -> str:
    """Bucket a predicted value into low/mid/high using training quantiles."""
    if value >= q_high:
        return "high"
    if value >= q_low:
        return "mid"
    return "low"


def _driver_names(shap_top: List[Dict[str, Any]]) -> set:
    names = set()
    for d in shap_top or []:
        if isinstance(d, dict):
            names.add(d.get("feature"))
        else:
            names.add(str(d))
    return {n for n in names if n}


def _matches(when: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
    if not when:
        return True
    if "risk_tier" in when and str(ctx["risk_tier"]).lower() not in [t.lower() for t in when["risk_tier"]]:
        return False
    if "value_tier" in when and ctx["value_tier"] not in when["value_tier"]:
        return False
    if "drivers_any" in when and not (_driver_names(ctx["shap_top"]) & set(when["drivers_any"])):
        return False
    if "segment_any" in when:
        persona = str(ctx.get("segment", ""))
        if not any(s.lower() in persona.lower() for s in when["segment_any"]):
            return False
    if "min_monetary" in when and float(ctx.get("monetary_total", 0.0)) < float(when["min_monetary"]):
        return False
    return True


def recommend(
    row: Dict[str, Any],
    shap_top: Optional[List[Dict[str, Any]]] = None,
    card: Optional[Dict[str, Any]] = None,
    rules: Optional[Dict[str, Any]] = None,
    q_low: float = 0.0,
    q_high: float = 1e12,
) -> Dict[str, Any]:
    """Return one recommendation: action, channel, reasons, priority score.

    ``row`` must contain at least ``risk_tier`` (Low/Moderate/High) and a
    predicted value; ``shap_top`` is the local explanation's top drivers.
    """
    if rules is None:
        rules = load_rules()
    shap_top = shap_top or []

    predicted_value = float(row.get("predicted_value", row.get("predicted_future_clv", 0.0)))
    churn_prob = float(row.get("churn_probability", 0.0))
    value_tier = row.get("value_tier") or value_tier_from_value(predicted_value, q_low, q_high)

    ctx: Dict[str, Any] = {
        "risk_tier": str(row.get("risk_tier", "Low")),
        "value_tier": value_tier,
        "shap_top": shap_top,
        "segment": row.get("segment", row.get("segment_persona", "")),
        "monetary_total": float(row.get("monetary_total", 0.0)),
        "predicted_value": predicted_value,
        "churn_probability": churn_prob,
    }

    for rule in rules.get("rules", []):
        if _matches(rule.get("when", {}), ctx):
            # A single, human-readable reason assembled from the rule + the top driver.
            reason = rule.get("reason", "")
            top = _names_in_order(shap_top)
            if top:
                reason = f"{reason} (top driver: {top[0].replace('_', ' ')})"
            return {
                "action": rule.get("action"),
                "channel": rule.get("channel"),
                "rule_id": rule.get("id"),
                "reasons": [reason],
                "priority_score": round(churn_prob * predicted_value, 2),
            }

    # Defensive fallback (a fallback rule should always exist in the YAML).
    return {
        "action": "Monitor only",
        "channel": "None",
        "rule_id": "fallback",
        "reasons": ["No retention trigger matched"],
        "priority_score": round(churn_prob * predicted_value, 2),
    }


def _names_in_order(shap_top: List[Dict[str, Any]]) -> List[str]:
    return [d.get("feature") for d in shap_top if isinstance(d, dict) and d.get("feature")]


def recommend_batch(
    rows: List[Dict[str, Any]],
    shap_lookup: Optional[Dict[Any, List[Dict[str, Any]]]] = None,
    card: Optional[Dict[str, Any]] = None,
    q_low: float = 0.0,
    q_high: float = 1e12,
) -> List[Dict[str, Any]]:
    """Apply ``recommend`` to a batch of customer rows."""
    rules = load_rules()
    shap_lookup = shap_lookup or {}
    return [
        recommend(
            row,
            shap_top=shap_lookup.get(row.get("customer_id"), []),
            card=card,
            rules=rules,
            q_low=q_low,
            q_high=q_high,
        )
        for row in rows
    ]
