"""Budget-constrained retention targeting (Beyond PRD, plan P3).

Turns scores into a *campaign decision*: given a budget B, a per-channel contact
cost c and an assumed save rate s, choose which customers to contact to maximise

    sum( p_churn * value * s - c )   subject to   sum(c) <= B

This is a knapsack problem; the greedy ratio rule (value-at-risk per £ spent) is
sufficient and is validated against a brute-force optimum in the tests.

There is no treatment/outcome data, so ``s`` is an assumption. The offline policy
evaluation therefore reports a **sensitivity grid** over s, and the report says so.
"""

from typing import Any, Dict, List, Tuple

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("campaign")


def channel_costs() -> Dict[str, float]:
    return load_config().get("campaign", {}).get("channel_costs", {"None": 0.0})


def _cost(row: Dict[str, Any], costs: Dict[str, float]) -> float:
    return float(costs.get(row.get("channel", "None"), costs.get("Email", 2.0)))


def greedy_select(
    candidates: List[Dict[str, Any]],
    budget: float,
    save_rate: float,
    costs: Dict[str, float] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Greedy knapsack by expected net gain per £ of contact cost.

    Each candidate needs ``value_at_risk`` (= calibrated P(churn) x 90-day value),
    ``channel`` and optionally ``priority_score``. Returns the selected rows
    (annotated) plus campaign totals.
    """
    costs = costs or channel_costs()
    scored = []
    for row in candidates:
        var = float(row.get("value_at_risk", 0.0))
        c = _cost(row, costs)
        expected_gain = save_rate * var - c
        ratio = expected_gain / c if c > 0 else float("inf")
        scored.append({**row, "_cost": c, "_expected_gain": expected_gain, "_ratio": ratio})

    # Positive-gain customers first, then by value per £; free channels ranked by VaR.
    scored.sort(key=lambda r: (r["_expected_gain"] > 0, r["_ratio"], r["value_at_risk"]), reverse=True)

    selected, spent = [], 0.0
    for row in scored:
        if row["_expected_gain"] <= 0:
            break
        if spent + row["_cost"] <= budget:
            selected.append({k: v for k, v in row.items() if not k.startswith("_")})
            spent += row["_cost"]

    retained = sum(save_rate * float(r["value_at_risk"]) for r in selected)
    totals = {
        "n_contacted": len(selected),
        "budget": float(budget),
        "spent": round(spent, 2),
        "save_rate": float(save_rate),
        "expected_value_retained": round(retained, 2),
        "expected_net_gain": round(retained - spent, 2),
        "value_per_pound": round(retained / spent, 4) if spent > 0 else 0.0,
    }
    return selected, totals


def plan_campaign(
    candidates: List[Dict[str, Any]],
    budget: float | None = None,
    save_rate: float | None = None,
) -> Dict[str, Any]:
    """Plan one campaign at the default (or supplied) budget and save rate."""
    cfg = load_config().get("campaign", {})
    budget = float(budget if budget is not None else cfg.get("default_budget", 5000.0))
    save_rate = float(save_rate if save_rate is not None else cfg.get("default_save_rate", 0.10))
    selected, totals = greedy_select(candidates, budget, save_rate)
    return {"selection": selected, "totals": totals}


def policy_comparison(
    candidates: List[Dict[str, Any]], budget: float | None = None
) -> List[Dict[str, Any]]:
    """Compare four policies on the same candidate pool with the same budget."""
    cfg = load_config().get("campaign", {})
    budget = float(budget if budget is not None else cfg.get("default_budget", 5000.0))
    save_rate = float(cfg.get("default_save_rate", 0.10))
    costs = channel_costs()

    def _evaluate(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        spent = sum(_cost(r, costs) for r in rows)
        retained = sum(save_rate * float(r.get("value_at_risk", 0.0)) for r in rows)
        return {
            "policy": name,
            "n_contacted": len(rows),
            "spent": round(spent, 2),
            "expected_value_retained": round(retained, 2),
            "value_per_pound": round(retained / spent, 4) if spent > 0 else 0.0,
        }

    # (a) top churn probability
    a = sorted(candidates, key=lambda r: float(r.get("calibrated_probability", 0.0)), reverse=True)
    # (b) top spenders (historical value proxy = value_at_risk / p)
    b = sorted(candidates, key=lambda r: float(r.get("predicted_value", 0.0)), reverse=True)
    # (c) current rule table: one per rule, by priority
    c = sorted(candidates, key=lambda r: float(r.get("priority_score", 0.0)), reverse=True)
    # (d) optimised value-at-risk policy
    d, _ = greedy_select(candidates, budget, save_rate, costs)

    def _take_within_budget(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out, spent = [], 0.0
        for r in rows:
            cost = _cost(r, costs)
            if r.get("channel") in ("None", None) or cost == 0:
                out.append(r)
            elif spent + cost <= budget:
                out.append(r)
                spent += cost
        return out

    return [
        _evaluate("(a) Top churn probability", _take_within_budget(a)),
        _evaluate("(b) Top spenders", _take_within_budget(b)),
        _evaluate("(c) Current rule table", _take_within_budget(c)),
        _evaluate("(d) Optimised value-at-risk", d),
    ]


def sensitivity_grid(
    candidates: List[Dict[str, Any]],
    budgets: List[float] | None = None,
    save_rates: List[float] | None = None,
) -> List[Dict[str, Any]]:
    """Run the optimiser across a grid of budgets x assumed save rates."""
    cfg = load_config().get("campaign", {})
    budgets = budgets or [1000.0, 5000.0, 20000.0]
    save_rates = save_rates or cfg.get("save_rate_grid", [0.05, 0.10, 0.20])
    rows = []
    for budget in budgets:
        for s in save_rates:
            _, totals = greedy_select(candidates, budget, s)
            rows.append({
                "budget": budget,
                "save_rate": s,
                "n_contacted": totals["n_contacted"],
                "spent": totals["spent"],
                "expected_value_retained": totals["expected_value_retained"],
                "expected_net_gain": totals["expected_net_gain"],
            })
    return rows
