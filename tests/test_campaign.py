import itertools

import pytest

from src.recommendations.campaign import (
    greedy_select,
    plan_campaign,
    policy_comparison,
    sensitivity_grid,
)

COSTS = {"Email": 10.0, "Account Manager": 10.0, "None": 0.0}


def _candidates():
    return [
        {"customer_id": 1, "value_at_risk": 500.0, "calibrated_probability": 0.9, "predicted_value": 555.0, "channel": "Email", "priority_score": 450},
        {"customer_id": 2, "value_at_risk": 300.0, "calibrated_probability": 0.8, "predicted_value": 375.0, "channel": "Email", "priority_score": 300},
        {"customer_id": 3, "value_at_risk": 200.0, "calibrated_probability": 0.7, "predicted_value": 285.0, "channel": "Email", "priority_score": 200},
        {"customer_id": 4, "value_at_risk": 100.0, "calibrated_probability": 0.6, "predicted_value": 166.0, "channel": "Email", "priority_score": 100},
    ]


def _brute_force_gain(cands, budget, save_rate, costs):
    best = 0.0
    n = len(cands)
    for r in range(n + 1):
        for combo in itertools.combinations(cands, r):
            cost = sum(costs.get(c["channel"], 0.0) for c in combo)
            if cost > budget:
                continue
            gain = sum(save_rate * c["value_at_risk"] - costs.get(c["channel"], 0.0) for c in combo)
            best = max(best, gain)
    return best


def test_greedy_respects_budget():
    selected, totals = greedy_select(_candidates(), budget=25.0, save_rate=0.10, costs=COSTS)
    assert totals["spent"] <= 25.0
    assert totals["n_contacted"] == len(selected)


def test_greedy_is_optimal_on_equal_cost_knapsack():
    # With equal per-item costs the ratio order is the value order, so greedy is
    # exactly optimal — verified against brute force.
    cands = _candidates()
    budget = 25.0
    save_rate = 0.10
    chosen, totals = greedy_select(cands, budget, save_rate, COSTS)
    brute = _brute_force_gain(cands, budget, save_rate, COSTS)
    assert totals["expected_net_gain"] == pytest.approx(brute, abs=1e-6)


def test_greedy_skips_negative_gain_items():
    # A high value-at-risk can still be loss-making if the contact cost exceeds
    # the expected retained value.
    cands = [{"customer_id": 9, "value_at_risk": 20.0, "channel": "Account Manager"}]
    selected, totals = greedy_select(cands, budget=100.0, save_rate=0.05, costs=COSTS)
    assert selected == []
    assert totals["spent"] == 0.0


def test_policy_comparison_returns_four_policies():
    rows = policy_comparison(_candidates(), budget=25.0)
    assert len(rows) == 4
    names = " ".join(r["policy"] for r in rows)
    assert "Optimised value-at-risk" in names
    assert all(r["spent"] <= 25.0 + 1e-6 for r in rows)


def test_sensitivity_grid_shape():
    grid = sensitivity_grid(_candidates(), budgets=[50.0, 100.0], save_rates=[0.05, 0.1])
    assert len(grid) == 4
    assert {"budget", "save_rate", "expected_net_gain"}.issubset(grid[0].keys())


def test_plan_campaign_defaults():
    result = plan_campaign(_candidates())
    assert "selection" in result and "totals" in result
    assert result["totals"]["budget"] > 0
