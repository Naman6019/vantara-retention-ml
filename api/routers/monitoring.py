"""Monitoring, revenue and campaign-planning endpoints (plan D6/D7, Beyond PRD P3)."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.serving import db

router = APIRouter()


class CampaignPlanRequest(BaseModel):
    budget: float = Field(5000.0, gt=0, description="Total contact budget (£)")
    save_rate: float = Field(0.10, gt=0, le=1.0, description="Assumed share of contacted churners retained")
    limit: int = Field(500, ge=1, le=2000, description="Candidate pool size from the at-risk leaderboard")


@router.post("/campaign/plan")
def campaign_plan(request: Request, body: Optional[CampaignPlanRequest] = None) -> Dict[str, Any]:
    """Budget-constrained retention targeting (Beyond PRD P3).

    Reads the value-at-risk leaderboard, then greedily selects the customers that
    maximise expected retained value minus contact cost within the budget. Also
    returns a policy comparison and a save-rate sensitivity grid.
    """
    from sqlalchemy import text

    from src.recommendations.campaign import plan_campaign, policy_comparison, sensitivity_grid

    body = body or CampaignPlanRequest()
    if db.get_engine() is None:
        db.ensure_engine()
    engine = db.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT p.customer_id, p.calibrated_probability, p.predicted_value, p.value_at_risk,
                       p.value_tier, r.channel, r.priority_score
                FROM predictions p LEFT JOIN recommendations r ON r.customer_id = p.customer_id
                ORDER BY p.value_at_risk DESC LIMIT :limit
                """
            ),
            {"limit": body.limit},
        ).fetchall()
    candidates = [dict(r._mapping) for r in rows]

    plan = plan_campaign(candidates, budget=body.budget, save_rate=body.save_rate)
    return {
        "totals": plan["totals"],
        "n_selected": len(plan["selection"]),
        "selection": plan["selection"][:100],
        "policy_comparison": policy_comparison(candidates, budget=body.budget),
        "sensitivity": sensitivity_grid(candidates),
    }


@router.get("/monitoring/drift")
def monitoring_drift() -> List[Dict[str, Any]]:
    """Per-feature PSI from the most recent drift report."""
    from sqlalchemy import text

    if db.get_engine() is None:
        db.ensure_engine()
    engine = db.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    with engine.connect() as conn:
        latest = conn.execute(text("SELECT MAX(batch_date) AS d FROM drift_reports")).fetchone()
        batch_date = latest._mapping["d"] if latest else None
        if batch_date is None:
            return []
        rows = conn.execute(
            text(
                "SELECT feature, psi, drifted, batch_date FROM drift_reports "
                "WHERE batch_date = :d ORDER BY psi DESC"
            ),
            {"d": batch_date},
        ).fetchall()
    return [dict(r._mapping) for r in rows]


@router.get("/revenue/monthly")
def revenue_monthly(request: Request) -> List[Dict[str, Any]]:
    """Monthly revenue read from the cleaned transactions when available."""
    import pandas as pd

    from src.utils.config import get_project_root

    path = get_project_root() / "data/interim/cleaned_transactions.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Cleaned transactions not available")
    df = pd.read_parquet(path, columns=["invoice_date", "line_total"])
    monthly = (
        df.groupby(df["invoice_date"].dt.to_period("M"))["line_total"]
        .sum()
        .reset_index()
        .rename(columns={"invoice_date": "month", "line_total": "revenue"})
    )
    monthly["month"] = monthly["month"].astype(str)
    return monthly.to_dict(orient="records")


@router.get("/monitoring/performance")
def monitoring_performance(request: Request) -> Dict[str, Any]:
    """Delayed-label performance recorded on the model card (AUC/recall per split)."""
    card = request.app.state.artifacts["card"]
    return {
        "model_version": card["model_version"],
        "validation": card.get("validation_metrics", {}),
        "test": card.get("test_metrics", {}),
        "clv_validation": card.get("clv_validation_metrics", {}),
        "clv_test": card.get("clv_test_metrics", {}),
    }
