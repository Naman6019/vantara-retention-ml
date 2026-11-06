"""Customer, segment and value-at-risk endpoints backed by PostgreSQL (plan D6)."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from api.state import get_artifacts
from src.serving import db
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger("api_customers")


def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(r._mapping) for r in rows]


def _as_dict(value: Any) -> Dict[str, Any]:
    """JSON columns come back as dicts (PostgreSQL) or strings (SQLite)."""
    import json

    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


@router.get("/customers/at-risk")
def at_risk_customers(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    segment: Optional[str] = None,
    country: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Value-at-risk leaderboard, sorted by calibrated P(churn) x predicted value."""
    from sqlalchemy import text

    if db.get_engine() is None:
        db.ensure_engine()
    engine = db.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    sql = """
        SELECT p.customer_id, p.calibrated_probability, p.predicted_value, p.value_at_risk,
               p.risk_tier, p.value_tier, s.persona, r.action, r.channel, r.priority_score
        FROM predictions p
        LEFT JOIN segments s ON s.customer_id = p.customer_id
        LEFT JOIN recommendations r ON r.customer_id = p.customer_id
        WHERE 1=1
    """
    params: Dict[str, Any] = {"limit": limit}
    if segment:
        sql += " AND s.persona LIKE :segment"
        params["segment"] = f"%{segment}%"
    if country:
        sql += " AND p.customer_id IN (SELECT customer_id FROM customers WHERE primary_country = :country)"
        params["country"] = country
    sql += " ORDER BY p.value_at_risk DESC LIMIT :limit"

    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return _rows_to_dicts(rows)


@router.get("/customers/{customer_id}")
def get_customer(request: Request, customer_id: float) -> Dict[str, Any]:
    """Full Customer-360 record: prediction, segment, recommendation and SHAP drivers."""
    from sqlalchemy import text

    if db.get_engine() is None:
        db.ensure_engine()
    engine = db.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    sql = """
        SELECT p.customer_id, p.calibrated_probability, p.churn_probability, p.predicted_value,
               p.value_at_risk, p.risk_tier, p.value_tier, p.shap_top,
               s.persona, s.kmeans_cluster, r.action, r.channel, r.reasons, r.priority_score
        FROM predictions p
        LEFT JOIN segments s ON s.customer_id = p.customer_id
        LEFT JOIN recommendations r ON r.customer_id = p.customer_id
        WHERE p.customer_id = :cid
        ORDER BY p.scored_at DESC
        LIMIT 1
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql), {"cid": customer_id}).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return dict(row._mapping)


@router.get("/customers/{customer_id}/recommendations")
def customer_recommendations(request: Request, customer_id: float) -> List[Dict[str, Any]]:
    """Recommended next products: personalised, or labelled best-sellers if there is no history."""
    from sqlalchemy import text

    from src.recommendations.products import recommend_for_customer

    trained = recommend_for_customer(customer_id, top_k=10)
    if trained:
        return trained

    if db.get_engine() is None:
        db.ensure_engine()
    engine = db.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    # Products this customer already bought.
    with engine.connect() as conn:
        bought = conn.execute(
            text("SELECT DISTINCT product_id FROM customer_products WHERE customer_id = :cid"),
            {"cid": customer_id},
        ).fetchall()
        bought_ids = {r[0] for r in bought}
        top = conn.execute(
            text(
                "SELECT product_id, COUNT(*) AS buyers, SUM(quantity) AS units "
                "FROM customer_products WHERE product_id != '' "
                "GROUP BY product_id ORDER BY buyers DESC LIMIT 40"
            )
        ).fetchall()

    return [
        {"product_id": r[0], "buyers": int(r[1]), "units": float(r[2] or 0), "source": "popularity"}
        for r in top
        if r[0] not in bought_ids
    ][:10]


@router.get("/customers/{customer_id}/explain")
def explain_customer(request: Request, customer_id: float) -> Dict[str, Any]:
    """Per-customer SHAP and LIME explanation, plus a plain-language narrative.

    Both explainers use the served model and the stored feature vector, so the
    dashboard never loads model artifacts directly (PRD acceptance criterion 4).
    """
    import numpy as np
    from sqlalchemy import text

    if db.get_engine() is None:
        db.ensure_engine()
    engine = db.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT features, calibrated_probability, risk_tier "
                "FROM predictions WHERE customer_id = :cid ORDER BY scored_at DESC LIMIT 1"
            ),
            {"cid": customer_id},
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    import pandas as pd


    art = get_artifacts(request)
    feature_cols = art["card"]["feature_cols"]
    vector = {c: float(_as_dict(row._mapping["features"]).get(c, 0.0) or 0.0) for c in feature_cols}
    X = pd.DataFrame([vector])[feature_cols]

    import shap

    model = art["churn_model"]
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:  # noqa: BLE001
        explainer = shap.LinearExplainer(model, X)
    values = np.asarray(explainer.shap_values(X))
    if values.ndim == 3:
        values = values[:, :, -1]
    row_vals = values[0]
    shap_drivers = sorted(
        [
            {
                "feature": c,
                "value": round(float(X.iloc[0][c]), 4),
                "shap_impact": round(float(row_vals[j]), 4),
                "direction": "Increases Churn" if row_vals[j] > 0 else "Decreases Churn",
            }
            for j, c in enumerate(feature_cols)
        ],
        key=lambda d: abs(d["shap_impact"]),
        reverse=True,
    )[:8]

    # LIME on the same customer.
    lime_drivers: List[Dict[str, Any]] = []
    try:
        from lime.lime_tabular import LimeTabularExplainer

        background = art.get("background", X)
        lime_explainer = LimeTabularExplainer(
            training_data=background.to_numpy(),
            feature_names=feature_cols,
            class_names=["Retained", "Churned"],
            mode="classification",
            random_state=42,
        )
        exp = lime_explainer.explain_instance(
            X.iloc[0].to_numpy(), model.predict_proba, num_features=8
        )
        lime_drivers = [
            {"rule": rule, "weight": round(float(weight), 4)} for rule, weight in exp.as_list()
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"LIME explanation skipped: {exc}")

    prob = float(row._mapping["calibrated_probability"])
    pos = [d for d in shap_drivers if d["shap_impact"] > 0]
    narrative = f"Calibrated churn risk {prob:.1%} ({row._mapping['risk_tier']}). "
    if pos:
        narrative += f"Top warning sign: {pos[0]['feature'].replace('_', ' ')}."

    # Beyond PRD P1: probabilistic P(alive) from BG/NBD for the same customer.
    p_alive = None
    try:
        import json as _json

        import pandas as _pd

        from src.models.probabilistic_clv import (
            build_rfm_summary,
            deserialize_models,
            predict,
        )
        from src.utils.config import get_project_root, load_config

        path = get_project_root() / "models_artifacts" / "probabilistic_clv.json"
        cleaned = get_project_root() / "data/interim/cleaned_transactions.parquet"
        if path.exists() and cleaned.exists():
            df_clean = _pd.read_parquet(cleaned)
            cutoff = _pd.Timestamp(
                load_config().get("features", {}).get("temporal", {}).get("cutoff_date", "2011-09-01")
            )
            summary = build_rfm_summary(df_clean, cutoff)
            row_summary = summary[summary["customer_id"] == customer_id]
            if not row_summary.empty:
                models = deserialize_models(_json.loads(path.read_text(encoding="utf-8")))
                p_alive = float(predict(models, row_summary)["p_alive"].iloc[0])
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"P(alive) skipped: {exc}")

    return {
        "customer_id": customer_id,
        "risk_tier": row._mapping["risk_tier"],
        "calibrated_probability": prob,
        "p_alive": round(p_alive, 4) if p_alive is not None else None,
        "shap": shap_drivers,
        "lime": lime_drivers,
        "narrative": narrative.strip(),
    }


@router.get("/segments")
def segments_overview(request: Request) -> List[Dict[str, Any]]:
    """Aggregate the segmentation table into per-persona summary statistics."""
    from sqlalchemy import text

    if db.get_engine() is None:
        db.ensure_engine()
    engine = db.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    sql = """
        SELECT s.persona,
               COUNT(*) AS n_customers,
               AVG(p.calibrated_probability) AS avg_churn_prob,
               AVG(p.predicted_value) AS avg_predicted_value,
               SUM(p.value_at_risk) AS total_value_at_risk
        FROM segments s
        LEFT JOIN predictions p ON p.customer_id = s.customer_id
        GROUP BY s.persona
        ORDER BY total_value_at_risk DESC
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()
    return _rows_to_dicts(rows)
