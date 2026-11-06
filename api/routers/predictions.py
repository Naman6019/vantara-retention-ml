"""Prediction endpoints: single-customer, batch CSV, metadata and health (plan D6/D8).

The served model, its features and the decision threshold all come from
``models_artifacts/model_card.json`` — there is no hard-coded algorithm or metric.
"""

import io
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from api.schemas.customer import (
    CustomerFeaturesInput,
    DriverImpact,
    ModelMetadataResponse,
    PredictionResponse,
)
from api.state import log_prediction
from src.recommendations.engine import recommend

router = APIRouter()


def _artifacts(request: Request) -> Dict[str, Any]:
    return request.app.state.artifacts


def _churn_matrix(request: Request, X: pd.DataFrame) -> pd.DataFrame:
    card = _artifacts(request)["card"]
    if card.get("churn_use_scaled", True):
        scaler = _artifacts(request)["scaler"]
        return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    return X


def _predict_frame(request: Request, X: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Shared scoring core for single and batch predictions."""
    art = _artifacts(request)
    card = art["card"]

    churn_prob = art["churn_model"].predict_proba(_churn_matrix(request, X))[:, 1]
    calibrated = art["calibrator"].predict(churn_prob) if art.get("calibrator") else churn_prob
    tiers = art["calibrator"].tiers(calibrated) if art.get("calibrator") else np.where(
        calibrated >= 0.65, "High", np.where(calibrated >= 0.40, "Moderate", "Low")
    )

    X_clv = X.copy()
    if card.get("clv_needs_aux", True):
        # The value model was trained with an extra momentum-baseline column.
        # Default it to 0 when the caller does not supply it.
        X_clv["spend_last_90d"] = (
            raw["spend_last_90d"].to_numpy() if "spend_last_90d" in raw.columns else 0.0
        )
    clv = art["clv_model"].predict(X_clv)
    if card.get("clv_log_target", False):
        clv = np.expm1(clv)
    clv = np.clip(clv, 0.0, None)

    threshold = card.get("decision_threshold", 0.5)
    out = pd.DataFrame({
        "churn_probability": np.round(calibrated, 4),
        "raw_probability": np.round(churn_prob, 4),
        "is_churn_risk": calibrated >= threshold,
        "risk_tier": tiers,
        "predicted_future_clv": np.round(clv, 2),
    })
    return out


def _shap_drivers(request: Request, X: pd.DataFrame) -> List[Dict[str, Any]]:
    art = _artifacts(request)
    feature_cols = art["card"]["feature_cols"]
    raw = X[feature_cols]
    impacts = []
    try:
        import shap

        model = art["churn_model"]
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:  # noqa: BLE001
            explainer = shap.LinearExplainer(model, raw)
        values = np.asarray(explainer.shap_values(raw))
        if values.ndim == 3:
            values = values[:, :, -1]
        row = values[0]
        for j, col in enumerate(feature_cols):
            impacts.append({
                "feature": col,
                "value": float(raw.iloc[0, j]),
                "shap_impact": round(float(row[j]), 4),
                "direction": "Increases Churn" if row[j] > 0 else "Decreases Churn",
            })
        impacts.sort(key=lambda d: abs(d["shap_impact"]), reverse=True)
    except Exception:  # noqa: BLE001
        pass
    return impacts


@router.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "Vantara Customer Behavior Prediction Platform"}


@router.get("/metadata", response_model=ModelMetadataResponse)
def get_model_metadata(request: Request) -> ModelMetadataResponse:
    """Model metadata read from the model card (no hard-coded metrics)."""
    card = _artifacts(request)["card"]
    return ModelMetadataResponse(
        platform="Vantara Retail Solutions Customer Behavior Prediction Platform",
        version=card["model_version"],
        churn_model=card["churn_model"],
        clv_model=card["clv_model"],
        segmentation_model=card.get("segmentation_model", "K-Means"),
        test_roc_auc=card.get("test_metrics", {}).get("roc_auc", 0.0),
        test_clv_r2=card.get("clv_test_metrics", {}).get("r2_score", 0.0),
        decision_threshold=card.get("decision_threshold", 0.5),
        test_recall=card.get("test_metrics", {}).get("recall", 0.0),
        features_expected=card["feature_cols"],
    )


@router.post("/predict", response_model=PredictionResponse)
def predict_single_customer(request: Request, features: CustomerFeaturesInput) -> PredictionResponse:
    """Real-time single-customer inference with calibrated risk and the model-card threshold."""
    art = _artifacts(request)
    card = art["card"]
    feature_cols = card["feature_cols"]

    row_dict = features.model_dump()
    cust_id = row_dict.pop("customer_id", None)
    # Only keep model features (the request may include auxiliary fields).
    df_row = pd.DataFrame([row_dict]).reindex(columns=feature_cols, fill_value=0.0)
    raw = df_row.copy()
    if "spend_last_90d" in row_dict:
        raw["spend_last_90d"] = row_dict["spend_last_90d"]

    t0 = time.perf_counter()
    try:
        scored = _predict_frame(request, df_row, raw)
        prob = float(scored.iloc[0]["churn_probability"])
        is_risk = bool(scored.iloc[0]["is_churn_risk"])
        risk_tier = str(scored.iloc[0]["risk_tier"])
        future_clv = float(scored.iloc[0]["predicted_future_clv"])

        impacts = _shap_drivers(request, df_row)
        top_drivers = [DriverImpact(**d) for d in impacts[:4]]

        # Segmentation persona.
        personas = art.get("personas", {})
        seg_scaler = art.get("seg_scaler")
        persona = "Unknown"
        if seg_scaler is not None and art.get("kmeans") is not None:
            seg_features = ["recency", "frequency", "monetary_total", "return_rate", "engagement_score"]
            seg_X = seg_scaler.transform(df_row[seg_features])
            cluster = int(art["kmeans"].predict(seg_X)[0])
            persona = personas.get(cluster, f"Segment {cluster}")

        rec = recommend(
            {
                "risk_tier": risk_tier,
                "segment": persona,
                "monetary_total": float(df_row.iloc[0]["monetary_total"]),
                "predicted_value": future_clv,
                "churn_probability": prob,
            },
            shap_top=impacts[:3],
            q_low=card.get("value_tier_quantiles", {}).get("low", 0.0),
            q_high=card.get("value_tier_quantiles", {}).get("high", 1e12),
        )

        pos = [d for d in top_drivers if d.shap_impact > 0]
        neg = [d for d in top_drivers if d.shap_impact < 0]
        narrative = f"Customer flagged as {risk_tier} churn risk ({prob:.1%} calibrated). "
        if pos:
            narrative += f"Primary warning sign: {pos[0].feature.replace('_', ' ')} (+{pos[0].shap_impact:.2f}). "
        if neg:
            narrative += f"Mitigating factor: {neg[0].feature.replace('_', ' ')} ({neg[0].shap_impact:.2f}). "

        response = PredictionResponse(
            customer_id=cust_id,
            churn_probability=round(prob, 4),
            is_churn_risk=is_risk,
            risk_tier=risk_tier,
            predicted_future_clv=round(future_clv, 2),
            value_at_risk=round(prob * future_clv, 2),
            predicted_segment=persona,
            recommended_action=rec["action"],
            recommendation_channel=rec["channel"],
            recommendation_reasons=rec["reasons"],
            narrative_explanation=narrative.strip(),
            top_risk_drivers=top_drivers,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e

    latency_ms = (time.perf_counter() - t0) * 1000.0
    log_prediction(cust_id, latency_ms, "/predict")
    return response


@router.post("/batch-score")
async def batch_score_csv(request: Request, file: UploadFile = File(...)) -> List[Dict[str, Any]]:
    """Score an uploaded CSV of customer features with the calibrated production model."""
    feature_cols = _artifacts(request)["card"]["feature_cols"]
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Uploaded CSV is missing required feature columns: {missing}")

    X = df[feature_cols].copy()
    scored = _predict_frame(request, X, df)
    cust_ids = df["customer_id"].tolist() if "customer_id" in df.columns else list(range(len(df)))

    records = scored.to_dict("records")
    log_prediction(None, 0.0, "/batch-score")
    return [
        {
            "customer_id": cid,
            "churn_probability": rec["churn_probability"],
            "is_churn_risk": bool(rec["is_churn_risk"]),
            "risk_tier": rec["risk_tier"],
            "predicted_future_clv": rec["predicted_future_clv"],
        }
        for cid, rec in zip(cust_ids, records, strict=False)
    ]
