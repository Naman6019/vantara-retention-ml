"""Batch scoring: features as of "today" -> scores, segments, SHAP, recommendations -> DB.

Plan D6. The latest single-cutoff snapshot (cutoff = today) is scored with the
production-refit models named in the model card, and everything is written to
PostgreSQL (or SQLite in tests).
"""

import time
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.models.data_split import FEATURE_COLS
from src.models.model_card import read_model_card
from src.recommendations.engine import recommend
from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("batch_score")

SEG_FEATURES = ["recency", "frequency", "monetary_total", "return_rate", "engagement_score"]


def load_artifacts(art_dir=None) -> Dict[str, Any]:
    """Load the production models, scaler, calibrator and segmentation artifacts."""
    root = get_project_root()
    art_dir = art_dir or (root / "models_artifacts")
    card = read_model_card(art_dir / "model_card.json")
    artifacts: Dict[str, Any] = {
        "card": card,
        "churn_model": joblib.load(art_dir / "churn_best_model.joblib"),
        "clv_model": joblib.load(art_dir / "clv_best_model.joblib"),
        "scaler": joblib.load(art_dir / "scaler.joblib"),
        "kmeans": joblib.load(art_dir / "kmeans_model.joblib"),
        "seg_scaler": joblib.load(art_dir / "segmentation_scaler.joblib"),
        "personas": joblib.load(art_dir / "segment_personas.joblib"),
    }
    calib_path = art_dir / "churn_calibrator.joblib"
    if calib_path.exists():
        from src.models.calibration import ProbabilityCalibrator

        artifacts["calibrator"] = ProbabilityCalibrator.load(calib_path)
    try:
        artifacts["autoencoder"] = _load_autoencoder(art_dir)
    except Exception:  # noqa: BLE001 - anomaly detection is best-effort
        artifacts["autoencoder"] = None
    return artifacts


def _load_autoencoder(art_dir):
    import torch

    from src.models.dl_models import SpendingAutoencoder

    model = SpendingAutoencoder(input_dim=len(FEATURE_COLS))
    model.load_state_dict(torch.load(art_dir / "autoencoder_model.pt", map_location="cpu"))
    model.eval()
    threshold = joblib.load(art_dir / "autoencoder_threshold.joblib")
    return {"model": model, "threshold": threshold}


def score_customers(
    df: pd.DataFrame,
    artifacts: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Score a customer feature frame and return a results dataframe.

    ``df`` must contain ``customer_id`` plus all ``FEATURE_COLS`` and a
    ``spend_last_90d`` column (the value-model baseline).
    """
    if artifacts is None:
        artifacts = load_artifacts()
    card = artifacts["card"]
    churn_model = artifacts["churn_model"]
    clv_model = artifacts["clv_model"]
    scaler = artifacts["scaler"]

    X = df[FEATURE_COLS].copy()
    df = df.reset_index(drop=True)

    # ---- churn probability (raw + calibrated) ----
    X_churn = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLS) if card.get("churn_use_scaled", True) else X
    raw_prob = churn_model.predict_proba(X_churn)[:, 1]
    if artifacts.get("calibrator") is not None:
        calibrated = artifacts["calibrator"].predict(raw_prob)
        tiers = artifacts["calibrator"].tiers(calibrated)
    else:
        calibrated = raw_prob
        tiers = np.where(calibrated >= 0.65, "High", np.where(calibrated >= 0.40, "Moderate", "Low"))

    # ---- 90-day value ----
    X_clv = X.copy()
    if card.get("clv_needs_aux", True) and "spend_last_90d" in df.columns:
        X_clv["spend_last_90d"] = df["spend_last_90d"].to_numpy()
    raw_clv = clv_model.predict(X_clv)
    if card.get("clv_log_target", False):
        raw_clv = np.expm1(raw_clv)
    predicted_value = np.clip(raw_clv, 0.0, None)

    # ---- segmentation ----
    seg_X = artifacts["seg_scaler"].transform(X[SEG_FEATURES])
    clusters = artifacts["kmeans"].predict(seg_X)
    personas = [artifacts["personas"].get(int(c), f"Segment {c}") for c in clusters]

    # ---- anomaly detection (best effort) ----
    anomaly = np.zeros(len(df))
    if artifacts.get("autoencoder") is not None:
        import torch

        ae = artifacts["autoencoder"]
        # The autoencoder was trained on the full scaled feature vector, not the
        # 5-dim segmentation matrix.
        ae_input = scaler.transform(X) if card.get("churn_use_scaled", True) else X.to_numpy()
        ae_tensor = torch.tensor(np.asarray(ae_input, dtype=np.float32))
        with torch.no_grad():
            recon = ae["model"](ae_tensor)
            err = torch.mean((ae_tensor - recon) ** 2, dim=1).numpy()
        anomaly = (err > ae["threshold"]).astype(int)

    threshold = card.get("decision_threshold", 0.5)
    value_q = card.get("value_tier_quantiles", {"low": 0.0, "high": 1e12})

    results = pd.DataFrame({
        "customer_id": df["customer_id"].to_numpy(),
        "churn_probability": np.round(raw_prob, 4),
        "calibrated_probability": np.round(calibrated, 4),
        "churn_prediction": (calibrated >= threshold).astype(int),
        "risk_tier": tiers,
        "predicted_value": np.round(predicted_value, 2),
        "value_at_risk": np.round(calibrated * predicted_value, 2),
        "persona": personas,
        "kmeans_cluster": clusters.astype(int),
        "anomaly_flag": anomaly.astype(int),
    })

    # ---- recommendations (SHAP top drivers per customer) ----
    shap_top = _shap_top_drivers(artifacts, X, top_k=3)
    results["shap_top"] = [shap_top.get(i, []) for i in range(len(results))]
    recs = []
    for i, row in results.iterrows():
        value = float(row["predicted_value"])
        tier = "high" if value >= value_q["high"] else ("low" if value < value_q["low"] else "mid")
        rec = recommend(
            {
                "risk_tier": row["risk_tier"],
                "value_tier": tier,
                "segment": row["persona"],
                "monetary_total": float(X.loc[i, "monetary_total"]),
                "predicted_value": value,
                "churn_probability": float(row["calibrated_probability"]),
            },
            shap_top=shap_top.get(i, []),
            card=card,
        )
        recs.append(rec)
    results["action"] = [r["action"] for r in recs]
    results["channel"] = [r["channel"] for r in recs]
    results["rule_id"] = [r["rule_id"] for r in recs]
    results["reasons"] = [r["reasons"] for r in recs]
    results["priority_score"] = [r["priority_score"] for r in recs]
    results["value_tier"] = [
        "high" if v >= value_q["high"] else ("low" if v < value_q["low"] else "mid")
        for v in results["predicted_value"]
    ]
    results["model_version"] = card["model_version"]
    results["primary_country"] = np.where(
        X["is_uk_customer"].to_numpy() >= 0.5, "United Kingdom", "International"
    )
    # Persist each customer's feature vector so the API can serve SHAP/LIME later.
    results["_features"] = [
        {c: (None if pd.isna(X.loc[i, c]) else float(X.loc[i, c])) for c in FEATURE_COLS}
        for i in range(len(results))
    ]
    results.attrs["feature_cols"] = FEATURE_COLS
    return results


def _shap_top_drivers(
    artifacts: Dict[str, Any], X_raw: pd.DataFrame, top_k: int = 3
) -> Dict[int, List[Dict[str, Any]]]:
    """Top-k SHAP drivers per customer. Falls back to zeros if SHAP is unavailable."""
    try:
        import shap

        model = artifacts["churn_model"]
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:  # noqa: BLE001 - not a tree model
            explainer = shap.LinearExplainer(model, X_raw)
        values = np.asarray(explainer.shap_values(X_raw))
        if values.ndim == 3:
            values = values[:, :, -1]
        out: Dict[int, List[Dict[str, Any]]] = {}
        for i in range(len(X_raw)):
            row = values[i]
            order = np.argsort(np.abs(row))[::-1][:top_k]
            out[i] = [
                {
                    "feature": FEATURE_COLS[j],
                    "shap_impact": round(float(row[j]), 4),
                    "direction": "Increases Churn" if row[j] > 0 else "Decreases Churn",
                }
                for j in order
            ]
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"SHAP computation skipped: {exc}")
        return {}


def persist_results(results: pd.DataFrame, session) -> int:
    """Upsert scored customers into the relational tables. Returns rows written."""
    from src.serving.models import (
        Customer,
        ModelRegistry,
        Prediction,
        Recommendation,
        Segment,
    )

    written = 0
    for _, row in results.iterrows():
        cid = float(row["customer_id"])
        if session.get(Customer, cid) is None:
            session.add(Customer(customer_id=cid, primary_country=str(row.get("primary_country", "United Kingdom"))))

        feature_vec = row.get("_features")
        if feature_vec is None or (isinstance(feature_vec, float) and np.isnan(feature_vec)):
            feature_vec = None

        pred = Prediction(
            customer_id=cid,
            model_version=row["model_version"],
            churn_probability=float(row["churn_probability"]),
            calibrated_probability=float(row["calibrated_probability"]),
            churn_prediction=int(row["churn_prediction"]),
            risk_tier=row["risk_tier"],
            predicted_value=float(row["predicted_value"]),
            value_at_risk=float(row["value_at_risk"]),
            value_tier=row.get("value_tier", "mid"),
            anomaly_flag=int(row["anomaly_flag"]),
            shap_top=list(row.get("shap_top", [])),
            features=feature_vec,
        )
        session.add(pred)
        session.add(Segment(
            customer_id=cid,
            kmeans_cluster=int(row["kmeans_cluster"]),
            gmm_cluster=int(row["kmeans_cluster"]),
            persona=row["persona"],
        ))
        session.add(Recommendation(
            customer_id=cid,
            action=row["action"],
            channel=row["channel"],
            rule_id=row["rule_id"],
            reasons=list(row["reasons"]),
            priority_score=float(row["priority_score"]),
        ))
        written += 1

    session.flush()
    # Register the model version once.
    card = read_model_card()
    if session.query(ModelRegistry).filter_by(model_version=card["model_version"]).first() is None:
        session.add(ModelRegistry(
            model_version=card["model_version"],
            churn_model=card["churn_model"],
            clv_model=card["clv_model"],
            decision_threshold=float(card["decision_threshold"]),
            metrics=card.get("test_metrics"),
        ))
    return written


def run_batch_scoring(score_date: Optional[str] = None) -> pd.DataFrame:
    """End-to-end batch job: build today's snapshot, score it, persist to the DB."""
    from src.data.cleaner import run_cleaning_pipeline
    from src.data.loader import load_raw_data
    from src.features.builder import build_customer_feature_table
    from src.serving import db

    config = load_config()
    root = get_project_root()
    interim = root / "data/interim/cleaned_transactions.parquet"
    df_clean = pd.read_parquet(interim) if interim.exists() else run_cleaning_pipeline(load_raw_data())

    score_date = score_date or config.get("features", {}).get("temporal", {}).get("live_scoring_date", "2011-12-09")
    features = build_customer_feature_table(df_clean, cutoff_date_str=score_date, save_processed=False)
    logger.info(f"Scoring {len(features):,} customers as of {score_date}")

    results = score_customers(features)
    db.ensure_engine()
    with db.get_session() as session:
        written = persist_results(results, session)
        _persist_product_history(session, df_clean)

    # Drift report (plan D7): PSI per feature against the training reference.
    from src.monitoring.drift import compute_drift, persist_drift

    try:
        drift_rows = compute_drift(features)
        persist_drift(drift_rows, batch_date=str(score_date))
        drifted = [r["feature"] for r in drift_rows if r["drifted"]]
        logger.info(f"Drift report written ({len(drift_rows)} features, {len(drifted)} drifted): {drifted}")
    except FileNotFoundError:
        logger.warning("Reference distribution not found; drift report skipped.")

    logger.info(f"Persisted {written:,} scored customers.")
    return results


def _persist_product_history(session, df_clean: pd.DataFrame) -> None:
    """Store each customer's product purchases (popularity recommender baseline)."""
    from src.serving.models import CustomerProduct

    try:
        session.query(CustomerProduct).delete()
        prod = (
            df_clean[["customer_id", "stock_code", "quantity"]]
            .dropna(subset=["customer_id"])
            .groupby(["customer_id", "stock_code"], as_index=False)["quantity"].sum()
        )
        session.add_all([
            CustomerProduct(
                customer_id=float(r.customer_id),
                product_id=str(r.stock_code),
                quantity=float(r.quantity),
            )
            for r in prod.itertuples()
        ])
        logger.info(f"Persisted {len(prod):,} customer-product rows.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Product history persistence skipped: {exc}")


if __name__ == "__main__":
    t0 = time.time()
    out = run_batch_scoring()
    print(f"Scored {len(out):,} customers in {time.time() - t0:.1f}s")
    print(out[["customer_id", "calibrated_probability", "predicted_value", "value_at_risk", "risk_tier", "action"]].head())
