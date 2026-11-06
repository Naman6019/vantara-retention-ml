"""Probabilistic CLV benchmark: BG/NBD + Gamma-Gamma (Beyond PRD, plan P1).

This is the textbook "buy-till-you-die" baseline that customer-value teams use.
It predicts, on the same training cutoffs as the supervised model:

  * how many orders a customer will place in the next 90 days (BG/NBD);
  * their average spend per order (Gamma-Gamma);
  * the probability the customer is still "alive" (BG/NBD).

The 90-day value is the product of the two, and it is benchmarked against the
LightGBM value model on the validation and test snapshots.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

from src.utils.logger import get_logger

logger = get_logger("probabilistic_clv")

PENALIZER = 0.01


def build_rfm_summary(df_clean: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Build the BG/NBD inputs from the observation window (strictly before T).

    ``frequency`` counts *repeat* orders (orders - 1), ``recency`` is the age at
    the last order, ``T`` the age at the cutoff, all in days.
    """
    cutoff = pd.Timestamp(cutoff_date)
    obs = df_clean[(df_clean["invoice_date"] < cutoff) & (df_clean["quantity"] > 0) & (~df_clean["is_cancellation"])]

    orders = (
        obs.groupby(["customer_id", "invoice"])
        .agg(order_date=("invoice_date", "min"), spend=("line_total", "sum"))
        .reset_index()
    )
    agg = orders.groupby("customer_id").agg(
        first_date=("order_date", "min"),
        last_date=("order_date", "max"),
        n_orders=("invoice", "nunique"),
        avg_spend=("spend", "mean"),
    )

    summary = pd.DataFrame({
        "customer_id": agg.index,
        "frequency": (agg["n_orders"] - 1).clip(lower=0).to_numpy(),
        "recency": (agg["last_date"] - agg["first_date"]).dt.total_seconds().to_numpy() / 86400.0,
        "T": (cutoff - agg["first_date"]).dt.total_seconds().to_numpy() / 86400.0,
        "monetary_value": agg["avg_spend"].to_numpy(),
    })
    # BG/NBD needs T > recency > 0 for the likelihood to be well defined.
    summary = summary[(summary["T"] > 0)].copy()
    summary["recency"] = summary["recency"].clip(lower=0.0, upper=summary["T"] - 1e-6)
    return summary.reset_index(drop=True)


def fit_models(summary: pd.DataFrame) -> Dict[str, Any]:
    """Fit BG/NBD and (on repeat buyers) Gamma-Gamma."""
    bgf = BetaGeoFitter(penalizer_coef=PENALIZER)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    repeat = summary[summary["frequency"] > 0].copy()
    ggf = GammaGammaFitter(penalizer_coef=PENALIZER)
    if len(repeat) >= 10:
        ggf.fit(repeat["frequency"], repeat["monetary_value"])
    else:  # not enough repeat buyers: fall back to a simple mean-spend model
        ggf = None
    return {"bgf": bgf, "ggf": ggf, "mean_spend": float(summary["monetary_value"].mean())}


def predict(models: Dict[str, Any], summary: pd.DataFrame, horizon_days: int = 90) -> pd.DataFrame:
    """Predict expected purchases, average spend, P(alive) and 90-day value."""
    bgf = models["bgf"]
    ggf = models["ggf"]

    exp_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
        horizon_days, summary["frequency"], summary["recency"], summary["T"]
    )
    p_alive = bgf.conditional_probability_alive(summary["frequency"], summary["recency"], summary["T"])

    if ggf is not None:
        avg_spend = ggf.conditional_expected_average_profit(summary["frequency"], summary["monetary_value"])
    else:
        avg_spend = np.full(len(summary), models["mean_spend"])

    value = np.clip(np.asarray(exp_purchases) * np.asarray(avg_spend), 0.0, None)
    return pd.DataFrame({
        "customer_id": summary["customer_id"].to_numpy(),
        "expected_purchases_90d": np.asarray(exp_purchases),
        "expected_avg_spend": np.asarray(avg_spend),
        "p_alive": np.asarray(p_alive),
        "probabilistic_value": value,
    })


def serialize_models(models: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-serialisable form of the fitted BG/NBD + Gamma-Gamma models.

    The lifetimes fitters cannot be pickled (they hold a closure), so we persist
    their fitted parameters and rebuild the fitters on load.
    """
    out: Dict[str, Any] = {"mean_spend": float(models["mean_spend"])}
    out["bgf"] = {k: float(v) for k, v in models["bgf"].params_.to_dict().items()}
    out["ggf"] = (
        {k: float(v) for k, v in models["ggf"].params_.to_dict().items()}
        if models.get("ggf") is not None
        else None
    )
    return out


def deserialize_models(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the fitted models from ``serialize_models`` output."""
    bgf = BetaGeoFitter(penalizer_coef=PENALIZER)
    bgf.params_ = pd.Series(payload["bgf"])
    ggf = None
    if payload.get("ggf"):
        ggf = GammaGammaFitter(penalizer_coef=PENALIZER)
        ggf.params_ = pd.Series(payload["ggf"])
    return {"bgf": bgf, "ggf": ggf, "mean_spend": float(payload["mean_spend"])}


def benchmark_vs_supervised(
    df_clean: pd.DataFrame,
    validation_cutoff: str = "2011-06-01",
    horizon_days: int = 90,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compare the probabilistic value against the actual outcome on the val snapshot."""
    from src.features.builder import realized_value
    from src.models.clv_regressor import top_decile_value_capture

    val_cutoff = pd.Timestamp(validation_cutoff)
    summary = build_rfm_summary(df_clean, val_cutoff)
    models = fit_models(summary)
    preds = predict(models, summary, horizon_days)

    actual = realized_value(df_clean, val_cutoff, horizon_days)
    merged = preds.merge(actual, on="customer_id", how="inner")
    mae = float(np.mean(np.abs(merged["probabilistic_value"] - merged["future_clv"])))
    capture = top_decile_value_capture(merged["future_clv"].to_numpy(), merged["probabilistic_value"].to_numpy())

    rows = pd.DataFrame([{
        "Model": "BG/NBD + Gamma-Gamma",
        "Customers": len(merged),
        "MAE (£)": round(mae, 2),
        "Top-Decile Value Capture": round(capture, 4),
    }])
    logger.info(f"Probabilistic CLV (validation): MAE £{mae:.2f}, top-decile capture {capture:.3f}")
    return rows, models


if __name__ == "__main__":
    import pandas as pd

    from src.utils.config import get_project_root

    root = get_project_root()
    interim = root / "data/interim/cleaned_transactions.parquet"
    df_clean = pd.read_parquet(interim)
    rows, _ = benchmark_vs_supervised(df_clean)
    print(rows.to_string(index=False))
