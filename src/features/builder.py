"""Customer feature construction with strict point-in-time discipline.

All customer-level features are computed only from rows strictly before the
cutoff (T). Targets (churn, 90-day value) are computed only from the window
``[T, T + churn_window_days)``. A customer is included in the retention cohort
only if they purchased within ``[T - active_lookback_days, T)``.
"""

import json
from typing import Optional, Set, Tuple

import pandas as pd

from src.features import engagement
from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("feature_builder")

DEFAULT_CHURN_WINDOW_DAYS = 90
DEFAULT_ACTIVE_LOOKBACK_DAYS = 365


def _temporal_config() -> dict:
    config = load_config()
    return config.get("features", {}).get("temporal", {})


def extract_temporal_windows(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    window_days: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split transaction logs into the observation window and the label window.

    Observation rows satisfy ``invoice_date < T``; label rows satisfy
    ``T <= invoice_date < T + window_days``. The hard temporal boundary
    guarantees no target leakage.
    """
    if window_days is None:
        window_days = _temporal_config().get("churn_window_days", DEFAULT_CHURN_WINDOW_DAYS)
    window_end = cutoff_date + pd.Timedelta(days=int(window_days))

    df_obs = df[df["invoice_date"] < cutoff_date].copy()
    df_pred = df[(df["invoice_date"] >= cutoff_date) & (df["invoice_date"] < window_end)].copy()

    logger.info(
        f"Temporal split at {cutoff_date} (window {window_days}d -> {window_end.date()}): "
        f"observation rows {len(df_obs):,}, label rows {len(df_pred):,}"
    )
    return df_obs, df_pred


def active_cohort(
    df_obs: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    active_lookback_days: Optional[int] = None,
) -> Set[float]:
    """Customers with at least one positive purchase in ``[T - lookback, T)``.

    Customers who have been inactive for longer than the lookback are not a
    realistic retention target and are dropped from the cohort (plan D2).
    """
    if active_lookback_days is None:
        active_lookback_days = _temporal_config().get(
            "active_lookback_days", DEFAULT_ACTIVE_LOOKBACK_DAYS
        )
    lookback_start = cutoff_date - pd.Timedelta(days=int(active_lookback_days))
    recent = df_obs[
        (df_obs["invoice_date"] >= lookback_start)
        & (df_obs["quantity"] > 0)
        & (~df_obs["is_cancellation"])
    ]
    return set(recent["customer_id"].dropna().unique())


def realized_value(
    df_clean: pd.DataFrame, cutoff_date: pd.Timestamp, window_days: int = 90
) -> pd.DataFrame:
    """Net realized spend in ``[T, T + window_days)`` per customer (benchmarking helper)."""
    cutoff = pd.Timestamp(cutoff_date)
    end = cutoff + pd.Timedelta(days=int(window_days))
    win = df_clean[(df_clean["invoice_date"] >= cutoff) & (df_clean["invoice_date"] < end)]
    spend = win.groupby("customer_id")["line_total"].sum().clip(lower=0.0).rename("future_clv")
    return spend.reset_index()


def compute_targets(
    df_obs: pd.DataFrame,
    df_pred: pd.DataFrame,
    cohort: Optional[Set[float]] = None,
) -> pd.DataFrame:
    """Compute ``is_churned`` and ``future_clv`` for the observation cohort.

    A customer churns when they place zero valid positive-purchase invoices in
    the label window. ``future_clv`` is the net spend in that window, clipped at 0.
    """
    if cohort is None:
        active_customers = sorted(df_obs["customer_id"].dropna().unique())
    else:
        active_customers = sorted(cohort)

    pred_positive = df_pred[(df_pred["quantity"] > 0) & (~df_pred["is_cancellation"])]
    pred_spend = df_pred.groupby("customer_id")["line_total"].sum().rename("future_clv")
    pred_invoices = (
        pred_positive.groupby("customer_id")["invoice"].nunique().rename("future_invoices")
    )

    targets_df = pd.DataFrame({"customer_id": active_customers})
    targets_df = targets_df.merge(pred_spend, on="customer_id", how="left").fillna({"future_clv": 0.0})
    targets_df = targets_df.merge(pred_invoices, on="customer_id", how="left").fillna({"future_invoices": 0})

    targets_df["is_churned"] = (targets_df["future_invoices"] == 0).astype(int)
    targets_df["future_clv"] = targets_df["future_clv"].clip(lower=0.0)
    targets_df.drop(columns=["future_invoices"], inplace=True)

    logger.info(
        f"Targets: {len(targets_df):,} customers, "
        f"churn rate {targets_df['is_churned'].mean():.2%}, "
        f"mean 90-day value £{targets_df['future_clv'].mean():.2f}"
    )
    return targets_df


def compute_rfm_and_spend_features(
    df_obs: pd.DataFrame, cutoff_date: pd.Timestamp
) -> pd.DataFrame:
    """RFM and core spend metrics computed strictly before the cutoff."""
    pos_df = df_obs[(df_obs["quantity"] > 0) & (~df_obs["is_cancellation"])]

    pos_agg = pos_df.groupby("customer_id").agg(
        frequency=("invoice", "nunique"),
        last_purchase_date=("invoice_date", "max"),
        first_purchase_date=("invoice_date", "min"),
        total_items_bought=("quantity", "sum"),
        unique_skus_bought=("stock_code", "nunique"),
    )
    net_spend = df_obs.groupby("customer_id")["line_total"].sum().rename("monetary_total")
    features = pos_agg.merge(net_spend, on="customer_id", how="left")

    # Simple momentum baseline for the 90-day value model (plan D3): net spend in
    # the 90 days immediately before T. Not a model feature; carried through the
    # snapshot table as an auxiliary column.
    last90_start = cutoff_date - pd.Timedelta(days=90)
    spend_last_90d = (
        df_obs[df_obs["invoice_date"] >= last90_start]
        .groupby("customer_id")["line_total"].sum().rename("spend_last_90d")
    )
    features = features.merge(spend_last_90d, on="customer_id", how="left")
    features["spend_last_90d"] = features["spend_last_90d"].fillna(0.0).clip(lower=0.0)

    features["recency"] = (cutoff_date - features["last_purchase_date"]).dt.total_seconds() / 86400.0
    features["customer_age_days"] = (cutoff_date - features["first_purchase_date"]).dt.total_seconds() / 86400.0
    features["monetary_avg"] = features["monetary_total"] / features["frequency"].clip(lower=1)
    features["avg_basket_size"] = features["total_items_bought"] / features["frequency"].clip(lower=1)
    # NB: historical CLV was dropped (plan D3) — it duplicated monetary_total.

    features.drop(columns=["last_purchase_date", "first_purchase_date"], inplace=True)
    return features.reset_index()


def compute_behavioral_dynamics(df_obs: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Inter-purchase gap statistics and the recent-vs-prior purchase trend."""
    pos_df = df_obs[(df_obs["quantity"] > 0) & (~df_obs["is_cancellation"])].copy()

    distinct_orders = (
        pos_df.groupby(["customer_id", "invoice"])["invoice_date"]
        .min()
        .reset_index()
        .sort_values(by=["customer_id", "invoice_date"])
    )
    distinct_orders["prev_date"] = distinct_orders.groupby("customer_id")["invoice_date"].shift(1)
    distinct_orders["gap_days"] = (
        distinct_orders["invoice_date"] - distinct_orders["prev_date"]
    ).dt.total_seconds() / 86400.0

    timing_stats = (
        distinct_orders.groupby("customer_id")["gap_days"]
        .agg(interpurchase_interval_mean="mean", interpurchase_interval_std="std")
        .fillna(0.0)
        .reset_index()
    )

    date_90d_ago = cutoff_date - pd.Timedelta(days=90)
    date_180d_ago = cutoff_date - pd.Timedelta(days=180)
    orders_last_90d = (
        pos_df[pos_df["invoice_date"] >= date_90d_ago]
        .groupby("customer_id")["invoice"].nunique().rename("orders_recent_90d")
    )
    orders_prior_90d = (
        pos_df[(pos_df["invoice_date"] >= date_180d_ago) & (pos_df["invoice_date"] < date_90d_ago)]
        .groupby("customer_id")["invoice"].nunique().rename("orders_prior_90d")
    )

    trend_df = pd.DataFrame({"customer_id": sorted(df_obs["customer_id"].dropna().unique())})
    trend_df = trend_df.merge(orders_last_90d, on="customer_id", how="left").fillna(0)
    trend_df = trend_df.merge(orders_prior_90d, on="customer_id", how="left").fillna(0)
    trend_df["purchase_frequency_trend"] = (
        (trend_df["orders_recent_90d"] - trend_df["orders_prior_90d"])
        / (trend_df["orders_recent_90d"] + trend_df["orders_prior_90d"] + 1.0)
    )
    trend_df.drop(columns=["orders_recent_90d", "orders_prior_90d"], inplace=True)

    dynamics = timing_stats.merge(trend_df, on="customer_id", how="right").fillna(0.0)
    return dynamics


def compute_risk_and_preference_features(df_obs: pd.DataFrame) -> pd.DataFrame:
    """Return rate, cancellation rate, seasonal concentration and discount sensitivity."""
    cust_ids = sorted(df_obs["customer_id"].dropna().unique())
    base_df = pd.DataFrame({"customer_id": cust_ids})

    return_items = (
        df_obs[df_obs["quantity"] < 0].groupby("customer_id")["quantity"].sum().abs().rename("returned_qty")
    )
    bought_items = (
        df_obs[df_obs["quantity"] > 0].groupby("customer_id")["quantity"].sum().rename("bought_qty")
    )
    total_invs = df_obs.groupby("customer_id")["invoice"].nunique().rename("total_invoices")
    cancel_invs = (
        df_obs[df_obs["is_cancellation"]].groupby("customer_id")["invoice"].nunique().rename("cancel_invoices")
    )
    q4_spend = (
        df_obs[df_obs["invoice_date"].dt.month.isin([10, 11, 12])]
        .groupby("customer_id")["line_total"].sum().rename("q4_spend")
    )
    total_spend = df_obs.groupby("customer_id")["line_total"].sum().rename("total_spend")

    median_price = df_obs[df_obs["quantity"] > 0]["price"].median()
    low_price_items = (
        df_obs[(df_obs["quantity"] > 0) & (df_obs["price"] < median_price)]
        .groupby("customer_id")["quantity"].sum().rename("discounted_qty")
    )
    primary_country = (
        df_obs.groupby("customer_id")["country"]
        .agg(lambda x: x.mode()[0] if not x.empty else "United Kingdom")
        .rename("primary_country")
    )

    df_feats = (
        base_df.merge(return_items, on="customer_id", how="left").fillna(0)
        .merge(bought_items, on="customer_id", how="left").fillna(0)
        .merge(total_invs, on="customer_id", how="left").fillna(1)
        .merge(cancel_invs, on="customer_id", how="left").fillna(0)
        .merge(q4_spend, on="customer_id", how="left").fillna(0)
        .merge(total_spend, on="customer_id", how="left").fillna(0)
        .merge(low_price_items, on="customer_id", how="left").fillna(0)
        .merge(primary_country, on="customer_id", how="left")
    )

    df_feats["return_rate"] = df_feats["returned_qty"] / (df_feats["bought_qty"] + df_feats["returned_qty"] + 1e-5)
    df_feats["cancellation_rate"] = df_feats["cancel_invoices"] / df_feats["total_invoices"].clip(lower=1)
    df_feats["seasonal_concentration"] = (df_feats["q4_spend"] / df_feats["total_spend"].clip(lower=1.0)).clip(0.0, 1.0)
    df_feats["discount_sensitivity"] = (df_feats["discounted_qty"] / (df_feats["bought_qty"] + 1e-5)).clip(0.0, 1.0)
    df_feats["is_uk_customer"] = (df_feats["primary_country"] == "United Kingdom").astype(int)

    df_feats.drop(
        columns=[
            "returned_qty", "bought_qty", "total_invoices", "cancel_invoices",
            "q4_spend", "total_spend", "discounted_qty", "primary_country",
        ],
        inplace=True,
    )
    return df_feats


def compute_engagement_score(
    df_features: pd.DataFrame, ref: Optional[dict] = None
) -> pd.Series:
    """Composite 0-100 engagement score (delegates to ``src.features.engagement``).

    Kept for backwards compatibility; when ``ref`` is None the breakpoints are
    fitted on ``df_features`` itself.
    """
    return engagement.score(df_features, ref)


def build_single_snapshot(
    df_clean: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    churn_window_days: Optional[int] = None,
    active_lookback_days: Optional[int] = None,
    engagement_ref: Optional[dict] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Build the leak-free customer table for one cutoff.

    Returns ``(customers_df, engagement_ref)`` where the reference is either the
    supplied one or newly fitted on this snapshot's observation rows.
    """
    if churn_window_days is None:
        churn_window_days = _temporal_config().get("churn_window_days", DEFAULT_CHURN_WINDOW_DAYS)
    if active_lookback_days is None:
        active_lookback_days = _temporal_config().get("active_lookback_days", DEFAULT_ACTIVE_LOOKBACK_DAYS)

    df_obs, df_pred = extract_temporal_windows(df_clean, cutoff_date, churn_window_days)

    rfm_df = compute_rfm_and_spend_features(df_obs, cutoff_date)
    dynamics_df = compute_behavioral_dynamics(df_obs, cutoff_date)
    risk_df = compute_risk_and_preference_features(df_obs)

    customer_df = rfm_df.merge(dynamics_df, on="customer_id", how="inner")
    customer_df = customer_df.merge(risk_df, on="customer_id", how="inner")

    if engagement_ref is None:
        engagement_ref = engagement.fit_reference(customer_df)
    customer_df["engagement_score"] = engagement.score(customer_df, engagement_ref)

    cohort = active_cohort(df_obs, cutoff_date, active_lookback_days)
    customer_df = customer_df[customer_df["customer_id"].isin(cohort)].copy()

    targets_df = compute_targets(df_obs, df_pred, cohort=cohort)
    final_df = customer_df.merge(targets_df, on="customer_id", how="inner")

    final_df.fillna(0.0, inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    return final_df, engagement_ref


def build_customer_feature_table(
    df_cleaned: pd.DataFrame,
    cutoff_date_str: Optional[str] = None,
    save_processed: bool = True,
    churn_window_days: Optional[int] = None,
    active_lookback_days: Optional[int] = None,
) -> pd.DataFrame:
    """Build the single-cutoff customer feature table and optionally persist it."""
    config = load_config()
    if cutoff_date_str is None:
        cutoff_date_str = _temporal_config().get("cutoff_date", "2011-09-01")
    cutoff_date = pd.Timestamp(cutoff_date_str)
    logger.info(f"Building customer feature table at cutoff {cutoff_date}")

    final_df, engagement_ref = build_single_snapshot(
        df_cleaned,
        cutoff_date,
        churn_window_days=churn_window_days,
        active_lookback_days=active_lookback_days,
    )

    logger.info(
        f"Feature table built: {len(final_df):,} customers, {final_df.shape[1]} columns. "
        f"Cohort churn rate {final_df['is_churned'].mean():.2%}"
    )

    if save_processed:
        root = get_project_root()
        proc_dir = root / config.get("paths", {}).get("processed_dir", "data/processed")
        proc_dir.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(proc_dir / "customer_features.parquet", index=False)

        art_dir = root / config.get("paths", {}).get("models_artifacts_dir", "models_artifacts")
        art_dir.mkdir(parents=True, exist_ok=True)
        with open(art_dir / "engagement_reference.json", "w", encoding="utf-8") as f:
            json.dump(engagement_ref, f)
        logger.info("Saved processed features and engagement reference artifact.")

    return final_df


if __name__ == "__main__":
    from src.data.cleaner import run_cleaning_pipeline
    from src.data.loader import load_raw_data

    root = get_project_root()
    interim_path = root / "data/interim/cleaned_transactions.parquet"
    if interim_path.exists():
        logger.info("Loading cleaned transactions from interim cache...")
        df_cleaned = pd.read_parquet(interim_path)
    else:
        df_raw = load_raw_data()
        df_cleaned = run_cleaning_pipeline(df_raw)

    features_df = build_customer_feature_table(df_cleaned)
    print(f"\nCustomers: {len(features_df):,}")
    print(f"Churn rate: {features_df['is_churned'].mean():.2%}")
    print(f"Mean 90-day value: £{features_df['future_clv'].mean():.2f}")
