"""Leakage-safe, time-based dataset splitting for stacked customer snapshots.

The old random 70/15/15 split let training and test customers come from the same
period, which does not show how the model performs on later data (plan gap G1).
Here we split by *cutoff date* instead:

===========  =========================================================
Split        Rule
===========  =========================================================
Train        every snapshot whose cutoff is strictly before the validation cutoff
Validation   the snapshot at ``features.temporal.snapshots.validation``
Test         the snapshot at ``features.temporal.snapshots.test`` (used once)
Refit        every snapshot whose cutoff is <= ``snapshots.refit_end``
===========  =========================================================

A split is only valid if every training label window ends before the validation
cutoff, and every validation label window ends before the test cutoff. With a
monthly cadence and a 90-day window this holds by construction, and it is
asserted in the test suite.
"""

from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("data_split")

# historical_clv is intentionally excluded: it duplicated monetary_total (plan D3).
FEATURE_COLS = [
    "frequency",
    "total_items_bought",
    "unique_skus_bought",
    "monetary_total",
    "recency",
    "customer_age_days",
    "monetary_avg",
    "avg_basket_size",
    "interpurchase_interval_mean",
    "interpurchase_interval_std",
    "purchase_frequency_trend",
    "return_rate",
    "cancellation_rate",
    "seasonal_concentration",
    "discount_sensitivity",
    "is_uk_customer",
    "engagement_score",
]

CHURN_TARGET = "is_churned"
CLV_TARGET = "future_clv"
ID_COL = "customer_id"
CUTOFF_COL = "cutoff_date"


def _split_cfg() -> dict:
    return load_config().get("features", {}).get("temporal", {}).get("snapshots", {})


def load_processed_feature_table(path: Optional[str] = None) -> pd.DataFrame:
    """Load the single-cutoff processed feature table (dashboard/API default)."""
    root = get_project_root()
    config = load_config()
    if path is None:
        proc_path = root / config.get("paths", {}).get("processed_dir", "data/processed") / "customer_features.parquet"
    else:
        proc_path = root / path
    if not proc_path.exists():
        raise FileNotFoundError(
            f"Processed feature table not found at: {proc_path}. Run the feature builder first."
        )
    return pd.read_parquet(proc_path)


def load_snapshots() -> pd.DataFrame:
    """Load the stacked multi-cutoff snapshot table produced by ``snapshots.py``."""
    root = get_project_root()
    config = load_config()
    snap_path = (
        root / config.get("paths", {}).get("processed_dir", "data/processed") / "customer_snapshots.parquet"
    )
    if not snap_path.exists():
        raise FileNotFoundError(
            f"Snapshot table not found at: {snap_path}. Run src/features/snapshots.py first."
        )
    return pd.read_parquet(snap_path)


def _require_cutoff(df: pd.DataFrame) -> pd.DataFrame:
    if CUTOFF_COL not in df.columns:
        raise ValueError(
            "Time-based splitting requires a 'cutoff_date' column. Build stacked "
            "snapshots with src.features.snapshots.build_snapshots() first."
        )
    df = df.copy()
    df[CUTOFF_COL] = pd.to_datetime(df[CUTOFF_COL])
    return df


def assign_roles(
    df: pd.DataFrame, cfg: Optional[dict] = None
) -> Dict[str, List[pd.Timestamp]]:
    """Map each unique cutoff to a train/val/test role (plan D1).

    Training uses cutoffs in ``[train_start, train_end]`` only. The monthly
    snapshots between ``train_end`` and ``refit_end`` exist solely for the
    production refit and are excluded from the train/val/test partitions, which
    guarantees every training label window closes before the validation cutoff.
    """
    if cfg is None:
        cfg = _split_cfg()
    train_end = pd.Timestamp(cfg.get("train_end", "2011-03-01"))
    val_cutoff = pd.Timestamp(cfg.get("validation", "2011-06-01"))
    test_cutoff = pd.Timestamp(cfg.get("test", "2011-09-01"))
    cutoffs = sorted(pd.to_datetime(df[CUTOFF_COL]).unique())

    roles: Dict[str, List[pd.Timestamp]] = {"train": [], "val": [], "test": []}
    for c in cutoffs:
        c = pd.Timestamp(c)
        if c == test_cutoff:
            roles["test"].append(c)
        elif c == val_cutoff:
            roles["val"].append(c)
        elif c <= train_end:
            roles["train"].append(c)
        # monthly cutoffs in (train_end, refit_end] are refit-only
    return roles


def _scaled(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(scaler.transform(X), columns=FEATURE_COLS, index=X.index)


def split_data(
    df: Optional[pd.DataFrame] = None,
    save_scaler: bool = True,
    cfg: Optional[dict] = None,
) -> Dict[str, Any]:
    """Build leakage-safe train/validation/test partitions from stacked snapshots.

    Returns raw and scaled feature matrices, targets, customer ids and cutoff
    dates for each partition. The scaler is fitted on the training partition only.
    """
    if df is None:
        df = load_snapshots()
    df = _require_cutoff(df)

    roles = assign_roles(df, cfg)
    if not roles["train"]:
        raise ValueError("No training snapshots found — check snapshot cutoff configuration.")
    if not roles["val"]:
        raise ValueError("No validation snapshot found — check snapshot cutoff configuration.")

    def _slice(cutoffs: List[pd.Timestamp]) -> pd.DataFrame:
        return df[df[CUTOFF_COL].isin(cutoffs)].copy()

    train_df = _slice(roles["train"])
    val_df = _slice(roles["val"])
    test_df = _slice(roles["test"]) if roles["test"] else val_df.iloc[0:0].copy()

    X_train = train_df[FEATURE_COLS].copy()
    scaler = StandardScaler().fit(X_train)

    out: Dict[str, Any] = {
        "X_train": X_train,
        "X_val": val_df[FEATURE_COLS].copy(),
        "X_test": test_df[FEATURE_COLS].copy(),
        "X_train_scaled": _scaled(scaler, X_train),
        "X_val_scaled": _scaled(scaler, val_df[FEATURE_COLS]),
        "X_test_scaled": _scaled(scaler, test_df[FEATURE_COLS]),
        "y_train": train_df[CHURN_TARGET].to_numpy(),
        "y_val": val_df[CHURN_TARGET].to_numpy(),
        "y_test": test_df[CHURN_TARGET].to_numpy(),
        "clv_train": train_df[CLV_TARGET].to_numpy(),
        "clv_val": val_df[CLV_TARGET].to_numpy(),
        "clv_test": test_df[CLV_TARGET].to_numpy(),
        "id_train": train_df[ID_COL].to_numpy(),
        "id_val": val_df[ID_COL].to_numpy(),
        "id_test": test_df[ID_COL].to_numpy(),
        "cutoff_train": train_df[CUTOFF_COL].to_numpy(),
        "cutoff_val": val_df[CUTOFF_COL].to_numpy(),
        "cutoff_test": test_df[CUTOFF_COL].to_numpy(),
        "aux_train": train_df["spend_last_90d"].to_numpy() if "spend_last_90d" in train_df else None,
        "aux_val": val_df["spend_last_90d"].to_numpy() if "spend_last_90d" in val_df else None,
        "aux_test": test_df["spend_last_90d"].to_numpy() if "spend_last_90d" in test_df else None,
        "feature_cols": FEATURE_COLS,
        "scaler": scaler,
        "roles": roles,
    }
    # Expose the raw single-cutoff test frame for batch scoring / drift (plan D6/D7).
    out["X_train_raw"] = train_df[FEATURE_COLS + ["customer_id"]].copy()
    out["X_test_raw"] = test_df[FEATURE_COLS + ["customer_id"]].copy()

    if save_scaler:
        root = get_project_root()
        config = load_config()
        art_dir = root / config.get("paths", {}).get("models_artifacts_dir", "models_artifacts")
        art_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, art_dir / "scaler.joblib")
        logger.info(f"Fitted StandardScaler persisted to: {art_dir / 'scaler.joblib'}")

    logger.info(
        f"Time split: train cutoffs {[str(pd.Timestamp(c).date()) for c in roles['train']]} "
        f"({len(train_df):,} rows), val {[str(pd.Timestamp(c).date()) for c in roles['val']]} "
        f"({len(val_df):,} rows), test {[str(pd.Timestamp(c).date()) for c in roles['test']]} "
        f"({len(test_df):,} rows)"
    )
    return out


def refit_data(df: Optional[pd.DataFrame] = None, cfg: Optional[dict] = None) -> Dict[str, Any]:
    """Return the production-refit partition (all snapshots up to ``refit_end``)."""
    if df is None:
        df = load_snapshots()
    df = _require_cutoff(df)
    if cfg is None:
        cfg = _split_cfg()
    refit_end = pd.Timestamp(cfg.get("refit_end", "2011-06-01"))
    refit_df = df[df[CUTOFF_COL] <= refit_end].copy()
    if refit_df.empty:
        raise ValueError("No snapshots at or before refit_end — check configuration.")

    X = refit_df[FEATURE_COLS].copy()
    scaler = StandardScaler().fit(X)
    return {
        "X": X,
        "X_scaled": _scaled(scaler, X),
        "y": refit_df[CHURN_TARGET].to_numpy(),
        "clv": refit_df[CLV_TARGET].to_numpy(),
        "aux": refit_df["spend_last_90d"].to_numpy() if "spend_last_90d" in refit_df else None,
        "id": refit_df[ID_COL].to_numpy(),
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
    }


if __name__ == "__main__":
    data = split_data()
    print(
        f"train={len(data['X_train']):,} val={len(data['X_val']):,} test={len(data['X_test']):,} "
        f"features={len(FEATURE_COLS)}"
    )
