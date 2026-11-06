"""Stacked monthly snapshot construction for leakage-safe, out-of-time evaluation.

For each cutoff ``T`` we build the leak-free customer table (features strictly
before ``T``, labels from ``[T, T + window)``). Stacking several monthly cutoffs
gives the model roughly an order of magnitude more training rows while keeping
every label window strictly earlier than the next split's cutoff (plan D1).
"""

from typing import List, Optional, Tuple

import pandas as pd

from src.features import engagement
from src.features.builder import build_single_snapshot
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("snapshots")


def _snapshot_cfg() -> dict:
    config = load_config()
    return config.get("features", {}).get("temporal", {})


def generate_cutoffs(cfg: Optional[dict] = None) -> List[pd.Timestamp]:
    """Return every needed cutoff: monthly refit range plus validation and test.

    Training cutoffs are the subset from ``train_start`` to ``train_end``; the
    extra monthly cutoffs up to ``refit_end`` are used only by the production
    refit, and the validation/test snapshots sit outside the monthly range.
    """
    if cfg is None:
        cfg = _snapshot_cfg().get("snapshots", {})
    start = pd.Timestamp(cfg.get("train_start", "2010-06-01"))
    refit_end = pd.Timestamp(cfg.get("refit_end", "2011-06-01"))
    freq = cfg.get("frequency", "MS")
    cutoffs = list(pd.date_range(start=start, end=refit_end, freq=freq))
    for extra in (cfg.get("validation", "2011-06-01"), cfg.get("test", "2011-09-01")):
        extra = pd.Timestamp(extra)
        if extra not in cutoffs:
            cutoffs.append(extra)
    return sorted(cutoffs)


def build_snapshots(
    df_clean: pd.DataFrame,
    cutoffs: Optional[List[pd.Timestamp]] = None,
    engagement_ref: Optional[dict] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Stack one leak-free customer table per cutoff, tagged with ``cutoff_date``.

    Returns ``(snapshots_df, engagement_ref)``. The engagement reference is fitted
    once on the *earliest* snapshot and reused for all of them, so the score is
    consistent and reproducible (plan D10).
    """
    temporal = _snapshot_cfg()
    window_days = temporal.get("churn_window_days", 90)
    lookback_days = temporal.get("active_lookback_days", 365)

    if cutoffs is None:
        cutoffs = generate_cutoffs()

    frames = []
    for i, cutoff in enumerate(cutoffs):
        snap, engagement_ref = build_single_snapshot(
            df_clean,
            pd.Timestamp(cutoff),
            churn_window_days=window_days,
            active_lookback_days=lookback_days,
            engagement_ref=engagement_ref,
        )
        snap = snap.copy()
        snap["cutoff_date"] = pd.Timestamp(cutoff)
        frames.append(snap)
        logger.info(
            f"Snapshot {i + 1}/{len(cutoffs)} @ {pd.Timestamp(cutoff).date()}: "
            f"{len(snap):,} customers, churn {snap['is_churned'].mean():.2%}"
        )

    stacked = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Stacked {len(cutoffs)} snapshots -> {len(stacked):,} rows, "
        f"{stacked['customer_id'].nunique():,} unique customers."
    )
    return stacked, engagement_ref


def fit_engagement_reference(early_window_df: pd.DataFrame) -> dict:
    """Convenience wrapper to fit the engagement reference on a raw window."""
    return engagement.fit_reference(early_window_df)


def save_snapshots(
    snaps: pd.DataFrame,
    engagement_ref: dict,
    path: Optional[str] = None,
    root=None,
) -> str:
    """Persist the stacked snapshot table and the engagement reference artifact."""
    import json

    from src.utils.config import get_project_root, load_config

    root = root or get_project_root()
    config = load_config()
    proc_dir = root / config.get("paths", {}).get("processed_dir", "data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_path = root / path if path else proc_dir / "customer_snapshots.parquet"
    snaps.to_parquet(out_path, index=False)

    art_dir = root / config.get("paths", {}).get("models_artifacts_dir", "models_artifacts")
    art_dir.mkdir(parents=True, exist_ok=True)
    with open(art_dir / "engagement_reference.json", "w", encoding="utf-8") as f:
        json.dump(engagement_ref, f)

    logger.info(f"Saved {len(snaps):,} snapshot rows to {out_path}")
    return str(out_path)


if __name__ == "__main__":
    from src.data.cleaner import run_cleaning_pipeline
    from src.data.loader import load_raw_data
    from src.utils.config import get_project_root

    root = get_project_root()
    interim = root / "data/interim/cleaned_transactions.parquet"
    df_clean = pd.read_parquet(interim) if interim.exists() else run_cleaning_pipeline(load_raw_data())

    snaps, ref = build_snapshots(df_clean)
    save_snapshots(snaps, ref)
    print(f"Snapshots: {len(snaps):,} rows")
    print(snaps.groupby("cutoff_date")["is_churned"].agg(["size", "mean"]))
