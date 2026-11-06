"""Single-command pipeline orchestrator (plan D12 / PRD acceptance criterion 1).

    python -m src.pipeline all          # ingest -> clean -> validate -> snapshots
                                         #   -> train -> segment -> explain -> recommender -> score -> drift
    python -m src.pipeline features     # data + features only
    python -m src.pipeline train        # snapshots + models only

Each stage is importable and idempotent so it can also be driven by DVC.
"""

import argparse
import sys
import time
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger("pipeline")


def stage_ingest():
    from src.data.loader import load_raw_data

    logger.info("[ingest] downloading/loading raw transactions")
    return load_raw_data()


def stage_clean(df_raw):
    from src.data.cleaner import run_cleaning_pipeline

    logger.info("[clean] cleaning transactions")
    return run_cleaning_pipeline(df_raw)


def stage_validate(df_clean):
    from src.data.contracts import validate_raw

    logger.info("[validate] checking data contracts")
    try:
        validate_raw(df_clean)
    except ValueError as exc:
        logger.warning(f"[validate] raw contract warning (continuing): {exc}")
    return df_clean


def stage_features(df_clean):
    from src.features.builder import build_customer_feature_table

    logger.info("[features] building single-cutoff feature table")
    return build_customer_feature_table(df_clean)


def stage_snapshots(df_clean):
    from src.features.snapshots import build_snapshots, save_snapshots

    logger.info("[snapshots] building stacked monthly snapshots")
    snaps, ref = build_snapshots(df_clean)
    save_snapshots(snaps, ref)
    return snaps


def stage_train():
    from src.models.train import train_all

    logger.info("[train] tuning, selecting, calibrating, refitting, logging to MLflow")
    return train_all()


def stage_segment():
    from src.segmentation.clusterer import run_customer_segmentation

    logger.info("[segment] K-Means + GMM with business personas")
    return run_customer_segmentation()


def stage_explain():
    from src.explainability.explainer import run_explainability_suite

    logger.info("[explain] SHAP / LIME / PDP suite")
    return run_explainability_suite()


def stage_recommender():
    from src.recommendations.products import run_product_recommender

    logger.info("[recommender] item-item recommender, evaluated out-of-time")
    return run_product_recommender()


def stage_batch_score():
    from src.serving.batch_score import run_batch_scoring

    logger.info("[score] batch scoring into PostgreSQL/SQLite")
    return run_batch_scoring()


def run_all() -> None:
    t0 = time.time()
    df_raw = stage_ingest()
    df_clean = stage_clean(df_raw)
    stage_validate(df_clean)
    stage_features(df_clean)
    stage_snapshots(df_clean)
    stage_train()
    stage_segment()
    try:
        stage_explain()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[explain] skipped: {exc}")
    stage_recommender()
    stage_batch_score()
    logger.info(f"Pipeline complete in {time.time() - t0:.1f}s")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Vantara pipeline orchestrator")
    parser.add_argument(
        "stage",
        choices=["all", "ingest", "features", "snapshots", "train", "segment", "explain", "recommender", "score"],
        nargs="?",
        default="all",
    )
    args = parser.parse_args(argv)

    if args.stage == "all":
        run_all()
    elif args.stage == "ingest":
        stage_ingest()
    elif args.stage == "features":
        stage_features(stage_clean(stage_ingest()))
    elif args.stage == "snapshots":
        stage_snapshots(stage_clean(stage_ingest()))
    elif args.stage == "train":
        stage_train()
    elif args.stage == "segment":
        stage_segment()
    elif args.stage == "explain":
        stage_explain()
    elif args.stage == "recommender":
        stage_recommender()
    elif args.stage == "score":
        stage_batch_score()
    return 0


if __name__ == "__main__":
    sys.exit(main())
