"""Model card: the single source of truth for the serving layer (plan D8).

The API, the batch scorer and the report all read from
``models_artifacts/model_card.json``. The API loads whatever model the card
names, so there is no hard-coded algorithm, and the metrics shown in
``/metadata`` are the ones actually recorded at training time.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("model_card")

CARD_FILENAME = "model_card.json"


def _git_sha() -> str:
    import subprocess

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(get_project_root()),
            stderr=subprocess.DEVNULL,
        )
        return sha.decode().strip()
    except Exception:
        return "unknown"


def build_model_card(
    *,
    churn_model_name: str,
    churn_model_path: str,
    clv_model_name: str,
    clv_model_path: str,
    feature_cols: list,
    train_cutoffs: list,
    val_cutoff: str,
    test_cutoff: str,
    threshold: float,
    calibrator_path: str,
    validation_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    clv_validation_metrics: Optional[Dict[str, Any]] = None,
    clv_test_metrics: Optional[Dict[str, Any]] = None,
    reference_distribution_path: Optional[str] = None,
    segmentation_model: str = "K-Means",
    value_capture_decile: float = 0.10,
    churn_use_scaled: bool = True,
    clv_use_scaled: bool = False,
    clv_needs_aux: bool = True,
    clv_log_target: bool = False,
    value_tier_quantiles: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Assemble the model card dictionary."""
    config = load_config()
    version = config.get("serving", {}).get("model_version", "1.0.0")
    card = {
        "model_version": f"{version}+{_git_sha()}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "churn_model": churn_model_name,
        "churn_model_path": churn_model_path,
        "churn_use_scaled": churn_use_scaled,
        "clv_model": clv_model_name,
        "clv_model_path": clv_model_path,
        "clv_use_scaled": clv_use_scaled,
        "clv_needs_aux": clv_needs_aux,
        "clv_log_target": clv_log_target,
        "value_tier_quantiles": value_tier_quantiles or {"low": 0.0, "high": 1e12},
        "segmentation_model": segmentation_model,
        "feature_cols": list(feature_cols),
        "train_cutoffs": train_cutoffs,
        "validation_cutoff": val_cutoff,
        "test_cutoff": test_cutoff,
        "decision_threshold": threshold,
        "calibrator_path": calibrator_path,
        "reference_distribution_path": reference_distribution_path,
        "value_capture_decile": value_capture_decile,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "clv_validation_metrics": clv_validation_metrics or {},
        "clv_test_metrics": clv_test_metrics or {},
    }
    return card


def write_model_card(card: Dict[str, Any], path: Optional[Path] = None) -> Path:
    root = get_project_root()
    config = load_config()
    if path is None:
        art_dir = root / config.get("paths", {}).get("models_artifacts_dir", "models_artifacts")
        path = art_dir / CARD_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    logger.info(f"Wrote model card to {path}")
    return path


def read_model_card(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the model card; a missing card is surfaced to the caller."""
    root = get_project_root()
    config = load_config()
    if path is None:
        art_dir = root / config.get("paths", {}).get("models_artifacts_dir", "models_artifacts")
        path = art_dir / CARD_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Model card not found at {path}. Run the training pipeline (src/models/pipeline.py)."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)
