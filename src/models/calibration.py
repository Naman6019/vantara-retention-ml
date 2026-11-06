"""Probability calibration and decision-threshold selection (plan D4).

The raw probability from a classifier is not directly usable for a
*value-at-risk* ranking: ``value_at_risk = P(churn) x predicted_value`` needs
``P(churn)`` to be calibrated. We fit an isotonic regressor on the validation
snapshot, then choose the operating threshold as the highest-precision
threshold that still keeps recall at or above the PRD target.
"""

from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from src.utils.logger import get_logger

logger = get_logger("calibration")

DEFAULT_RECALL_TARGET = 0.70


class ProbabilityCalibrator:
    """Isotonic (or sigmoid) probability calibrator with quantile risk tiers."""

    def __init__(
        self,
        method: str = "isotonic",
        tier_quantiles: tuple = (0.40, 0.65),
    ):
        self.method = method
        self.tier_quantiles = tier_quantiles
        self.model_: Optional[Any] = None
        self.tier_cutoffs_: np.ndarray = np.array([0.0, 0.5])

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        probs = np.asarray(probs, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        else:
            from sklearn.linear_model import LogisticRegression

            self.model_ = LogisticRegression()
            probs = probs.reshape(-1, 1)
        self.model_.fit(probs, y)

        calibrated = self.predict(np.asarray(probs).reshape(-1) if self.method != "isotonic" else probs)
        self.tier_cutoffs_ = np.quantile(calibrated, list(self.tier_quantiles))
        logger.info(
            f"Fitted {self.method} calibrator; risk-tier cutoffs "
            f"(moderate/high) = {self.tier_cutoffs_.round(4)}"
        )
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Calibrator is not fitted.")
        probs = np.asarray(probs, dtype=float)
        if self.method == "isotonic":
            return np.clip(self.model_.predict(probs), 0.0, 1.0)
        return self.model_.predict_proba(probs.reshape(-1, 1))[:, 1]

    def tiers(self, calibrated_probs: np.ndarray) -> np.ndarray:
        """Map calibrated probabilities to Low/Moderate/High using saved cutoffs."""
        lo, hi = float(self.tier_cutoffs_[0]), float(self.tier_cutoffs_[1])
        out = np.full(len(calibrated_probs), "Low", dtype=object)
        arr = np.asarray(calibrated_probs)
        out[(arr >= lo) & (arr < hi)] = "Moderate"
        out[arr >= hi] = "High"
        return out

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "ProbabilityCalibrator":
        return joblib.load(path)


def choose_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    recall_target: float = DEFAULT_RECALL_TARGET,
) -> Dict[str, float]:
    """Return the highest threshold that keeps recall >= ``recall_target``.

    Ties are broken toward the highest precision, which is what the retention
    team wants once the recall floor is met.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)

    candidates = np.unique(np.round(probs, 6))
    candidates = np.concatenate([candidates, [candidates[-1] + 1e-9]]) if candidates.size else np.array([0.5])

    best: Optional[Dict[str, float]] = None
    fallback: Optional[Dict[str, float]] = None
    for t in candidates:
        pred = (probs >= t).astype(int)
        if pred.sum() == 0:
            continue
        recall = recall_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        entry = {
            "threshold": float(t),
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        }
        if fallback is None or recall > fallback["recall"]:
            fallback = entry
        if recall >= recall_target:
            if best is None or entry["threshold"] > best["threshold"]:
                best = entry

    if best is None:
        logger.warning(
            f"No threshold met recall >= {recall_target:.2f}; falling back to max-recall threshold."
        )
        assert fallback is not None
        return fallback

    logger.info(
        f"Chosen threshold {best['threshold']:.4f}: recall={best['recall']:.3f}, "
        f"precision={best['precision']:.3f}, f1={best['f1']:.3f}"
    )
    return best
