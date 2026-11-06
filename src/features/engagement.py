"""Engagement score with a persisted, training-only quantile reference.

The engagement score is a composite of recency (inverted), frequency and
monetary value, expressed as a 0-100 score. To keep a single customer's score
reproducible at inference time (plan D10), the percentile breakpoints are
*fitted* on the training snapshots only and then applied against the saved
reference rather than re-ranked against whichever batch happens to be scored.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("engagement")

REFERENCE_KEYS = ["recency", "frequency", "monetary_total"]


def _weights() -> Dict[str, float]:
    config = load_config()
    w = config.get("features", {}).get("weights", {})
    return {
        "r": float(w.get("recency_weight", 0.4)),
        "f": float(w.get("frequency_weight", 0.3)),
        "m": float(w.get("monetary_weight", 0.3)),
    }


def fit_reference(
    df: pd.DataFrame,
    quantiles: np.ndarray | None = None,
) -> Dict[str, Dict[str, list]]:
    """Fit percentile breakpoints for recency/frequency/monetary.

    Recency is inverted at scoring time, so we store its raw breakpoints and
    invert the *rank* rather than the distribution.
    """
    if quantiles is None:
        quantiles = np.linspace(0.0, 1.0, 101)

    ref: Dict[str, Dict[str, list]] = {}
    for col in REFERENCE_KEYS:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            continue
        breakpoints = np.quantile(values, quantiles)
        # np.interp needs strictly increasing x; collapse ties deterministically.
        breakpoints = np.maximum.accumulate(breakpoints)
        ref[col] = {"quantiles": quantiles.tolist(), "breakpoints": breakpoints.tolist()}
    return ref


def _percentile_rank(values: pd.Series, col: str, ref: Dict[str, Any]) -> pd.Series:
    """Map raw values to [0, 1] using the saved breakpoints (linear interp)."""
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if col not in ref:
        # Fallback: rank against the incoming batch (legacy behaviour).
        return numeric.rank(pct=True)
    breakpoints = np.asarray(ref[col]["breakpoints"], dtype=float)
    quantiles = np.asarray(ref[col]["quantiles"], dtype=float)
    ranks = np.interp(
        numeric.to_numpy(), breakpoints, quantiles, left=quantiles[0], right=quantiles[-1]
    )
    return pd.Series(ranks, index=values.index)


def score(df: pd.DataFrame, ref: Dict[str, Any] | None = None) -> pd.Series:
    """Compute the 0-100 engagement score for every row in ``df``.

    When ``ref`` is provided (the fitted training reference), a single row scored
    alone yields exactly the same value as when scored inside a batch.
    """
    if ref is None:
        ref = fit_reference(df)

    w = _weights()
    r_rank = 1.0 - _percentile_rank(df["recency"], "recency", ref)
    f_rank = _percentile_rank(df["frequency"], "frequency", ref)
    m_rank = _percentile_rank(df["monetary_total"], "monetary_total", ref)

    score_values = 100.0 * (w["r"] * r_rank + w["f"] * f_rank + w["m"] * m_rank)
    return pd.Series(score_values, index=df.index).round(2)
