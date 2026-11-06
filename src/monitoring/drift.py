"""Population Stability Index (PSI) drift monitoring (plan D7).

Reference decile breakpoints are saved at training time
(``models_artifacts/reference_distribution.json``). After each batch run we
compare the current batch distribution against that reference per feature and
flag PSI > ``monitoring.psi_threshold``.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("drift")

_EPS = 1e-6


def psi(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    """Population Stability Index between two aligned proportion vectors."""
    ref = np.asarray(ref_counts, dtype=float)
    cur = np.asarray(cur_counts, dtype=float)
    ref = ref / max(ref.sum(), _EPS)
    cur = cur / max(cur.sum(), _EPS)
    ref = np.clip(ref, _EPS, None)
    cur = np.clip(cur, _EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_from_edges(ref_values: np.ndarray, cur_values: np.ndarray, edges: np.ndarray) -> float:
    """PSI of ``cur_values`` against ``ref_values`` using fixed bin edges."""
    edges = np.asarray(edges, dtype=float)
    # Make edges strictly increasing so histogram doesn't error.
    edges = np.maximum.accumulate(edges)
    ref_counts, _ = np.histogram(ref_values, bins=edges)
    cur_counts, _ = np.histogram(cur_values, bins=edges)
    return psi(ref_counts, cur_counts)


def load_reference(path=None) -> Dict[str, Dict[str, List[float]]]:
    import json

    root = get_project_root()
    if path is None:
        config = load_config()
        art_dir = root / config.get("paths", {}).get("models_artifacts_dir", "models_artifacts")
        path = art_dir / "reference_distribution.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_drift(
    current: pd.DataFrame,
    reference: Dict[str, Any] | None = None,
    threshold: float | None = None,
) -> List[Dict[str, Any]]:
    """Return per-feature PSI rows for ``current`` against the saved reference."""
    if reference is None:
        reference = load_reference()
    if threshold is None:
        threshold = load_config().get("monitoring", {}).get("psi_threshold", 0.2)

    rows = []
    for feature, ref in reference.items():
        if feature not in current.columns:
            continue
        edges = np.asarray(ref["edges"], dtype=float)
        ref_mid = (edges[:-1] + edges[1:]) / 2.0
        cur_values = pd.to_numeric(current[feature], errors="coerce").dropna().to_numpy()
        if cur_values.size == 0:
            continue
        value = psi_from_edges(ref_mid, cur_values, edges)
        rows.append({
            "feature": feature,
            "psi": round(value, 4),
            "drifted": int(value > threshold),
        })
    rows.sort(key=lambda r: r["psi"], reverse=True)
    return rows


def persist_drift(rows: List[Dict[str, Any]], batch_date: str, session=None) -> int:
    """Write drift rows to the ``drift_reports`` table."""
    from src.serving import db
    from src.serving.models import DriftReport

    def _write(sess) -> int:
        for row in rows:
            sess.add(DriftReport(feature=row["feature"], psi=row["psi"], drifted=row["drifted"], batch_date=batch_date))
        return len(rows)

    if session is not None:
        return _write(session)
    db.ensure_engine()
    with db.get_session() as sess:
        return _write(sess)


if __name__ == "__main__":
    import json

    from src.models.data_split import split_data

    data = split_data(save_scaler=False)
    ref = load_reference()
    rows = compute_drift(data["X_test_raw"], ref)
    print(json.dumps(rows, indent=2))
