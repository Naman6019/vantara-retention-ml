"""90-day customer value regression with a momentum baseline and value-capture metric.

Plan D3:
  * target = net spend in ``[T, T+90d)``, clipped at 0;
  * every model must beat a **last-90-days spend** baseline;
  * candidates: Ridge on ``log1p`` target, LightGBM with a tweedie objective
    (fits zero-inflated heavy-tailed spend), and Random Forest;
  * selection is on validation MAE **and** top-decile value capture — the metric
    the retention team actually cares about (how much of the future value sits in
    the customers the model ranks highest). R^2 is reported, not used to select.
"""

import time
from typing import Any, Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler as SkScaler

from src.utils.logger import get_logger

logger = get_logger("clv_regressor")


class SpendBaseline:
    """Predict 90-day value as the spend in the last 90 days before T."""

    def __init__(self, col: str = "spend_last_90d"):
        self.col = col
        self.columns_ = None

    @staticmethod
    def _as_frame(X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    def fit(self, X: Any, y: Any = None) -> "SpendBaseline":
        self.columns_ = list(self._as_frame(X).columns)
        return self

    def predict(self, X: Any) -> np.ndarray:
        df = self._as_frame(X)
        if self.col in df.columns:
            return df[self.col].to_numpy(dtype=float)
        return np.zeros(len(df), dtype=float)


def initialize_regressors(seed: int = 42) -> Dict[str, Any]:
    """Instantiate the CLV candidates plus the momentum baseline."""
    models = {
        "Spend Baseline (last 90d)": {"model": SpendBaseline(), "use_scaled": False, "needs_aux": True},
        "Ridge Regression (log1p)": {
            "model": Pipeline([("scaler", SkScaler()), ("model", Ridge(alpha=10.0, random_state=seed))]),
            "use_scaled": False,
            "log_target": True,
            "needs_aux": True,
        },
        "LightGBM Tweedie": {
            "model": lgb.LGBMRegressor(
                objective="tweedie", tweedie_variance_power=1.5, n_estimators=300,
                max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                random_state=seed, verbosity=-1, n_jobs=-1,
            ),
            "use_scaled": False,
            "needs_aux": True,
        },
        "Random Forest Regressor": {
            "model": RandomForestRegressor(
                n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=seed, n_jobs=-1
            ),
            "use_scaled": False,
            "needs_aux": True,
        },
    }
    return models


def top_decile_value_capture(y_true: np.ndarray, y_pred: np.ndarray, decile: float = 0.10) -> float:
    """Share of total actual value held by the top-``decile`` ranked by prediction."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    total = y_true.sum()
    if total <= 0 or len(y_true) == 0:
        return 0.0
    k = max(1, int(np.ceil(len(y_true) * decile)))
    top_idx = np.argsort(y_pred)[::-1][:k]
    return float(y_true[top_idx].sum() / total)


def evaluate_regressor(
    model: Any,
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    y_test: np.ndarray,
    use_log_transform: bool = False,
    decile: float = 0.10,
) -> Dict[str, Any]:
    """Fit a regressor and return MAE/RMSE/R² plus top-decile value capture."""
    start_time = time.time()
    y_fit = np.log1p(np.maximum(0, y_train)) if use_log_transform else y_train
    model.fit(X_train, y_fit)
    train_time = round(time.time() - start_time, 3)

    raw = model.predict(X_test)
    y_pred = np.expm1(raw) if use_log_transform else raw
    # A customer's next-90-day value cannot plausibly exceed the largest value
    # observed in training; clipping keeps a mis-specified linear model honest
    # instead of letting expm1 explode on a heavy tail.
    upper = float(np.max(np.maximum(0, y_train))) if len(y_train) else float("inf")
    y_pred = np.clip(y_pred, 0.0, upper)

    return {
        "r2_score": round(float(r2_score(y_test, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "value_capture_top_decile": round(top_decile_value_capture(y_test, y_pred, decile), 4),
        "train_time_sec": train_time,
    }


def _train_matrix(data: Dict[str, Any], X: pd.DataFrame) -> pd.DataFrame:
    """Attach the auxiliary baseline column for value-model training."""
    if "spend_last_90d" in data and len(data["spend_last_90d"]) == len(X):
        out = X.copy()
        out["spend_last_90d"] = np.asarray(data["spend_last_90d"])
        return out
    return X


def tune_and_compare_clv_regressors(
    data: Dict[str, Any], decile: float = 0.10
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compare CLV candidates on the validation snapshot; return leaderboard + fitted."""
    models = initialize_regressors()
    X_train = _train_matrix(data, data["X_train"])
    X_val = _train_matrix(data, data["X_val"])
    y_train, y_val = data["clv_train"], data["clv_val"]

    results = []
    fitted: Dict[str, Any] = {}
    for name, item in models.items():
        logger.info(f"Fitting CLV candidate: {name}")
        metrics = evaluate_regressor(
            item["model"], X_train, y_train, X_val, y_val,
            use_log_transform=item.get("log_target", False), decile=decile,
        )
        results.append({
            "Model": name,
            "Val R² Score": metrics["r2_score"],
            "Val MAE (£)": metrics["mae"],
            "Val RMSE (£)": metrics["rmse"],
            "Top-Decile Value Capture": metrics["value_capture_top_decile"],
            "Training Time (s)": metrics["train_time_sec"],
        })
        fitted[name] = item["model"]

    # Selection: best MAE, tie-broken by value capture.
    leaderboard = pd.DataFrame(results).sort_values(
        by=["Val MAE (£)", "Top-Decile Value Capture"], ascending=[True, False]
    )
    leaderboard.reset_index(drop=True, inplace=True)
    return leaderboard, fitted


if __name__ == "__main__":
    from src.models.data_split import split_data

    lb, _ = tune_and_compare_clv_regressors(split_data())
    print(lb.to_string(index=False))
