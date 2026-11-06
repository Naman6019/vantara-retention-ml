"""Classical churn models with leakage-safe selection and tuning.

Selection discipline (plan D1/G3):
  * hyperparameters are tuned by forward-chaining CV *inside the training
    cutoffs only*;
  * the best model is chosen on the **validation** snapshot;
  * the **test** snapshot is scored exactly once, after selection.

``select_best`` deliberately only accepts validation metrics so that a test-set
metric can never leak into model selection (there is a unit test for this).
"""

import time
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("churn_models")


def initialize_models(seed: int = 42) -> Dict[str, Any]:
    """Instantiate the 6 classical churn models required by PRD Section 8.1."""
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"),
            "use_scaled": True,
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, random_state=seed, class_weight="balanced"),
            "use_scaled": False,
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=seed,
                class_weight="balanced", n_jobs=-1,
            ),
            "use_scaled": False,
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=seed, eval_metric="logloss", n_jobs=-1,
            ),
            "use_scaled": False,
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=seed, class_weight="balanced",
                verbosity=-1, n_jobs=-1,
            ),
            "use_scaled": False,
        },
        "KNN": {
            "model": KNeighborsClassifier(n_neighbors=7, weights="distance"),
            "use_scaled": True,
        },
    }
    return models


def _positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cv_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit and evaluate a model with stratified CV on train and a held-out score set."""
    start_time = time.time()

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    model.fit(X_train, y_train)
    train_time = round(time.time() - start_time, 3)

    y_pred = model.predict(X_test)
    y_prob = _positive_proba(model, X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "cv_roc_auc_mean": round(float(np.mean(cv_auc_scores)), 4),
        "cv_roc_auc_std": round(float(np.std(cv_auc_scores)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_time_sec": train_time,
    }
    return metrics


def forward_chaining_cv_auc(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    cutoffs: np.ndarray,
    n_splits: int = 3,
) -> float:
    """Forward-chaining CV ROC-AUC over training cutoffs.

    The same customer appears at several cutoffs, so a shuffled k-fold would leak
    across time. Instead each fold trains on earlier cutoffs and validates on a
    later one; folds with a single class are skipped.
    """
    unique = np.sort(np.unique(cutoffs))
    if len(unique) < 2:
        return float("nan")

    n_splits = min(n_splits, len(unique) - 1)
    fold_scores = []
    for i in range(n_splits):
        split_at = len(unique) - n_splits + i
        train_cutoffs = unique[:split_at]
        val_cutoffs = [unique[split_at]]
        tr = np.isin(cutoffs, train_cutoffs)
        va = np.isin(cutoffs, val_cutoffs)
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        model.fit(X[tr], y[tr])
        fold_scores.append(roc_auc_score(y[va], _positive_proba(model, X[va])))
    return float(np.mean(fold_scores)) if fold_scores else float("nan")


def _optuna_search(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    cutoffs: np.ndarray,
    n_trials: int,
    seed: int,
) -> Tuple[Dict[str, Any], float]:
    """Optuna TPE search scored by forward-chaining CV ROC-AUC (training cutoffs only)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        if name == "Logistic Regression":
            params = {"C": trial.suggest_float("C", 0.01, 10.0, log=True)}
            model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, **params)
        elif name == "Decision Tree":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 40),
            }
            model = DecisionTreeClassifier(random_state=seed, class_weight="balanced", **params)
        elif name == "Random Forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 4, 14),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 40),
            }
            model = RandomForestClassifier(random_state=seed, class_weight="balanced", n_jobs=-1, **params)
        elif name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            model = xgb.XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=-1, **params)
        elif name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            model = lgb.LGBMClassifier(
                random_state=seed, class_weight="balanced", verbosity=-1, n_jobs=-1, **params
            )
        elif name == "KNN":
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            }
            model = KNeighborsClassifier(**params)
        else:
            raise ValueError(f"Unknown model for search: {name}")

        score = forward_chaining_cv_auc(model, X, y, cutoffs)
        return score if not np.isnan(score) else 0.0

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value)


def _model_from_params(name: str, params: Dict[str, Any], seed: int) -> Any:
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, **params)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=seed, class_weight="balanced", **params)
    if name == "Random Forest":
        return RandomForestClassifier(random_state=seed, class_weight="balanced", n_jobs=-1, **params)
    if name == "XGBoost":
        return xgb.XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=-1, **params)
    if name == "LightGBM":
        return lgb.LGBMClassifier(random_state=seed, class_weight="balanced", verbosity=-1, n_jobs=-1, **params)
    if name == "KNN":
        return KNeighborsClassifier(**params)
    raise ValueError(name)


def select_best(validation_results: Dict[str, Dict[str, float]], metric: str = "roc_auc") -> str:
    """Return the best model name from *validation* metrics only.

    This function's signature intentionally carries no test data: model selection
    can never see the test snapshot.
    """
    if not validation_results:
        raise ValueError("No validation results to select from.")
    return max(validation_results, key=lambda name: validation_results[name][metric])


def evaluate_on_test(model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
    """Score a fitted model on the test snapshot. Call this exactly once."""
    y_pred = model.predict(X_test)
    y_prob = _positive_proba(model, X_test)
    return {
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "probabilities": y_prob,
    }


def tune_and_compare_churn_models(
    data: Dict[str, Any],
    n_trials: Optional[int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Tune every classical model and return validation-ranked results.

    Test data is never passed in; the caller evaluates the selected model on test
    separately (see ``src.models.train``).
    """
    config = load_config()
    seed = seed if seed is not None else config.get("project", {}).get("random_seed", 42)
    n_trials = n_trials if n_trials is not None else config.get("modeling", {}).get("search", {}).get("n_trials", 30)

    base_models = initialize_models(seed=seed)
    X_tr, X_val = data["X_train"], data["X_val"]
    y_tr, y_val = data["y_train"], data["y_val"]
    X_tr_s, X_val_s = data["X_train_scaled"], data["X_val_scaled"]
    cutoffs = data["cutoff_train"]

    results = []
    fitted: Dict[str, Any] = {}
    for name, item in base_models.items():
        logger.info(f"Tuning {name} ({n_trials} trials)...")
        search_X = X_tr_s if item["use_scaled"] else X_tr
        best_params, best_cv = _optuna_search(name, search_X, y_tr, cutoffs, n_trials, seed)

        model = _model_from_params(name, best_params, seed)
        fit_X = X_tr_s if item["use_scaled"] else X_tr
        val_X = X_val_s if item["use_scaled"] else X_val
        model.fit(fit_X, y_tr)
        val_prob = _positive_proba(model, val_X)
        val_pred = model.predict(val_X)

        results.append({
            "Model": name,
            "Val ROC-AUC": round(float(roc_auc_score(y_val, val_prob)), 4),
            "Val Recall": round(float(recall_score(y_val, val_pred, zero_division=0)), 4),
            "Val Precision": round(float(precision_score(y_val, val_pred, zero_division=0)), 4),
            "Val F1": round(float(f1_score(y_val, val_pred, zero_division=0)), 4),
            "Val Accuracy": round(float(accuracy_score(y_val, val_pred)), 4),
            "CV ROC-AUC (fwd-chain)": round(best_cv, 4),
            "Best Params": best_params,
            "use_scaled": item["use_scaled"],
        })
        fitted[name] = model

    leaderboard = pd.DataFrame(results).sort_values(by=["Val ROC-AUC", "Val Recall"], ascending=False)
    leaderboard.reset_index(drop=True, inplace=True)
    return leaderboard, fitted


def save_leaderboard(leaderboard: pd.DataFrame, art_dir) -> None:
    lb = leaderboard.drop(columns=["Best Params"], errors="ignore")
    lb.to_parquet(art_dir / "churn_model_comparison.parquet")
    lb.to_csv(art_dir / "churn_model_comparison.csv", index=False)


if __name__ == "__main__":
    from src.models.data_split import split_data

    leaderboard, _ = tune_and_compare_churn_models(split_data())
    print(leaderboard.drop(columns=["Best Params"]).to_string(index=False))
