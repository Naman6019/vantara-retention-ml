"""Generate the PRD reporting evidence that is not a model artifact.

Produces:
  * ``models_artifacts/five_fold_cv.csv``      - 5-fold stratified CV (PRD Section 11)
  * ``models_artifacts/test_confusion.json``   - confusion matrix at the deployed threshold
  * ``docs/confusion_matrix.png``              - confusion-matrix figure
  * ``docs/loss_curves.png``                   - ANN/LSTM train-vs-validation loss
  * ``docs/shap_force_*.png``                  - SHAP force plots for low/high/borderline customers
  * ``docs/shap_force_summary.json``           - the customers and their top drivers

Usage:
    python -m src.models.dl_models        # writes the two loss-history JSON files
    python scripts/make_report_figures.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.calibration import ProbabilityCalibrator  # noqa: E402
from src.models.churn_models import _model_from_params  # noqa: E402
from src.models.data_split import split_data  # noqa: E402
from src.utils.config import get_project_root  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("report_figures")

SCALED_MODELS = {"Logistic Regression", "KNN"}


def _coerce(value: str) -> Any:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def load_churn_params(root: Path) -> Dict[str, Dict[str, Any]]:
    """Read the tuned hyperparameters that ``train_all`` logged to MLflow."""
    params: Dict[str, Dict[str, Any]] = {}
    mlruns = root / "mlruns"
    if not mlruns.exists():
        raise FileNotFoundError("mlruns/ not found - run `python -m src.models.train` first.")
    for experiment in mlruns.iterdir():
        if not experiment.is_dir() or not experiment.name.isdigit():
            continue
        for run in experiment.iterdir():
            meta = run / "meta.yaml"
            if not meta.exists():
                continue
            match = re.search(r"run_name:\s*(.+)", meta.read_text(encoding="utf-8"))
            name = match.group(1).strip() if match else ""
            if not name.startswith("churn::"):
                continue
            model_name = name.split("::", 1)[1]
            run_params = run / "params"
            if model_name in params or not run_params.is_dir():
                continue
            params[model_name] = {
                item.name: _coerce(item.read_text(encoding="utf-8").strip())
                for item in run_params.iterdir()
            }
    if not params:
        raise FileNotFoundError("No churn::* MLflow runs found.")
    return params


def five_fold_cv(root: Path, data: Dict[str, Any], params: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """5-fold stratified CV on the training snapshots (PRD Section 11)."""
    rows: List[Dict[str, Any]] = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, run_params in sorted(params.items()):
        use_scaled = name in SCALED_MODELS
        X = data["X_train_scaled"] if use_scaled else data["X_train"]
        model = _model_from_params(name, run_params, 42)
        scores = cross_val_score(model, X, data["y_train"], cv=cv, scoring="roc_auc", n_jobs=-1)
        rows.append({
            "Model": name,
            "CV ROC-AUC (5-fold)": round(float(np.mean(scores)), 4),
            "CV std": round(float(np.std(scores)), 4),
            "Folds": len(scores),
        })
        logger.info(f"{name}: 5-fold ROC-AUC {rows[-1]['CV ROC-AUC (5-fold)']:.4f}")
    table = pd.DataFrame(rows).sort_values("CV ROC-AUC (5-fold)", ascending=False)
    table.reset_index(drop=True, inplace=True)
    table.to_csv(root / "models_artifacts/five_fold_cv.csv", index=False)
    return table


def test_confusion(
    root: Path, data: Dict[str, Any], params: Dict[str, Dict[str, Any]], card: Dict[str, Any]
) -> Dict[str, Any]:
    """Refit the selected model on train, calibrate, and score the test snapshot once."""
    name = card["churn_model"]
    model = _model_from_params(name, params[name], 42)
    use_scaled = bool(card.get("churn_use_scaled", name in SCALED_MODELS))
    X_train = data["X_train_scaled"] if use_scaled else data["X_train"]
    X_test = data["X_test_scaled"] if use_scaled else data["X_test"]
    model.fit(X_train, data["y_train"])

    calibrator = ProbabilityCalibrator.load(root / card["calibrator_path"])
    raw_test = model.predict_proba(X_test)[:, 1]
    calibrated = calibrator.predict(raw_test)
    threshold = float(card["decision_threshold"])
    pred = (calibrated >= threshold).astype(int)
    matrix = confusion_matrix(data["y_test"], pred)
    tn, fp, fn, tp = matrix.ravel()
    payload = {
        "model": name,
        "threshold": threshold,
        "roc_auc": round(float(roc_auc_score(data["y_test"], raw_test)), 4),
        "confusion_matrix": matrix.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": round(float((tp + tn) / matrix.sum()), 4),
        "precision": round(float(tp / (tp + fp)), 4) if tp + fp else 0.0,
        "recall": round(float(tp / (tp + fn)), 4) if tp + fn else 0.0,
    }
    (root / "models_artifacts/test_confusion.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def plot_confusion(root: Path, payload: Dict[str, Any]) -> None:
    matrix = np.array(payload["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(matrix, cmap="Blues")
    labels = ["Retained", "Churned"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"Churn confusion matrix - test snapshot\n"
        f"threshold {payload['threshold']:.4f}, recall {payload['recall']:.1%}"
    )
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{matrix[i, j]:,}",
                ha="center", va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "#0f172a",
                fontsize=14,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(root / "docs/confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(root: Path) -> Dict[str, Any]:
    art = root / "models_artifacts"
    histories = {
        "ann": json.loads((art / "ann_loss_history.json").read_text(encoding="utf-8")),
        "lstm": json.loads((art / "lstm_loss_history.json").read_text(encoding="utf-8")),
    }
    metrics_path = art / "dl_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    titles = {"ann": "Feed-forward ANN", "lstm": "Sequence LSTM"}
    for ax, key in zip(axes, ["ann", "lstm"], strict=False):
        history = histories[key]
        ax.plot(history["epoch"], history["train_loss"], label="Training loss", linewidth=2)
        ax.plot(history["epoch"], history["val_loss"], label="Validation loss", linewidth=2)
        if key == "ann" and metrics.get("ann_best_epoch"):
            best_epoch = metrics["ann_best_epoch"]
            ax.axvline(best_epoch, color="#64748b", linestyle="--", linewidth=1)
            ax.annotate(
                f"best epoch {best_epoch}",
                xy=(best_epoch, max(history["val_loss"])),
                xytext=(best_epoch + 0.4, max(history["val_loss"])),
                fontsize=8,
                color="#475569",
            )
        ax.set_title(titles[key])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary cross-entropy loss")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Deep-learning convergence: training vs validation loss", fontsize=13)
    plt.tight_layout()
    fig.savefig(root / "docs/loss_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return metrics


def _select_representatives(probs: np.ndarray, threshold: float) -> Dict[str, int]:
    low = int(np.argmin(probs))
    high = int(np.argmax(probs))
    candidates = [i for i in range(len(probs)) if i not in {low, high}]
    borderline = min(candidates, key=lambda i: abs(probs[i] - threshold))
    return {"low_risk": low, "high_risk": high, "borderline": borderline}


def shap_force_plots(root: Path, data: Dict[str, Any], card: Dict[str, Any]) -> Dict[str, Any]:
    """Force plots for the *served* model on raw feature values (PRD Section 10)."""
    model = joblib.load(root / card["churn_model_path"])
    scaler = joblib.load(root / "models_artifacts/scaler.joblib")
    calibrator = ProbabilityCalibrator.load(root / card["calibrator_path"])
    X_test = data["X_test"]
    threshold = float(card["decision_threshold"])

    scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    calibrated = calibrator.predict(model.predict_proba(scaled)[:, 1])
    picks = _select_representatives(calibrated, threshold)

    def predict_fn(raw: Any) -> np.ndarray:
        frame = pd.DataFrame(raw, columns=X_test.columns)
        frame_scaled = pd.DataFrame(
            scaler.transform(frame), columns=X_test.columns, index=frame.index
        )
        return model.predict_proba(frame_scaled)[:, 1]

    background = X_test.sample(min(80, len(X_test)), random_state=42)
    explainer = shap.Explainer(
        predict_fn, shap.maskers.Independent(background, max_samples=len(background))
    )

    records: Dict[str, Any] = {}
    for label, index in picks.items():
        row = X_test.iloc[[index]]
        explanation = explainer(row)
        base = float(np.ravel(explanation.base_values)[0])
        values = explanation.values[0]
        display = row.round(3)

        shap.force_plot(
            base,
            values,
            display.iloc[0],
            matplotlib=True,
            show=False,
            figsize=(13, 3.4),
            text_rotation=18,
            contribution_threshold=0.08,
        )
        plt.tight_layout()
        plt.savefig(root / f"docs/shap_force_{label}.png", dpi=180, bbox_inches="tight")
        plt.close()

        order = np.argsort(-np.abs(values))[:5]
        records[label] = {
            "customer_id": float(data["id_test"][index]),
            "calibrated_probability": round(float(calibrated[index]), 4),
            "prediction": "churn" if calibrated[index] >= threshold else "retained",
            "top_drivers": [
                {
                    "feature": X_test.columns[j],
                    "value": round(float(row.iloc[0, j]), 3),
                    "shap_impact": round(float(values[j]), 4),
                }
                for j in order
            ],
        }

    (root / "docs/shap_force_summary.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return records


def main() -> None:
    root = get_project_root()
    (root / "docs").mkdir(exist_ok=True)
    (root / "models_artifacts").mkdir(exist_ok=True)

    data = split_data(save_scaler=False)
    card = json.loads((root / "models_artifacts/model_card.json").read_text(encoding="utf-8"))
    params = load_churn_params(root)

    cv_table = five_fold_cv(root, data, params)
    print("\n5-fold stratified CV (training snapshots)")
    print(cv_table.to_string(index=False))

    payload = test_confusion(root, data, params, card)
    print("\nTest confusion matrix at the deployed threshold")
    print(payload)

    plot_confusion(root, payload)
    plot_loss_curves(root)

    records = shap_force_plots(root, data, card)
    print("\nSHAP force-plot customers")
    for label, record in records.items():
        print(f"- {label}: customer {record['customer_id']:.0f}, P(churn)={record['calibrated_probability']:.1%}")

    print(f"\nWrote figures to {root / 'docs'}")


if __name__ == "__main__":
    main()
