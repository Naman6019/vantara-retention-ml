"""Regenerate the report comparison tables from MLflow and the model card (PRD §8.3).

Usage: python scripts/make_report_tables.py
Writes Markdown tables to docs/tables.md so the final report is never hand-typed.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_project_root  # noqa: E402


def _df_to_md(df: pd.DataFrame, index: bool = False) -> str:
    return df.to_markdown(index=index)


def main() -> None:
    root = get_project_root()
    art = root / "models_artifacts"
    out_dir = root / "docs"
    out_dir.mkdir(exist_ok=True)

    card = json.loads((art / "model_card.json").read_text(encoding="utf-8"))
    churn = pd.read_csv(art / "churn_model_comparison.csv")
    clv = pd.read_csv(art / "clv_model_comparison.csv")
    operating = card["test_metrics"].get("operating_point", {})
    operating_recall = operating.get("recall", card["test_metrics"].get("recall"))

    sections = []
    sections.append("## Selected model (from model_card.json)\n")
    sections.append(
        f"- Churn model: **{card['churn_model']}** (threshold {card['decision_threshold']:.4f})\n"
        f"- Value model: **{card['clv_model']}**\n"
        f"- Validation ROC-AUC: {card['validation_metrics'].get('roc_auc')} | "
        f"Test ROC-AUC: {card['test_metrics'].get('roc_auc')} | "
        f"Test recall at the operating threshold: {operating_recall}\n"
        f"- CLV validation MAE: £{card['clv_validation_metrics'].get('mae')} | "
        f"top-decile value capture: {card['clv_validation_metrics'].get('value_capture_top_decile')}\n"
    )
    sections.append("\n## Churn model comparison (validation-ranked)\n")
    sections.append(_df_to_md(churn))
    sections.append("\n\n## Value model comparison\n")
    sections.append(_df_to_md(clv))

    five_fold_path = art / "five_fold_cv.csv"
    if five_fold_path.exists():
        five_fold = pd.read_csv(five_fold_path)
        sections.append("\n\n## 5-fold stratified CV (PRD Section 11)\n")
        sections.append(_df_to_md(five_fold))
        sections.append(
            "\n\n*Forward-chaining CV remains the selection criterion because the same "
            "customer appears at several stacked snapshots; the 5-fold column is the "
            "PRD-required diagnostic on the training partition.*\n"
        )

    confusion_path = art / "test_confusion.json"
    if confusion_path.exists():
        cm = json.loads(confusion_path.read_text(encoding="utf-8"))
        tn, fp = cm["confusion_matrix"][0]
        fn, tp = cm["confusion_matrix"][1]
        table = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actual retained", "Actual churned"],
            columns=["Predicted retained", "Predicted churn"],
        )
        sections.append(
            f"\n\n## Test confusion matrix (threshold {cm['threshold']:.4f})\n"
        )
        sections.append(_df_to_md(table, index=True))
        sections.append(
            f"\n\nAccuracy {cm['accuracy']:.1%}, precision {cm['precision']:.1%}, "
            f"recall {cm['recall']:.1%}, ROC-AUC {cm['roc_auc']:.4f}.\n"
        )

    (out_dir / "tables.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {out_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
