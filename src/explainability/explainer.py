from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay

from src.models.data_split import FEATURE_COLS, split_data
from src.utils.config import get_project_root
from src.utils.logger import get_logger

logger = get_logger("explainer")


class ModelExplainer:
    """
    Explainable AI (XAI) engine supporting SHAP, LIME, Partial Dependence,
    and plain-language narrative translation for retail stakeholders.
    """
    def __init__(self, model_path: Optional[Path] = None):
        root = get_project_root()
        art_dir = root / "models_artifacts"

        if model_path is None:
            # Prefer XGBoost or Random Forest for tree explainability
            xgb_path = art_dir / "churn_xgboost.joblib"
            rf_path = art_dir / "churn_random_forest.joblib"
            best_path = art_dir / "best_churn_model.joblib"
            if xgb_path.exists():
                model_path = xgb_path
            elif rf_path.exists():
                model_path = rf_path
            else:
                model_path = best_path

        self.model = joblib.load(model_path)
        self.model_name = type(self.model).__name__
        logger.info(f"Loaded model for XAI: {self.model_name}")

        self.data = split_data()
        self.X_train = self.data["X_train"]
        self.X_test = self.data["X_test"]
        self.feature_names = FEATURE_COLS

        # Initialize SHAP TreeExplainer
        self.shap_explainer = shap.TreeExplainer(self.model)

        # Initialize LIME Tabular Explainer
        self.lime_explainer = LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.feature_names,
            class_names=["Retained", "Churned"],
            mode="classification",
            random_state=42,
        )

    def generate_global_shap_summary(self, save_path: Optional[Path] = None) -> Path:
        """
        Generates and saves the Global SHAP beeswarm summary plot.
        """
        root = get_project_root()
        if save_path is None:
            save_path = root / "docs" / "shap_global_summary.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Computing global SHAP values on test set...")
        shap_values = self.shap_explainer(self.X_test)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, self.X_test, show=False)
        plt.title("Global Feature Importance — SHAP Summary Plot", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved Global SHAP plot to: {save_path}")
        return save_path

    def explain_single_customer_shap(self, customer_index: int = 0) -> Dict[str, Any]:
        """
        Computes local SHAP explanation for an individual customer.
        Returns top drivers and base values.
        """
        row = self.X_test.iloc[[customer_index]]
        shap_exp = self.shap_explainer(row)
        values = shap_exp.values[0]
        base_val = float(shap_exp.base_values[0]) if hasattr(shap_exp.base_values, "__len__") else float(shap_exp.base_values)

        # Rank features by absolute impact
        impacts = []
        for col, val, shap_val in zip(self.feature_names, row.iloc[0], values, strict=False):
            impacts.append({
                "feature": col,
                "value": float(val),
                "shap_impact": round(float(shap_val), 4),
                "direction": "Increases Churn" if shap_val > 0 else "Decreases Churn",
            })

        impacts.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
        return {
            "base_value": round(base_val, 4),
            "top_drivers": impacts[:6],
            "all_impacts": impacts,
        }

    def explain_single_customer_lime(self, customer_index: int = 0) -> List[Tuple[str, float]]:
        """
        Generates local LIME explanation for a single customer.
        """
        row = self.X_test.iloc[customer_index].values
        exp = self.lime_explainer.explain_instance(
            row, self.model.predict_proba, num_features=6
        )
        return exp.as_list()

    def generate_partial_dependence_plots(self, features: List[str] | None = None, save_path: Optional[Path] = None) -> Path:
        """
        Generates Partial Dependence Plots (PDP) illustrating non-linear effects
        of primary drivers (e.g. Recency and Frequency) on churn risk.
        """
        if features is None:
            features = ["recency", "frequency"]
        root = get_project_root()
        if save_path is None:
            save_path = root / "docs" / "pdp_plots.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating Partial Dependence Plots for: {features}...")
        fig, ax = plt.subplots(figsize=(10, 4))
        PartialDependenceDisplay.from_estimator(
            self.model, self.X_test, features=features, ax=ax, grid_resolution=40
        )
        plt.suptitle("Partial Dependence of Churn Probability on Key Drivers", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved PDP plot to: {save_path}")
        return save_path

    def translate_to_narrative(self, shap_explanation: Dict[str, Any], churn_prob: float) -> str:
        """
        PRD Section 10 requirement:
        Translates SHAP values into plain-language marketing sentences for non-technical stakeholders.
        """
        top_positive = [d for d in shap_explanation["top_drivers"] if d["shap_impact"] > 0]
        top_negative = [d for d in shap_explanation["top_drivers"] if d["shap_impact"] < 0]

        risk_tier = "High Churn Risk" if churn_prob >= 0.65 else ("Moderate Risk" if churn_prob >= 0.40 else "Low Churn Risk")

        narrative = f"Customer Risk Assessment: {risk_tier} (Predicted Churn Probability: {churn_prob:.1%}).\n"

        if top_positive:
            reasons = []
            for d in top_positive[:2]:
                feat_name = d["feature"].replace("_", " ").title()
                reasons.append(f"{feat_name} ({d['value']:.1f}, impact: +{d['shap_impact']:.2f})")
            narrative += f"Primary churn warning signs: {'; '.join(reasons)}. "

        if top_negative:
            positives = []
            for d in top_negative[:2]:
                feat_name = d["feature"].replace("_", " ").title()
                positives.append(f"{feat_name} ({d['value']:.1f}, impact: {d['shap_impact']:.2f})")
            narrative += f"Retention mitigating factors: {'; '.join(positives)}."

        return narrative


def run_explainability_suite() -> Dict[str, Any]:
    """
    Executes all explainability requirements: global SHAP, 3 persona local explanations,
    LIME comparison, PDP plots, and plain-English narrative translation.
    """
    explainer = ModelExplainer()

    # 1. Global SHAP plot
    global_path = explainer.generate_global_shap_summary()

    # 2. PDP plots
    pdp_path = explainer.generate_partial_dependence_plots()

    # 3. Pick 3 representative personas from test set: Low Risk, High Risk, Borderline
    probs = explainer.model.predict_proba(explainer.X_test)[:, 1]
    low_idx = int(np.argmin(probs))
    high_idx = int(np.argmax(probs))
    borderline_idx = int(np.argmin(np.abs(probs - 0.50)))

    personas_exps = {}
    for label, idx in [("Low-Risk Customer", low_idx), ("High-Risk Customer", high_idx), ("Borderline Customer", borderline_idx)]:
        prob = float(probs[idx])
        shap_exp = explainer.explain_single_customer_shap(idx)
        narrative = explainer.translate_to_narrative(shap_exp, prob)
        personas_exps[label] = {
            "index": idx,
            "churn_prob": round(prob, 4),
            "narrative": narrative,
            "top_drivers": shap_exp["top_drivers"][:3],
        }

    # 4. LIME local comparison on Borderline Customer
    lime_exp = explainer.explain_single_customer_lime(borderline_idx)

    logger.info("Explainability suite executed successfully.")
    return {
        "global_shap_plot": str(global_path),
        "pdp_plot": str(pdp_path),
        "representative_personas": personas_exps,
        "lime_borderline_explanation": lime_exp,
    }


if __name__ == "__main__":
    xai_res = run_explainability_suite()
    print("\n--- XAI Global Artifacts Saved ---")
    print("Global SHAP Plot:", xai_res["global_shap_plot"])
    print("PDP Plot:", xai_res["pdp_plot"])
    print("\n--- Plain-Language Marketing Narratives for 3 Personas ---")
    for persona, details in xai_res["representative_personas"].items():
        print(f"\n[{persona}] (Index {details['index']}):")
        print(details["narrative"])
    print("\n--- LIME Local Comparison (Borderline Account) ---")
    for rule, weight in xai_res["lime_borderline_explanation"]:
        print(f"  {rule}: {weight:+.4f}")
