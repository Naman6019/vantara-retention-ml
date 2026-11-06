"""Personalised product recommendations (PRD §1 and §2.3, plan P4).

Replaces best-seller logic with item-item cosine similarity on each customer's
purchase history. Products bought by the same customers score as similar, and a
customer is offered what is most similar to what they already bought.

Evaluation follows the same out-of-time discipline as the churn model: neighbour
count is chosen on the validation cutoff, the test cutoff is scored once, and the
target is products the customer had not bought before the cutoff.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("product_recommender")

MODEL_FILE = "product_recommender.joblib"
METRICS_FILE = "product_recommender_metrics.json"


def _purchases(transactions: pd.DataFrame) -> pd.DataFrame:
    """Keep real purchase lines: cancellations and returns are not interest."""
    keep = ~transactions["is_cancellation"] & (transactions["quantity"] > 0)
    return transactions.loc[keep].dropna(subset=["customer_id", "stock_code"])


class ItemCosineRecommender:
    def __init__(self, n_neighbours: int = 50) -> None:
        self.n_neighbours = n_neighbours

    def fit(self, transactions: pd.DataFrame) -> "ItemCosineRecommender":
        purchases = _purchases(transactions)
        customers = pd.Categorical(purchases["customer_id"])
        products = pd.Categorical(purchases["stock_code"].astype(str))
        self.product_ids_ = np.asarray(products.categories, dtype=str)
        self.customer_rows_ = {float(c): i for i, c in enumerate(customers.categories)}

        history = sparse.csr_matrix(
            (np.ones(len(purchases), dtype=np.float32), (customers.codes, products.codes)),
            shape=(len(customers.categories), len(products.categories)),
        )
        history.data[:] = 1.0  # csr_matrix sums duplicates; a repeat buy is still one buyer.
        self.history_ = history

        buyers = np.asarray(history.sum(axis=0)).ravel()
        self.popularity_order_ = np.argsort(-buyers, kind="stable")
        self.similarity_ = self._top_neighbours(history, buyers)

        descriptions = purchases.dropna(subset=["description"]).groupby("stock_code")["description"]
        self.descriptions_ = descriptions.agg(lambda s: s.value_counts().index[0]).to_dict()
        return self

    def _top_neighbours(self, history: sparse.csr_matrix, buyers: np.ndarray) -> sparse.csr_matrix:
        normalised = history @ sparse.diags(1.0 / np.sqrt(np.maximum(buyers, 1.0)))
        similarity = (normalised.T @ normalised).tocsr()
        similarity.setdiag(0)
        similarity.eliminate_zeros()

        rows, cols, values = [], [], []
        for item in range(similarity.shape[0]):
            start, end = similarity.indptr[item], similarity.indptr[item + 1]
            neighbours, scores = similarity.indices[start:end], similarity.data[start:end]
            best = np.argsort(-scores, kind="stable")[: self.n_neighbours]
            rows.append(np.full(len(best), item))
            cols.append(neighbours[best])
            values.append(scores[best])
        return sparse.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=similarity.shape,
        )

    def _describe(self, item: int) -> str:
        return str(self.descriptions_.get(self.product_ids_[item], ""))

    def _because(self, item: int, bought: np.ndarray) -> str:
        """Name the customer's own purchase that most supports this suggestion."""
        row = self.similarity_.getrow(item)
        in_history = np.isin(row.indices, bought)
        return self._describe(row.indices[in_history][np.argmax(row.data[in_history])])

    def popular_unbought(self, customer_id: float, top_k: int = 10) -> List[str]:
        """Best-seller baseline: the same logic this model replaces."""
        row = self.customer_rows_.get(float(customer_id))
        bought = set() if row is None else set(self.history_.getrow(row).indices)
        items = [i for i in self.popularity_order_ if i not in bought][:top_k]
        return [str(self.product_ids_[i]) for i in items]

    def recommend(self, customer_id: float, top_k: int = 10) -> List[Dict[str, Any]]:
        """Top products for a known customer; empty if they have no history."""
        row = self.customer_rows_.get(float(customer_id))
        if row is None:
            return []
        bought = self.history_.getrow(row).indices
        interest = np.zeros(self.similarity_.shape[1], dtype=np.float32)
        interest[bought] = 1.0
        scores = self.similarity_ @ interest
        scores[bought] = 0.0

        ranked = np.argsort(-scores, kind="stable")[:top_k]
        recs = [
            {
                "product_id": str(self.product_ids_[i]),
                "description": self._describe(i),
                "score": round(float(scores[i]), 4),
                "because": self._because(i, bought),
                "source": "personalised",
            }
            for i in ranked
            if scores[i] > 0
        ]
        # Thin histories can run out of similar items, so top up with best-sellers.
        chosen = {r["product_id"] for r in recs}
        lookup = {str(p): i for i, p in enumerate(self.product_ids_)}
        for product_id in self.popular_unbought(customer_id, top_k + len(chosen)):
            if len(recs) == top_k:
                break
            if product_id not in chosen:
                recs.append({
                    "product_id": product_id,
                    "description": self._describe(lookup[product_id]),
                    "score": 0.0,
                    "because": None,
                    "source": "popularity",
                })
        return recs


def time_split(
    transactions: pd.DataFrame, cutoff: pd.Timestamp, window_days: int
) -> Tuple[pd.DataFrame, Dict[float, Set[str]]]:
    """Purchases before ``cutoff``, and the new products each customer then bought."""
    purchases = _purchases(transactions)
    train = purchases[purchases["invoice_date"] < cutoff]
    end = cutoff + pd.Timedelta(days=window_days)
    future = purchases[(purchases["invoice_date"] >= cutoff) & (purchases["invoice_date"] < end)]

    bought_before = train.groupby("customer_id")["stock_code"].agg(lambda s: set(s.astype(str)))
    bought_after = future.groupby("customer_id")["stock_code"].agg(lambda s: set(s.astype(str)))
    targets: Dict[float, Set[str]] = {}
    for customer_id, products in bought_after.items():
        if customer_id in bought_before.index:
            new_products = products - bought_before[customer_id]
            if new_products:
                targets[float(customer_id)] = new_products
    return train, targets


def ranking_metrics(recommended: List[str], relevant: Set[str], k: int = 10) -> Dict[str, float]:
    """Recall, hit rate and binary NDCG for one customer's ranked list."""
    top = recommended[:k]
    gains = [1.0 / np.log2(rank + 2) for rank, item in enumerate(top) if item in relevant]
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return {
        "recall": len(gains) / len(relevant),
        "hit_rate": float(bool(gains)),
        "ndcg": float(sum(gains) / ideal),
    }


def _score_policy(
    ranked_lists: Dict[float, List[str]], targets: Dict[float, Set[str]], k: int, catalogue: int
) -> Dict[str, float]:
    rows = [ranking_metrics(ranked_lists[c], relevant, k) for c, relevant in targets.items()]
    shown = {product for ranked in ranked_lists.values() for product in ranked[:k]}
    summary = {name: float(np.mean([r[name] for r in rows])) for name in ("recall", "hit_rate", "ndcg")}
    summary["catalogue_coverage"] = len(shown) / catalogue
    return summary


def evaluate_at_cutoff(
    transactions: pd.DataFrame, cutoff: pd.Timestamp, window_days: int, k: int, n_neighbours: int
) -> Dict[str, Any]:
    """Compare the recommender with best-sellers on purchases after ``cutoff``."""
    train, targets = time_split(transactions, cutoff, window_days)
    model = ItemCosineRecommender(n_neighbours=n_neighbours).fit(train)
    personalised = {c: [r["product_id"] for r in model.recommend(c, k)] for c in targets}
    popular = {c: model.popular_unbought(c, k) for c in targets}
    catalogue = len(model.product_ids_)
    return {
        "cutoff": str(cutoff.date()),
        "customers": len(targets),
        "k": k,
        "n_neighbours": n_neighbours,
        "recommender": _score_policy(personalised, targets, k, catalogue),
        "popularity": _score_policy(popular, targets, k, catalogue),
    }


def _artifact_dir() -> Path:
    config = load_config()
    return get_project_root() / config.get("paths", {}).get("models_artifacts_dir", "models_artifacts")


def save_model(model: ItemCosineRecommender, path: Path) -> None:
    joblib.dump(model, path)
    _load_model.cache_clear()


@lru_cache(maxsize=4)
def _load_model(path: str) -> Optional[ItemCosineRecommender]:
    if not Path(path).exists():
        logger.warning(f"No product recommender at {path}; run `python -m src.pipeline recommender`.")
        return None
    # Safe to unpickle: built by this pipeline and shipped with the repo, like the other artifacts.
    return joblib.load(path)


def recommend_for_customer(
    customer_id: float, top_k: int = 10, path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    model = _load_model(str(path or _artifact_dir() / MODEL_FILE))
    return [] if model is None else model.recommend(customer_id, top_k)


def run_product_recommender(transactions: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Select on validation, score the test cutoff once, then refit on all history."""
    config = load_config()
    temporal = config["features"]["temporal"]
    settings = config["recommendations"]
    window, k = temporal["churn_window_days"], settings["top_k"]
    validation = pd.Timestamp(temporal["snapshots"]["validation"])
    test = pd.Timestamp(temporal["snapshots"]["test"])

    if transactions is None:
        transactions = pd.read_parquet(get_project_root() / "data/interim/cleaned_transactions.parquet")

    candidates = [
        evaluate_at_cutoff(transactions, validation, window, k, n)
        for n in settings["neighbour_grid"]
    ]
    best = max(candidates, key=lambda r: r["recommender"]["recall"])
    n_neighbours = best["n_neighbours"]
    logger.info(f"Selected n_neighbours={n_neighbours} on validation recall@{k}")

    metrics = {
        "selection": candidates,
        "n_neighbours": n_neighbours,
        "validation": best,
        "test": evaluate_at_cutoff(transactions, test, window, k, n_neighbours),
    }
    art_dir = _artifact_dir()
    art_dir.mkdir(parents=True, exist_ok=True)
    save_model(ItemCosineRecommender(n_neighbours=n_neighbours).fit(transactions), art_dir / MODEL_FILE)
    (art_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
