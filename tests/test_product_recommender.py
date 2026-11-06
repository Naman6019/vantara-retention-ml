import pandas as pd
import pytest

from src.recommendations.products import (
    ItemCosineRecommender,
    evaluate_at_cutoff,
    ranking_metrics,
    recommend_for_customer,
    save_model,
    time_split,
)

BEFORE = "2011-01-15"
CUTOFF = pd.Timestamp("2011-02-01")
INSIDE_WINDOW = "2011-03-01"
AFTER_WINDOW = "2011-06-01"


def _tx(rows, cancelled=()):
    """Build transactions from (customer, product, date) rows."""
    df = pd.DataFrame(rows, columns=["customer_id", "stock_code", "invoice_date"])
    df["customer_id"] = df["customer_id"].astype(float)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["description"] = "ITEM " + df["stock_code"]
    df["is_cancellation"] = [
        (c, s) in cancelled for c, s in zip(df["customer_id"], df["stock_code"], strict=True)
    ]
    df["quantity"] = df["is_cancellation"].map({True: -1, False: 1})
    return df


def _two_taste_groups():
    """Group one buys A/B/C. A larger group buys X/Y/Z, so X is the best seller.

    Customer 100 has bought only A and B, so a personalised model should offer C
    even though best-seller logic would offer X.
    """
    rows = []
    for customer in range(1, 6):
        rows += [(customer, item, BEFORE) for item in "ABC"]
    for customer in range(10, 20):
        rows += [(customer, item, BEFORE) for item in "XYZ"]
    rows += [(100, "A", BEFORE), (100, "B", BEFORE)]
    return _tx(rows)


def _fit(transactions=None, n_neighbours=10):
    data = _two_taste_groups() if transactions is None else transactions
    return ItemCosineRecommender(n_neighbours=n_neighbours).fit(data)


def test_recommends_co_purchased_item_over_best_seller():
    top = _fit().recommend(100.0, top_k=1)
    assert top[0]["product_id"] == "C"
    assert top[0]["source"] == "personalised"


def test_recommendation_names_the_purchase_behind_it():
    top = _fit().recommend(100.0, top_k=1)[0]
    assert top["description"] == "ITEM C"
    assert top["because"] in {"ITEM A", "ITEM B"}


def test_never_recommends_something_already_bought():
    ids = [r["product_id"] for r in _fit().recommend(100.0, top_k=5)]
    assert not {"A", "B"} & set(ids)


def test_tops_up_with_best_sellers_when_similar_items_run_out():
    recs = _fit().recommend(100.0, top_k=2)
    assert [r["source"] for r in recs] == ["personalised", "popularity"]
    assert recs[1]["because"] is None


def test_unknown_customer_gets_no_personalised_recommendations():
    assert _fit().recommend(999.0, top_k=5) == []


def test_cancelled_lines_do_not_count_as_purchases():
    rows = [(c, item, BEFORE) for c in range(1, 4) for item in "AB"]
    rows += [(100, "A", BEFORE), (100, "B", BEFORE)]
    ids = [r["product_id"] for r in _fit(_tx(rows, cancelled={(100.0, "B")})).recommend(100.0, top_k=1)]
    assert ids == ["B"]


def test_time_split_trains_only_on_purchases_before_the_cutoff():
    train, _ = time_split(_tx([(1, "A", BEFORE), (1, "B", INSIDE_WINDOW)]), CUTOFF, 90)
    assert train["invoice_date"].max() < CUTOFF


def test_time_split_targets_are_new_products_inside_the_window():
    data = _tx([
        (1, "A", BEFORE),
        (1, "A", INSIDE_WINDOW),  # repeat purchase: not a discovery
        (1, "B", INSIDE_WINDOW),  # new and in window: the target
        (1, "C", AFTER_WINDOW),  # too late
    ])
    _, targets = time_split(data, CUTOFF, 90)
    assert targets == {1.0: {"B"}}


def test_time_split_skips_customers_with_no_earlier_history():
    _, targets = time_split(_tx([(1, "A", BEFORE), (2, "B", INSIDE_WINDOW)]), CUTOFF, 90)
    assert targets == {}


def test_ranking_metrics_on_a_hand_worked_example():
    metrics = ranking_metrics(["a", "b", "c"], {"b", "z"}, k=3)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["hit_rate"] == 1.0
    assert metrics["ndcg"] == pytest.approx(0.3869, abs=1e-4)


def test_ranking_metrics_only_count_the_top_k():
    metrics = ranking_metrics(["a", "b", "c"], {"c"}, k=2)
    assert metrics == {"recall": 0.0, "hit_rate": 0.0, "ndcg": 0.0}


def test_evaluation_beats_popularity_when_tastes_differ():
    future = _tx([(100, "C", INSIDE_WINDOW)])
    result = evaluate_at_cutoff(
        pd.concat([_two_taste_groups(), future]), CUTOFF, window_days=90, k=1, n_neighbours=10
    )
    assert result["customers"] == 1
    assert result["recommender"]["recall"] == 1.0
    assert result["popularity"]["recall"] == 0.0


def test_recommend_for_customer_uses_the_saved_model(tmp_path):
    path = tmp_path / "product_recommender.joblib"
    save_model(_fit(), path)
    assert recommend_for_customer(100.0, top_k=1, path=path)[0]["product_id"] == "C"


def test_recommend_for_customer_without_a_model_returns_nothing(tmp_path):
    assert recommend_for_customer(100.0, top_k=1, path=tmp_path / "missing.joblib") == []


def test_endpoint_serves_personalised_recommendations(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app

    trained = [{"product_id": "C", "description": "ITEM C", "score": 1.0, "because": "ITEM A", "source": "personalised"}]
    monkeypatch.setattr(
        "src.recommendations.products.recommend_for_customer", lambda customer_id, top_k=10: trained
    )
    with TestClient(app) as client:
        assert client.get("/customers/12347/recommendations").json() == trained


def test_shipped_model_serves_a_real_customer():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        rows = client.get("/customers/12347/recommendations").json()
    assert len(rows) == 10
    assert rows[0]["source"] == "personalised"


def _scored_database_has_purchases() -> bool:
    """The best-seller fallback reads purchases the batch scorer writes to the database."""
    from sqlalchemy import inspect, text

    from src.serving import db

    engine = db.ensure_engine()
    if engine is None or not inspect(engine).has_table("customer_products"):
        return False
    with engine.connect() as conn:
        return conn.execute(text("SELECT 1 FROM customer_products LIMIT 1")).first() is not None


def test_endpoint_labels_its_best_seller_fallback(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app

    if not _scored_database_has_purchases():
        pytest.skip("needs a database filled by the batch scorer (python -m src.pipeline all)")
    monkeypatch.setattr(
        "src.recommendations.products.recommend_for_customer", lambda customer_id, top_k=10: []
    )
    with TestClient(app) as client:
        rows = client.get("/customers/12347/recommendations").json()
    assert rows
    assert {row["source"] for row in rows} == {"popularity"}
