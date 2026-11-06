import pandas as pd

from src.features.snapshots import build_snapshots, generate_cutoffs, save_snapshots


def _transactions(months: int = 8) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2011-01-01")
    for cid in range(1, 41):
        for m in range(months):
            date = start + pd.DateOffset(months=m)
            rows.append({
                "invoice": f"{cid}-{m}",
                "stock_code": "SKU",
                "description": "ITEM",
                "quantity": 2,
                "price": 5.0,
                "customer_id": float(cid),
                "invoice_date": date,
                "line_total": 10.0,
                "is_cancellation": False,
                "country": "United Kingdom",
            })
    return pd.DataFrame(rows)


def test_generate_cutoffs_includes_validation_and_test():
    cutoffs = generate_cutoffs({
        "train_start": "2010-06-01",
        "refit_end": "2011-06-01",
        "frequency": "MS",
        "validation": "2011-06-01",
        "test": "2011-09-01",
    })
    assert pd.Timestamp("2010-06-01") in cutoffs
    assert pd.Timestamp("2011-06-01") in cutoffs
    assert pd.Timestamp("2011-09-01") in cutoffs
    # monthly cadence
    assert (pd.Series(cutoffs).diff().dropna().dt.days > 0).all()


def test_build_snapshots_stacks_rows_with_cutoff_column():
    df = _transactions()
    cutoffs = [pd.Timestamp("2011-05-01"), pd.Timestamp("2011-06-01")]
    snaps, ref = build_snapshots(df, cutoffs=cutoffs)

    assert "cutoff_date" in snaps.columns
    assert snaps["cutoff_date"].nunique() == 2
    assert set(snaps["cutoff_date"].unique()) == set(cutoffs)
    assert {"customer_id", "is_churned", "future_clv", "engagement_score"}.issubset(snaps.columns)
    assert "recency" in ref  # engagement reference fitted
    # No (customer_id, cutoff_date) pair is duplicated.
    dupes = snaps.groupby(["customer_id", "cutoff_date"]).size()
    assert (dupes == 1).all()


def test_save_snapshots_writes_files(monkeypatch, tmp_path):
    import json

    df = _transactions()
    snaps, ref = build_snapshots(df, cutoffs=[pd.Timestamp("2011-05-01"), pd.Timestamp("2011-06-01")])
    out = save_snapshots(snaps, ref, root=tmp_path)
    assert (tmp_path / "data/processed/customer_snapshots.parquet").exists()
    assert (tmp_path / "models_artifacts/engagement_reference.json").exists()
    assert json.loads((tmp_path / "models_artifacts/engagement_reference.json").read_text()) == ref
    assert isinstance(out, str)
