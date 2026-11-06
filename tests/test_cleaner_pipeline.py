import pandas as pd

from src.data import cleaner


def _raw_frame():
    return pd.DataFrame(
        {
            "invoice": ["1001", "1002", "1002", "POST1"],
            "stock_code": ["A", "B", "B", "POST"],
            "description": [" item a ", "item b", "item b", "postage"],
            "quantity": [2, 1, 1, 1],
            "price": [10.0, 5.0, 5.0, 3.0],
            "customer_id": [1.0, 2.0, 2.0, None],
            "invoice_date": pd.to_datetime(
                ["2011-01-01", "2011-02-01", "2011-02-01", "2011-03-01"]
            ),
            "country": ["United Kingdom", "France", "France", "United Kingdom"],
        }
    )


def test_standardize_descriptions():
    df = pd.DataFrame({"description": ["  widget ", "Gadget"]})
    out = cleaner.standardize_descriptions(df)
    assert out["description"].tolist() == ["WIDGET", "GADGET"]


def test_run_cleaning_pipeline_without_saving(monkeypatch):
    monkeypatch.setattr(cleaner, "load_config", lambda: {})
    out = cleaner.run_cleaning_pipeline(_raw_frame(), save_interim=False)

    assert "line_total" in out.columns
    assert out["customer_id"].isna().sum() == 0
    assert out["stock_code"].str.upper().isin(["POST"]).sum() == 0
    assert out["is_cancellation"].dtype == bool
    # duplicate exact row for invoice 1002 collapsed
    assert len(out) == 2


def test_run_cleaning_pipeline_saves_interim(monkeypatch, tmp_path):
    monkeypatch.setattr(cleaner, "load_config", lambda: {})
    monkeypatch.setattr(cleaner, "get_project_root", lambda: tmp_path)

    out = cleaner.run_cleaning_pipeline(_raw_frame(), save_interim=True)
    interim = tmp_path / "data/interim/cleaned_transactions.parquet"
    assert interim.exists()
    reloaded = pd.read_parquet(interim)
    assert len(reloaded) == len(out)
    assert "line_total" in reloaded.columns
