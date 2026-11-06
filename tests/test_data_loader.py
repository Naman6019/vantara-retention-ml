import io
import zipfile

import pandas as pd
import pytest

from src.data import loader

CONFIG = {
    "paths": {
        "raw_dir": "data/raw",
        "raw_file": "data/raw/online_retail_II.xlsx",
        "raw_zip": "data/raw/online_retail_ii.zip",
    },
    "dataset": {
        "download_url": "https://example.com/online_retail_ii.zip",
        "sheets": ["Year 2009-2010", "Year 2010-2011"],
        "column_mapping": {
            "Invoice": "invoice",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "price",
            "Customer ID": "customer_id",
            "Country": "country",
        },
    },
}


def _patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(loader, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(loader, "load_config", lambda: CONFIG)


def _make_zip(*names) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name in names:
            archive.writestr(name, b"dummy-bytes")
    return buf.getvalue()


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1024 * 1024):
        yield self.payload


def test_download_raw_dataset_returns_existing_file(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    excel = tmp_path / "data/raw/online_retail_II.xlsx"
    excel.parent.mkdir(parents=True)
    excel.write_bytes(b"existing")

    def fail_get(*args, **kwargs):
        raise AssertionError("requests.get should not be called when file exists")

    monkeypatch.setattr(loader.requests, "get", fail_get)
    result = loader.download_raw_dataset(force=False)
    assert result == excel


def test_download_raw_dataset_force_download(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    payload = _make_zip("online_retail_II.xlsx")
    monkeypatch.setattr(loader.requests, "get", lambda *a, **k: FakeResponse(payload))

    result = loader.download_raw_dataset(force=True)
    assert result.name == "online_retail_II.xlsx"
    assert result.exists()


def test_download_raw_dataset_missing_xlsx_raises(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    payload = _make_zip("readme.txt")
    monkeypatch.setattr(loader.requests, "get", lambda *a, **k: FakeResponse(payload))

    with pytest.raises(FileNotFoundError, match="\\.xlsx not found"):
        loader.download_raw_dataset(force=True)


def test_load_raw_data_uses_parquet_cache(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True)
    cache = raw_dir / "online_retail_II_combined.parquet"
    pd.DataFrame({"invoice": ["1", "2"], "value": [3, 4]}).to_parquet(cache)

    df = loader.load_raw_data()
    assert len(df) == 2
    assert list(df.columns) == ["invoice", "value"]


def test_load_raw_data_from_excel_standardizes_schema(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "online_retail_II.xlsx").write_bytes(b"dummy")

    source = pd.DataFrame(
        {
            "Invoice": ["1002", "1001"],
            "StockCode": ["B", "A"],
            "Description": ["Y", "X"],
            "Quantity": [2, 1],
            "InvoiceDate": ["2011-02-01", "2011-01-01"],
            "Price": [2.0, 1.0],
            "Customer ID": [2.0, 1.0],
            "Country": ["France", "United Kingdom"],
        }
    )
    monkeypatch.setattr(loader.pd, "read_excel", lambda *a, **k: source.copy())

    df = loader.load_raw_data(use_cache=False)
    expected = [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "price",
        "customer_id",
        "country",
    ]
    assert list(df.columns) == expected
    assert df["invoice_date"].is_monotonic_increasing
    assert df["quantity"].dtype.kind in "if"
    assert df["customer_id"].dtype.kind == "f"
