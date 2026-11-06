import pandas as pd
import torch

from src.models import dl_models
from src.models.dl_models import (
    ChurnANN,
    PurchaseLSTM,
    SpendingAutoencoder,
    prepare_lstm_sequences,
    train_churn_ann,
    train_purchase_lstm,
    train_spending_autoencoder,
)


def test_churn_ann_forward_shape():
    model = ChurnANN(input_dim=8, hidden_dim1=16, hidden_dim2=8)
    out = model(torch.randn(5, 8))
    assert out.shape == (5, 1)


def test_purchase_lstm_forward_shape():
    model = PurchaseLSTM(input_dim=3, hidden_dim=8)
    out = model(torch.randn(4, 6, 3))
    assert out.shape == (4, 1)


def test_spending_autoencoder_reconstruction():
    model = SpendingAutoencoder(input_dim=10, latent_dim=3)
    x = torch.randn(6, 10)
    recon = model(x)
    assert recon.shape == x.shape


def test_train_churn_ann(monkeypatch, tmp_path, split_result):
    monkeypatch.setattr(dl_models, "split_data", lambda: split_result)
    monkeypatch.setattr(dl_models, "get_project_root", lambda: tmp_path)
    (tmp_path / "models_artifacts").mkdir(parents=True)

    result = train_churn_ann(
        epochs=2, batch_size=32, lr=0.01, patience=2, save_artifacts=True
    )

    assert "model" in result and "metrics" in result
    metrics = result["metrics"]
    assert 0.0 <= metrics["Test ROC-AUC"] <= 1.0
    assert (tmp_path / "models_artifacts/ann_churn_model.pt").exists()


def test_train_spending_autoencoder(monkeypatch, tmp_path, split_result):
    monkeypatch.setattr(dl_models, "split_data", lambda: split_result)
    monkeypatch.setattr(dl_models, "get_project_root", lambda: tmp_path)
    (tmp_path / "models_artifacts").mkdir(parents=True)

    result = train_spending_autoencoder(epochs=2, batch_size=32, save_artifacts=True)

    assert "threshold" in result
    assert result["threshold"] >= 0.0
    assert len(result["errors"]) == len(split_result["X_train"]) + len(
        split_result["X_val"]
    ) + len(split_result["X_test"])
    assert (tmp_path / "models_artifacts/autoencoder_model.pt").exists()
    assert (tmp_path / "models_artifacts/autoencoder_threshold.joblib").exists()


def _interim_transactions() -> pd.DataFrame:
    rows = []
    for cid in range(1, 61):
        for i, day in enumerate(["2011-05-01", "2011-06-01", "2011-07-01"]):
            rows.append(
                {
                    "invoice": f"{cid}00{i}",
                    "stock_code": "SKU",
                    "description": "ITEM",
                    "quantity": 2,
                    "price": 5.0,
                    "customer_id": float(cid),
                    "invoice_date": pd.Timestamp(day),
                    "line_total": 10.0,
                    "is_cancellation": False,
                    "country": "United Kingdom",
                }
            )
    # Half the customers retain after cutoff (balanced churn label)
    for cid in range(1, 31):
        rows.append(
            {
                "invoice": f"{cid}99",
                "stock_code": "SKU",
                "description": "ITEM",
                "quantity": 1,
                "price": 5.0,
                "customer_id": float(cid),
                "invoice_date": pd.Timestamp("2011-10-01"),
                "line_total": 5.0,
                "is_cancellation": False,
                "country": "United Kingdom",
            }
        )
    return pd.DataFrame(rows)


def test_prepare_lstm_sequences(monkeypatch, tmp_path):
    monkeypatch.setattr(dl_models, "get_project_root", lambda: tmp_path)
    interim = tmp_path / "data/interim"
    interim.mkdir(parents=True)
    _interim_transactions().to_parquet(interim / "cleaned_transactions.parquet")

    X, y = prepare_lstm_sequences(max_seq_len=4)
    assert X.shape[1:] == (4, 3)
    assert y.shape[1] == 1
    assert X.shape[0] == y.shape[0]
    assert set(torch.unique(y).tolist()).issubset({0.0, 1.0})


def test_train_purchase_lstm(monkeypatch, tmp_path):
    monkeypatch.setattr(dl_models, "get_project_root", lambda: tmp_path)
    interim = tmp_path / "data/interim"
    interim.mkdir(parents=True)
    (tmp_path / "models_artifacts").mkdir(parents=True)
    _interim_transactions().to_parquet(interim / "cleaned_transactions.parquet")

    result = train_purchase_lstm(epochs=2, batch_size=4, save_artifacts=True)

    assert result["Model"] == "Sequence LSTM"
    assert 0.0 <= result["Test ROC-AUC"] <= 1.0
    assert (tmp_path / "models_artifacts/lstm_churn_model.pt").exists()
