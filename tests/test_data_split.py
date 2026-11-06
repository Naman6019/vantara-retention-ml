import joblib
import pandas as pd
import pytest

from src.models import data_split
from src.models.data_split import (
    CUTOFF_COL,
    FEATURE_COLS,
    ID_COL,
    assign_roles,
    load_processed_feature_table,
    refit_data,
    split_data,
)


def test_assign_roles_partitions_by_cutoff(feature_table):
    roles = assign_roles(feature_table)
    assert pd.Timestamp("2011-01-01") in roles["train"]
    assert pd.Timestamp("2011-02-01") in roles["train"]
    assert roles["val"] == [pd.Timestamp("2011-06-01")]
    assert roles["test"] == [pd.Timestamp("2011-09-01")]


def test_time_split_partitions_sum_to_total(feature_table):
    data = split_data(feature_table, save_scaler=False)
    total = len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"])
    assert total == len(feature_table)
    assert data["X_train"].shape[1] == len(FEATURE_COLS)


def test_time_split_is_chronological(feature_table):
    data = split_data(feature_table, save_scaler=False)
    train_max = pd.to_datetime(data["cutoff_train"]).max()
    val_cut = pd.to_datetime(data["cutoff_val"]).max()
    test_cut = pd.to_datetime(data["cutoff_test"]).max()
    # Training cutoffs strictly precede validation, which precedes the test cutoff.
    assert train_max < val_cut < test_cut


def test_training_label_window_ends_before_validation_cutoff(feature_table):
    """Leakage guard (plan D1): every train label window must close before val."""
    data = split_data(feature_table, save_scaler=False)
    window_days = 90
    train_label_end = pd.to_datetime(data["cutoff_train"]).max() + pd.Timedelta(days=window_days)
    val_cutoff = pd.to_datetime(data["cutoff_val"]).min()
    assert train_label_end <= val_cutoff


def test_validation_label_window_ends_before_test_cutoff(feature_table):
    data = split_data(feature_table, save_scaler=False)
    window_days = 90
    val_label_end = pd.to_datetime(data["cutoff_val"]).max() + pd.Timedelta(days=window_days)
    test_cutoff = pd.to_datetime(data["cutoff_test"]).min()
    assert val_label_end <= test_cutoff


def test_no_duplicate_customer_cutoff_pairs(feature_table):
    pairs = feature_table.groupby([ID_COL, CUTOFF_COL]).size()
    assert (pairs == 1).all()


def test_scaler_fitted_on_train_only(feature_table):
    data = split_data(feature_table, save_scaler=False)
    # Train features standardised with train-only statistics -> mean ~ 0.
    assert abs(data["X_train_scaled"].mean().mean()) < 0.2
    # Scaler mean equals the training-only mean of an unscaled column.
    raw_mean = feature_table.loc[
        feature_table[CUTOFF_COL] < pd.Timestamp("2011-06-01"), "recency"
    ].mean()
    idx = FEATURE_COLS.index("recency")
    assert data["scaler"].mean_[idx] == pytest.approx(raw_mean)


def test_split_requires_cutoff_column(feature_table):
    with pytest.raises(ValueError, match="cutoff_date"):
        split_data(feature_table.drop(columns=[CUTOFF_COL]), save_scaler=False)


def test_refit_uses_all_snapshots_up_to_refit_end(feature_table):
    refit = refit_data(feature_table)
    # refit_end is 2011-06-01, so train + validation cutoffs are included, test is not.
    assert len(refit["X"]) == len(feature_table) - (feature_table[CUTOFF_COL] == pd.Timestamp("2011-09-01")).sum()


def test_split_data_saves_scaler(feature_table, monkeypatch, tmp_path):
    monkeypatch.setattr(data_split, "get_project_root", lambda: tmp_path)
    split_data(feature_table, save_scaler=True)
    scaler_path = tmp_path / "models_artifacts/scaler.joblib"
    assert scaler_path.exists()
    assert hasattr(joblib.load(scaler_path), "transform")


def test_load_processed_feature_table_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(data_split, "get_project_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="Processed feature table not found"):
        load_processed_feature_table()


def test_load_processed_feature_table_reads_parquet(feature_table, monkeypatch, tmp_path):
    monkeypatch.setattr(data_split, "get_project_root", lambda: tmp_path)
    proc = tmp_path / "data/processed"
    proc.mkdir(parents=True)
    feature_table.to_parquet(proc / "customer_features.parquet")

    loaded = load_processed_feature_table()
    assert len(loaded) == len(feature_table)
