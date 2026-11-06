import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.data_split import FEATURE_COLS


@pytest.fixture
def tmp_path():
    """Isolated temporary directory that avoids the shared pytest basetemp tree."""
    path = Path(tempfile.mkdtemp(prefix="internmo-test-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def make_feature_table(n: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    data = {}
    for i, col in enumerate(FEATURE_COLS):
        data[col] = rng.uniform(1.0 + i, 50.0 + i * 3, n)
    # Keep bounded columns within their valid ranges
    data["return_rate"] = rng.uniform(0.0, 0.4, n)
    data["cancellation_rate"] = rng.uniform(0.0, 0.3, n)
    data["seasonal_concentration"] = rng.uniform(0.0, 1.0, n)
    data["discount_sensitivity"] = rng.uniform(0.0, 1.0, n)
    data["is_uk_customer"] = rng.randint(0, 2, n)
    data["engagement_score"] = rng.uniform(0.0, 100.0, n)
    data["purchase_frequency_trend"] = rng.uniform(-1.0, 1.0, n)

    df = pd.DataFrame(data)
    df["customer_id"] = np.arange(10_000, 10_000 + n, dtype=float)
    df["is_churned"] = rng.randint(0, 2, n)
    df["future_clv"] = np.where(
        df["is_churned"] == 1, 0.0, rng.uniform(0.0, 2000.0, n)
    )
    # Auxiliary momentum baseline for the value model (not a model feature).
    df["spend_last_90d"] = rng.uniform(0.0, 800.0, n)
    # Stacked snapshots: two training cutoffs, one validation, one test cutoff.
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    cutoffs = (
        [pd.Timestamp("2011-01-01")] * (n_train // 2)
        + [pd.Timestamp("2011-02-01")] * (n_train - n_train // 2)
        + [pd.Timestamp("2011-06-01")] * n_val
        + [pd.Timestamp("2011-09-01")] * (n - n_train - n_val)
    )
    df["cutoff_date"] = pd.to_datetime(cutoffs)
    return df


@pytest.fixture
def feature_table():
    return make_feature_table()


@pytest.fixture
def split_result(feature_table):
    from src.models.data_split import split_data

    return split_data(feature_table, save_scaler=False)
