import pandas as pd
import pytest

from src.data.validator import REQUIRED_COLUMNS, validate_raw_data
from src.utils.config import get_project_root, load_config


def test_project_root_exists():
    root = get_project_root()
    assert root.exists()
    assert (root / "config" / "config.yaml").exists()


def test_load_config():
    config = load_config()
    assert "project" in config
    assert "paths" in config
    assert "dataset" in config
    assert config["project"]["name"] == "customer-behavior-prediction"


def test_validator_detects_missing_columns():
    invalid_df = pd.DataFrame({"invoice": ["123"], "quantity": [1]})
    with pytest.raises(ValueError, match="Schema validation failed"):
        validate_raw_data(invalid_df)


def test_validator_detects_small_dataset():
    # Construct a valid schema dataframe with too few rows
    data = {col: [1] for col in REQUIRED_COLUMNS}
    data["invoice_date"] = [pd.Timestamp("2010-01-01")]
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Row count check failed"):
        validate_raw_data(df)
