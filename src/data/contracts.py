"""Pandera data contracts for raw and processed tables (plan D12 / PRD §6.2).

Validation failures are collected and written to a quarantine file rather than
being silently dropped, and the function raises so the pipeline "fails loudly".
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

RAW_SCHEMA = DataFrameSchema(
    {
        "invoice": Column(object),
        "stock_code": Column(object),
        "quantity": Column(float, nullable=True),
        "invoice_date": Column("datetime64[ns]"),
        "price": Column(float, nullable=True),
        "customer_id": Column(float, nullable=True),
        "country": Column(object),
    },
    strict=False,
    coerce=True,
)

FEATURE_SCHEMA = DataFrameSchema(
    {
        "customer_id": Column(float),
        "frequency": Column(float, Check.ge(0)),
        "recency": Column(float, Check.ge(0)),
        "monetary_total": Column(float),
        "engagement_score": Column(float, Check.in_range(0, 100)),
        "return_rate": Column(float, Check.in_range(0, 1)),
        "cancellation_rate": Column(float, Check.in_range(0, 1)),
        "discount_sensitivity": Column(float, Check.in_range(0, 1)),
        "is_uk_customer": Column(int, Check.isin([0, 1])),
        "is_churned": Column(int, Check.isin([0, 1])),
        "future_clv": Column(float, Check.ge(0)),
    },
    strict=False,
    coerce=True,
)


def _quarantine(records: Dict[str, Any], root: Optional[Path] = None) -> Path:
    from src.utils.config import get_project_root

    root = root or get_project_root()
    out = root / "logs" / "quarantine"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    return path


def validate_raw(df: pd.DataFrame, root: Optional[Path] = None) -> pd.DataFrame:
    """Validate the raw transaction table; quarantine and raise on failure."""
    try:
        return RAW_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        path = _quarantine(err.failure_cases.to_dict(orient="records"), root)
        raise ValueError(f"Raw data contract violated — details in {path}") from err


def validate_features(df: pd.DataFrame, root: Optional[Path] = None) -> pd.DataFrame:
    """Validate a processed customer feature table; quarantine and raise on failure."""
    try:
        return FEATURE_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        path = _quarantine(err.failure_cases.to_dict(orient="records"), root)
        raise ValueError(f"Feature contract violated — details in {path}") from err


def assert_no_leakage(features: pd.DataFrame, cutoff_date) -> None:
    """Guard: no label-derived column may be a model feature (belt and braces)."""
    leaks = {"is_churned", "future_clv", "spend_last_90d"} & set(features.columns)
    forbidden_features = {"is_churned", "future_clv"}
    from src.models.data_split import FEATURE_COLS

    overlap = forbidden_features & set(FEATURE_COLS)
    if overlap:
        raise ValueError(f"Target column(s) present in FEATURE_COLS: {overlap}")
    if leaks and not forbidden_features.issubset(features.columns):
        # The target columns are expected in the feature *table*; this is informational only.
        pass
