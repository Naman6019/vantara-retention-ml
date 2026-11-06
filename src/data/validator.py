from typing import Any, Dict, Tuple

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("data_validator")

REQUIRED_COLUMNS = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]


def validate_raw_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates the raw transaction dataset against predefined quality thresholds:
    - Schema conformity
    - Date range consistency
    - Critical column null-rate limits
    - Minimum sample size

    Args:
        df: Raw combined DataFrame.

    Returns:
        Tuple of (passed: bool, validation_report: dict).
        Raises ValueError if any critical check fails.
    """
    report: Dict[str, Any] = {
        "total_records": len(df),
        "checks_passed": [],
        "checks_failed": [],
        "warnings": [],
    }

    # 1. Schema check
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        msg = f"Schema validation failed: missing columns {missing_cols}"
        report["checks_failed"].append(msg)
        logger.error(msg)
        raise ValueError(msg)
    report["checks_passed"].append("Schema contains all required columns.")

    # 2. Minimum volume check
    if len(df) < 500_000:
        msg = f"Row count check failed: Expected >= 500k rows, found {len(df):,}"
        report["checks_failed"].append(msg)
        logger.error(msg)
        raise ValueError(msg)
    report["checks_passed"].append(f"Row count validation passed: {len(df):,} records.")

    # 3. Critical column null checks
    for col in ["invoice", "stock_code", "quantity", "price", "invoice_date"]:
        null_count = df[col].isna().sum()
        if null_count > 0:
            msg = f"Integrity check failed: Column '{col}' contains {null_count} nulls (expected 0)."
            report["checks_failed"].append(msg)
            logger.error(msg)
            raise ValueError(msg)
    report["checks_passed"].append("Zero nulls verified in critical transaction fields.")

    # 4. Customer ID missing rate check
    cust_id_null_pct = df["customer_id"].isna().mean() * 100
    report["customer_id_missing_pct"] = round(cust_id_null_pct, 2)
    if cust_id_null_pct > 35.0:
        msg = f"Customer ID missing rate excessive: {cust_id_null_pct:.2f}% (threshold 35%)."
        report["checks_failed"].append(msg)
        logger.error(msg)
        raise ValueError(msg)
    report["checks_passed"].append(
        f"Customer ID missing rate within tolerance: {cust_id_null_pct:.2f}%."
    )

    # 5. Date range check
    min_date = df["invoice_date"].min()
    max_date = df["invoice_date"].max()
    report["date_range"] = {"min": str(min_date), "max": str(max_date)}
    if min_date.year < 2009 or max_date.year > 2012:
        msg = f"Unexpected date range: {min_date} to {max_date}"
        report["warnings"].append(msg)
        logger.warning(msg)
    else:
        report["checks_passed"].append(f"Date range valid: {min_date.date()} to {max_date.date()}.")

    # 6. Unique customer count check
    unique_customers = df["customer_id"].dropna().nunique()
    report["unique_customers"] = unique_customers
    if unique_customers < 4_000:
        msg = f"Unique customer count low: {unique_customers} (expected >= 4,000)."
        report["checks_failed"].append(msg)
        logger.error(msg)
        raise ValueError(msg)
    report["checks_passed"].append(f"Unique customers count valid: {unique_customers:,}.")

    logger.info("All raw data validation checks passed successfully.")
    return True, report
