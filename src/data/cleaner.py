from typing import List, Optional

import pandas as pd

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("data_cleaner")


def clean_missing_customers(df: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
    """
    Identifies rows with missing Customer IDs.
    Drops or flags them based on configuration.
    """
    initial_count = len(df)
    missing_mask = df["customer_id"].isna()
    missing_count = missing_mask.sum()

    logger.info(f"Transactions missing Customer ID: {missing_count:,} ({missing_count / initial_count:.2%})")

    if drop:
        df_cleaned = df[~missing_mask].copy()
        logger.info(f"Dropped missing customer rows. Remaining: {len(df_cleaned):,}")
        return df_cleaned
    else:
        df_flagged = df.copy()
        df_flagged["is_guest_customer"] = missing_mask
        return df_flagged


def deduplicate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes exact duplicate rows across all fields while preserving legitimate
    repeat transactions.
    """
    initial_count = len(df)
    df_dedup = df.drop_duplicates().copy()
    duplicates_removed = initial_count - len(df_dedup)
    logger.info(f"Deduplication: removed {duplicates_removed:,} exact duplicate rows.")
    return df_dedup


def filter_admin_codes(df: pd.DataFrame, excluded_codes: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filters out non-product administrative stock codes (e.g. POST, BANK CHARGES, MANUAL).
    """
    if excluded_codes is None:
        config = load_config()
        excluded_codes = config.get("data_cleaning", {}).get(
            "exclude_admin_stock_codes", ["POST", "D", "M", "BANK CHARGES", "PADS", "DOT", "CR"]
        )

    initial_count = len(df)
    # Match exact or case-insensitive starts
    excluded_set = {c.strip().upper() for c in excluded_codes}
    mask = df["stock_code"].str.strip().str.upper().isin(excluded_set)

    df_filtered = df[~mask].copy()
    filtered_count = initial_count - len(df_filtered)
    logger.info(f"Administrative stock code filtering: removed {filtered_count:,} rows. Remaining: {len(df_filtered):,}")
    return df_filtered


def flag_returns_and_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicitly flags returns and cancellations (Quantity < 0 or Invoice starts with 'C')
    rather than discarding them, as return behavior is an important churn signal.
    """
    df_result = df.copy()
    df_result["is_cancellation"] = (
        df_result["invoice"].str.startswith("C", na=False) | (df_result["quantity"] < 0)
    )
    cancellations_count = df_result["is_cancellation"].sum()
    logger.info(f"Flagged {cancellations_count:,} returns/cancellations ({cancellations_count / len(df_result):.2%}).")
    return df_result


def clean_outliers_and_prices(
    df: pd.DataFrame,
    qty_min: int = -10000,
    qty_max: int = 10000,
    price_min: float = 0.001,
    price_max: float = 5000.0,
) -> pd.DataFrame:
    """
    Cleans extreme data-entry errors and non-positive prices for non-cancellation rows.
    """
    initial_count = len(df)

    # Filter price: for regular sales, price must be positive
    valid_price = (df["price"] >= price_min) & (df["price"] <= price_max)

    # Filter quantity bounds
    valid_qty = (df["quantity"] >= qty_min) & (df["quantity"] <= qty_max)

    df_cleaned = df[valid_price & valid_qty].copy()
    removed = initial_count - len(df_cleaned)
    logger.info(f"Outlier & invalid price filtering: removed {removed:,} rows. Remaining: {len(df_cleaned):,}")
    return df_cleaned


def standardize_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes descriptions: uppercase, strip whitespace.
    """
    df_clean = df.copy()
    df_clean["description"] = df_clean["description"].str.strip().str.upper()
    return df_clean


def run_cleaning_pipeline(df: pd.DataFrame, save_interim: bool = True) -> pd.DataFrame:
    """
    Executes the end-to-end data cleaning pipeline and saves the interim dataset.

    Args:
        df: Raw DataFrame from loader.
        save_interim: If True, writes cleaned data to data/interim/cleaned_transactions.parquet.

    Returns:
        Cleaned transaction-level DataFrame ready for feature engineering.
    """
    logger.info(f"Starting data cleaning pipeline on {len(df):,} raw records...")

    config = load_config()
    cleaning_cfg = config.get("data_cleaning", {})
    drop_cust = cleaning_cfg.get("drop_missing_customer_id", True)
    admin_codes = cleaning_cfg.get("exclude_admin_stock_codes", [])
    outlier_cfg = cleaning_cfg.get("outliers", {})

    # 1. Missing customer handling
    df_clean = clean_missing_customers(df, drop=drop_cust)

    # 2. Exact deduplication
    df_clean = deduplicate_transactions(df_clean)

    # 3. Filter administrative codes
    df_clean = filter_admin_codes(df_clean, excluded_codes=admin_codes)

    # 4. Flag returns / cancellations
    df_clean = flag_returns_and_cancellations(df_clean)

    # 5. Outliers and price validity
    df_clean = clean_outliers_and_prices(
        df_clean,
        qty_min=outlier_cfg.get("quantity_min", -10000),
        qty_max=outlier_cfg.get("quantity_max", 10000),
        price_min=outlier_cfg.get("price_min", 0.001),
        price_max=outlier_cfg.get("price_max", 5000.0),
    )

    # 6. Description standardization
    df_clean = standardize_descriptions(df_clean)

    # Compute line total (spend)
    df_clean["line_total"] = df_clean["quantity"] * df_clean["price"]

    df_clean.reset_index(drop=True, inplace=True)
    logger.info(f"Cleaning complete! Final cleaned transaction count: {len(df_clean):,}")

    if save_interim:
        root = get_project_root()
        interim_dir = root / config.get("paths", {}).get("interim_dir", "data/interim")
        interim_dir.mkdir(parents=True, exist_ok=True)
        interim_path = interim_dir / "cleaned_transactions.parquet"
        logger.info(f"Saving cleaned dataset to: {interim_path}")
        df_clean.to_parquet(interim_path, index=False)

    return df_clean
