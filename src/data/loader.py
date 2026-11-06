import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("data_loader")


def download_raw_dataset(force: bool = False) -> Path:
    """
    Downloads the Online Retail II dataset archive from the UCI ML Repository
    and extracts online_retail_II.xlsx into data/raw/.

    Args:
        force: If True, re-downloads even if the file already exists.

    Returns:
        Path to the extracted Excel file.
    """
    root = get_project_root()
    config = load_config()

    raw_dir = root / config.get("paths", {}).get("raw_dir", "data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    excel_path = root / config.get("paths", {}).get("raw_file", "data/raw/online_retail_II.xlsx")
    zip_path = root / config.get("paths", {}).get("raw_zip", "data/raw/online_retail_ii.zip")
    url = config.get("dataset", {}).get(
        "download_url", "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
    )

    if excel_path.exists() and not force:
        logger.info(f"Raw Excel dataset already exists at: {excel_path}")
        return excel_path

    logger.info(f"Downloading raw dataset archive from: {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    logger.info(f"Saved archive to {zip_path}. Extracting...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)

    if not excel_path.exists():
        # Search for any extracted xlsx in raw_dir
        found_xlsx = list(raw_dir.glob("*.xlsx"))
        if found_xlsx:
            excel_path = found_xlsx[0]
        else:
            raise FileNotFoundError(f"Extraction completed but .xlsx not found in {raw_dir}")

    logger.info(f"Extracted dataset successfully to: {excel_path}")
    return excel_path


def load_raw_data(force_reload: bool = False, use_cache: bool = True) -> pd.DataFrame:
    """
    Loads both sheets of the Online Retail II dataset, standardizes columns,
    and returns a combined chronologically-ordered DataFrame. Caches to Parquet
    for near-instant subsequent loads.

    Args:
        force_reload: Re-read from Excel even if cached Parquet exists.
        use_cache: Cache combined output to Parquet.

    Returns:
        pd.DataFrame containing ~1.06M transaction records.
    """
    root = get_project_root()
    config = load_config()
    raw_dir = root / config.get("paths", {}).get("raw_dir", "data/raw")
    cache_parquet = raw_dir / "online_retail_II_combined.parquet"

    if use_cache and cache_parquet.exists() and not force_reload:
        logger.info(f"Loading cached raw dataset from: {cache_parquet}")
        df = pd.read_parquet(cache_parquet)
        logger.info(f"Loaded {len(df):,} rows from Parquet cache.")
        return df

    excel_path = root / config.get("paths", {}).get("raw_file", "data/raw/online_retail_II.xlsx")
    if not excel_path.exists():
        logger.info("Excel dataset not found locally. Triggering automated download...")
        excel_path = download_raw_dataset()

    sheets = config.get("dataset", {}).get("sheets", ["Year 2009-2010", "Year 2010-2011"])
    column_mapping = config.get("dataset", {}).get("column_mapping", {})

    dfs = []
    for sheet in sheets:
        logger.info(f"Reading sheet: '{sheet}' from {excel_path.name} (this may take a minute)...")
        sheet_df = pd.read_excel(excel_path, sheet_name=sheet)
        logger.info(f"Sheet '{sheet}' read successfully: {len(sheet_df):,} records.")
        dfs.append(sheet_df)

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined total raw records: {len(combined_df):,}")

    if column_mapping:
        combined_df.rename(columns=column_mapping, inplace=True)

    # Standardize types and column formatting
    combined_df["invoice"] = combined_df["invoice"].astype(str)
    combined_df["stock_code"] = combined_df["stock_code"].astype(str)
    combined_df["description"] = combined_df["description"].astype(str)
    combined_df["invoice_date"] = pd.to_datetime(combined_df["invoice_date"])
    combined_df["quantity"] = pd.to_numeric(combined_df["quantity"], errors="coerce")
    combined_df["price"] = pd.to_numeric(combined_df["price"], errors="coerce")
    combined_df["customer_id"] = pd.to_numeric(combined_df["customer_id"], errors="coerce")
    combined_df["country"] = combined_df["country"].astype(str)

    # Sort chronologically
    combined_df.sort_values(by="invoice_date", ascending=True, inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    if use_cache:
        logger.info(f"Caching combined DataFrame to Parquet: {cache_parquet}")
        combined_df.to_parquet(cache_parquet, index=False)

    return combined_df


if __name__ == "__main__":
    logger.info("Starting dataset loader script...")
    df = load_raw_data()
    print("\n--- Dataset Summary ---")
    print(f"Total Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date Range: {df['invoice_date'].min()} to {df['invoice_date'].max()}")
    print(f"Unique Customers: {df['customer_id'].nunique():,}")
    print(f"Missing Customer ID rows: {df['customer_id'].isna().sum():,}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
    print("\nHead Preview:")
    print(df.head(5))
