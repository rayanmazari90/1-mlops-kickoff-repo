"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Inputs:
    - df_raw: The uncleaned, raw pandas DataFrame.
    - target_column: The name of the target variable column.
    Outputs:
    - pd.DataFrame containing the cleaned data.
    """
    logger.info("Cleaning raw dataframe ...")

    df_clean = df_raw.copy()
    initial_rows = len(df_clean)

    df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(" ", "_")

    df_clean = df_clean.drop_duplicates()

    critical_cols = [target_column, "tourney_date", "surface", "winner_id", "loser_id"]
    existing_cols = [col for col in critical_cols if col in df_clean.columns]
    if existing_cols:
        df_clean = df_clean.dropna(subset=existing_cols)

    if "tourney_date" in df_clean.columns:
        df_clean["tourney_date"] = pd.to_datetime(
            df_clean["tourney_date"], format="%Y%m%d", errors="coerce"
        )
        df_clean = df_clean.dropna(subset=["tourney_date"])

    for col in ["winner_rank", "loser_rank"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(999999.0)

    df_clean = df_clean.reset_index(drop=True)

    logger.info(
        "Cleaning complete: %d -> %d rows (dropped %d)",
        initial_rows,
        len(df_clean),
        initial_rows - len(df_clean),
    )

    return df_clean
