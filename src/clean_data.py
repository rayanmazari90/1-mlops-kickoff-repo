"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).

Educational Goal:
- Why this module exists in an MLOps system: Standardizes data cleaning
  separately from feature engineering.
- Responsibility (separation of concerns): Handling missing values,
  standardizing formats, and removing pure noise.
- Pipeline contract (inputs and outputs): Takes a raw DataFrame, outputs
  a cleaned DataFrame ready for split and feature engineering.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported
      from config.yml in a later session
"""

import pandas as pd


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Inputs:
    - df_raw: The uncleaned, raw pandas DataFrame.
    - target_column: The name of the target variable column.
    Outputs:
    - pd.DataFrame containing the cleaned data.
    Why this contract matters for reliable ML delivery:
    - Prevents pipeline crashes caused by corrupt data and enforces
      a consistent state before modeling.
    """
    print("Cleaning raw dataframe...")  # TODO: replace with logging later

    df_clean = df_raw.copy()

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # Standardize column names
    df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(" ", "_")

    # Remove duplicates
    df_clean = df_clean.drop_duplicates()

    # Drop rows where critical columns are missing
    critical_cols = [target_column, "tourney_date", "surface", "winner_id", "loser_id"]
    existing_cols = [col for col in critical_cols if col in df_clean.columns]
    if existing_cols:
        df_clean = df_clean.dropna(subset=existing_cols)

    # Handle tourney_date
    if "tourney_date" in df_clean.columns:
        # Convert to datetime and coerce errors to NaT
        df_clean["tourney_date"] = pd.to_datetime(
            df_clean["tourney_date"], format="%Y%m%d", errors="coerce"
        )
        # Drop any dates that failed to parse (NaT)
        df_clean = df_clean.dropna(subset=["tourney_date"])

    # Handle missing ranks by imputing with explicit rule (999999 for unranked)
    for col in ["winner_rank", "loser_rank"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(999999.0)

    # Reset index after drops
    df_clean = df_clean.reset_index(drop=True)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return df_clean
