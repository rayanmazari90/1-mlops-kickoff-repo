"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Standardizes data cleaning separately from feature engineering.
- Responsibility (separation of concerns): Handling missing values, standardizing formats, and removing pure noise.
- Pipeline contract (inputs and outputs): Takes a raw DataFrame, outputs a cleaned DataFrame ready for split and feature engineering.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
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
    - Prevents pipeline crashes caused by corrupt data and enforces a consistent state before modeling.
    """
    print("Cleaning raw dataframe...") # TODO: replace with logging later
    
    df_clean = df_raw.copy()
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Every dataset has unique quality issues (e.g., dropping empty rows, fixing typos, removing outliers)
    # Examples:
    # 1. df_clean = df_clean.dropna(subset=[target_column])
    # 2. df_clean['date'] = pd.to_datetime(df_clean['date'])
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    return df_clean