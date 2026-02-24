"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Acts as a quality gate before expensive model training occurs.
- Responsibility (separation of concerns): Validating schemas, checking for data drift, and ensuring required columns exist.
- Pipeline contract (inputs and outputs): Takes a DataFrame and requirements; returns a boolean or raises an Exception.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df: The DataFrame to validate.
    - required_columns: List of strings representing columns that must be present.
    Outputs:
    - bool: True if valid, otherwise raises an exception.
    Why this contract matters for reliable ML delivery:
    - Fails fast. It is cheaper and safer to fail immediately rather than deploying a broken model trained on broken data.
    """
    print("Validating dataframe schema and constraints...") # TODO: replace with logging later
    
    if df.empty:
        raise ValueError("Validation failed: DataFrame is completely empty.")
        
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Validation failed: Missing required columns: {missing_cols}")
        
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Different models have strict requirements regarding null thresholds or value ranges
    # Examples:
    # 1. assert df['age'].min() >= 0, "Age cannot be negative"
    # 2. assert df.isnull().sum().max() == 0, "No nulls allowed in this dataset"
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    return True