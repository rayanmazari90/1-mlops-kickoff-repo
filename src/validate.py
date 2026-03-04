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

def validate_dataframe(df: pd.DataFrame, config: dict) -> bool:
    """
    Inputs:
    - df: The DataFrame to validate.
    - config: Dictionary containing the 'schema' configuration (required columns, allowed surfaces, etc.).
    Outputs:
    - bool: True if valid, otherwise raises ValueError.
    Why this contract matters for reliable ML delivery:
    - Fails fast. It is cheaper and safer to fail immediately rather than deploying a broken model trained on broken data.
    """
    print("Validating dataframe schema and constraints...") # TODO: replace with logging later
    
    if df.empty:
        raise ValueError("Validation failed: DataFrame is completely empty.")
        
    schema = config.get("schema", {})
    required_columns = schema.get("required_columns", [])
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Validation failed: Missing required columns: {missing_cols}")
        
    # Check non-null columns constraints
    non_null_cols = schema.get("non_null_columns", [])
    for col in non_null_cols:
        if col in df.columns and df[col].isnull().any():
            raise ValueError(f"Validation failed: Column {col} contains null values.")
            
    # Check tourney_date parseable
    if 'tourney_date' in df.columns:
        parsed_dates = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
        # If it was originally not null, but coercion resulted in NaT, it's unparseable
        invalid_dates = df['tourney_date'].notnull() & parsed_dates.isnull()
        if invalid_dates.any():
            raise ValueError("Validation failed: tourney_date contains invalid date formats.")

    # Check allowed surfaces
    allowed_surfaces = schema.get("allowed_surfaces", ["Hard", "Clay", "Grass", "Carpet"])
    if 'surface' in df.columns:
        invalid_surfaces = set(df['surface'].dropna()) - set(allowed_surfaces)
        if invalid_surfaces:
            raise ValueError(f"Validation failed: Invalid surfaces found: {invalid_surfaces}")
            
    # Check ranks are positive if present
    rank_cols = ['winner_rank', 'loser_rank', 'player_1_rank', 'player_2_rank']
    for rank_col in rank_cols:
        if rank_col in df.columns:
            if (df[rank_col] <= 0).any():
                raise ValueError(f"Validation failed: Column {rank_col} contains non-positive ranks.")
                
    # Check classification target constraints
    target_col = schema.get("target")
    if target_col and target_col in df.columns:
        unique_targets = set(df[target_col].dropna().unique())
        if not unique_targets.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"Validation failed: Target column {target_col} contains values other than 0 and 1.")

    return True