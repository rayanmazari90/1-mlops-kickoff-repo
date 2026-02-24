"""
Module: Data Loader
-------------------
Role: Ingest raw data from sources (CSV, SQL, API).
Input: Path to file or connection string.
Output: pandas.DataFrame (Raw).
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Ingests raw data from external sources into the pipeline.
- Responsibility (separation of concerns): Handling data fetching, downloading, or reading from the raw zone.
- Pipeline contract (inputs and outputs): Takes a source location, outputs a raw Pandas DataFrame.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd
from pathlib import Path
from src.utils import load_csv, save_csv

def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Inputs:
    - raw_data_path: Path object pointing to the raw CSV file.
    Outputs:
    - pd.DataFrame containing the raw, unmodified data.
    Why this contract matters for reliable ML delivery:
    - Establishes an immutable starting point for the pipeline, ensuring reproducibility.
    """
    print(f"Attempting to load raw data from {raw_data_path}...") # TODO: replace with logging later
    
    if not raw_data_path.exists():
        print(f"LOUD WARNING: {raw_data_path} does not exist.")
        print("Creating a dummy dataset with hardcoded columns for scaffolding purposes only!")
        print("Students must replace this dataset and update their SETTINGS dict.")
        
        dummy_data = {
            "num_feature": [1.5, 2.3, 3.1, 4.8, 5.0, 6.2, 7.1, 8.9, 9.4, 10.1],
            "cat_feature": ["A", "B", "A", "C", "B", "A", "C", "C", "B", "A"],
            "target": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
        }
        df_dummy = pd.DataFrame(dummy_data)
        save_csv(df_dummy, raw_data_path)
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: In real environments, you might fetch from a SQL database, S3 bucket, or an API
    # Examples:
    # 1. df = pd.read_sql("SELECT * FROM table", conn)
    # 2. df = fetch_from_s3(bucket, key)
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    return load_csv(raw_data_path)