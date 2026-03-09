"""
Educational Goal:
- Why this module exists in an MLOps system: Consolidates common I/O
  operations to prevent repetitive boilerplates.
- Responsibility (separation of concerns): Reading and writing data and
  models to disk.
- Pipeline contract (inputs and outputs): File paths go in, standard
  Python/Pandas objects come out, or vice versa.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported
      from config.yml in a later session
"""

import pandas as pd
import joblib
from pathlib import Path


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath: Path object pointing to the CSV file.
    Outputs:
    - pd.DataFrame containing the loaded data.
    Why this contract matters for reliable ML delivery:
    - Standardizes how we read tabular data, allowing us to swap backends
      (e.g., to Parquet) easily in the future.
    """
    print(f"Loading CSV from {filepath}...")
    # TODO: replace with logging later
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Adjust parameters (like sep, encoding)
    # if your dataset requires it
    # Why: Different datasets have different CSV formatting standard
    # Examples:
    # 1. pd.read_csv(filepath, sep=';')
    # 2. pd.read_csv(filepath, encoding='utf-8')
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student:
    # You must implement this logic to proceed!")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Inputs:
    - df: The DataFrame to save.
    - filepath: Path object indicating where to save the file.
    Outputs:
    - None
    Why this contract matters for reliable ML delivery:
    - Ensures intermediate datasets are reliably persisted for debugging
      and auditability.
    """
    print(f"Saving CSV to {filepath}...")  # TODO: replace with logging later
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Adjust to_csv params if needed (e.g. retaining index)
    # Why: Downstream tools may expect specific delimiters or index handling
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """
    Inputs:
    - model: The trained scikit-learn Pipeline or estimator.
    - filepath: Path object indicating where to save the model.
    Outputs:
    - None
    Why this contract matters for reliable ML delivery:
    - Allows us to separate the training compute environment from the
      inference compute environment.
    """
    print(f"Saving model to {filepath}...")  # TODO: replace with logging later
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Update serialization logic
    # if using a framework other than joblib.
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    joblib.dump(model, filepath)


def load_model(filepath: Path):
    """
    Inputs:
    - filepath: Path object pointing to the serialized model file.
    Outputs:
    - The deserialized scikit-learn Pipeline or estimator object.
    Why this contract matters for reliable ML delivery:
    - Ensures the exact same model artifact evaluated in development is
      the one used in production.
    """
    print(f"Loading model from {filepath}...")
    # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Update deserialization logic if using a
    # framework other than joblib
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return joblib.load(filepath)
