"""
Module: Utilities
-----------------
Consolidates common I/O operations to prevent repetitive boilerplate.
"""

import joblib
from pathlib import Path

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    logger.info("Loading CSV from %s ...", filepath)
    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Persist a DataFrame as CSV."""
    logger.info("Saving CSV to %s ...", filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """Serialize a trained model to disk via joblib."""
    logger.info("Saving model to %s ...", filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: Path):
    """Deserialize a model from disk via joblib."""
    logger.info("Loading model from %s ...", filepath)
    return joblib.load(filepath)
