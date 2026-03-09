import pytest
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import joblib

from src.utils import load_csv, save_csv, save_model, load_model


def test_save_and_load_csv(tmp_path: Path):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    filepath = tmp_path / "test.csv"

    # Test saving
    save_csv(df, filepath)
    assert filepath.exists()

    # Test loading
    loaded_df = load_csv(filepath)
    pd.testing.assert_frame_equal(df, loaded_df)


def test_save_and_load_model(tmp_path: Path):
    # Dummy model
    model = LogisticRegression()
    filepath = tmp_path / "model.joblib"

    # Test saving
    save_model(model, filepath)
    assert filepath.exists()

    # Test loading
    loaded_model = load_model(filepath)
    assert isinstance(loaded_model, LogisticRegression)
