import pytest
import pandas as pd
import numpy as np
from src.clean_data import clean_dataframe


def test_clean_dataframe_standardizes_columns():
    df_raw = pd.DataFrame(
        {" Tourney Date ": ["20220101"], "Winner ID": [123], "target": [1]}
    )
    df_clean = clean_dataframe(df_raw, "target")
    assert "tourney_date" in df_clean.columns
    assert "winner_id" in df_clean.columns
    assert "target" in df_clean.columns


def test_clean_dataframe_removes_duplicates():
    df_raw = pd.DataFrame(
        {
            "tourney_date": ["20220101", "20220101"],
            "winner_id": [123, 123],
            "target": [1, 1],
        }
    )
    df_clean = clean_dataframe(df_raw, "target")
    assert len(df_clean) == 1


def test_clean_dataframe_parses_dates():
    df_raw = pd.DataFrame({"tourney_date": ["20220101", "20220102"], "target": [1, 1]})
    df_clean = clean_dataframe(df_raw, "target")
    assert pd.api.types.is_datetime64_any_dtype(df_clean["tourney_date"])


def test_clean_dataframe_drops_missing_dates():
    df_raw = pd.DataFrame({"tourney_date": ["invalid_date", None], "target": [1, 1]})
    df_clean = clean_dataframe(df_raw, "target")
    assert len(df_clean) == 0


def test_clean_dataframe_resets_index():
    df_raw = pd.DataFrame(
        {"tourney_date": ["20220101", None, "20220103"], "target": [1, 0, 1]}
    )
    df_clean = clean_dataframe(df_raw, "target")
    assert len(df_clean) == 2
    assert list(df_clean.index) == [0, 1]


def test_clean_dataframe_imputes_ranks():
    df_raw = pd.DataFrame(
        {
            "tourney_date": ["20220101"],
            "winner_rank": [None],
            "loser_rank": [np.nan],
            "target": [1],
        }
    )
    df_clean = clean_dataframe(df_raw, "target")
    assert df_clean["winner_rank"].iloc[0] == 999999.0
    assert df_clean["loser_rank"].iloc[0] == 999999.0


def test_clean_dataframe_drops_missing_surface():
    df_raw = pd.DataFrame(
        {
            "tourney_date": ["20220101", "20220102"],
            "surface": ["Hard", None],
            "target": [1, 1],
        }
    )
    df_clean = clean_dataframe(df_raw, "target")
    assert len(df_clean) == 1
    assert df_clean["surface"].iloc[0] == "Hard"
