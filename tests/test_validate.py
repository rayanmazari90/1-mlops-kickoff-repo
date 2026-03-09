import pandas as pd
import pytest
from src.validate import validate_dataframe

@pytest.fixture
def mock_config():
    return {
        "schema": {
            "required_columns": ["tourney_date", "surface", "winner_id", "loser_id"],
            "allowed_surfaces": ["Hard", "Clay", "Grass", "Carpet"],
            "non_null_columns": ["tourney_date", "winner_id", "loser_id"],
            "target": "target_col"
        }
    }

def test_validate_empty_df(mock_config):
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is completely empty"):
        validate_dataframe(df, mock_config)

def test_validate_missing_columns(mock_config):
    df = pd.DataFrame({
        "tourney_date": ["20220101"],
        "surface": ["Hard"],
        "winner_id": [1]
        # Missing loser_id
    })
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(df, mock_config)

def test_validate_bad_surface(mock_config):
    df = pd.DataFrame({
        "tourney_date": ["20220101"],
        "surface": ["Sand"], # Invalid
        "winner_id": [1],
        "loser_id": [2]
    })
    with pytest.raises(ValueError, match="Validation failed:"):
        validate_dataframe(df, mock_config)

def test_validate_bad_date(mock_config):
    df = pd.DataFrame({
        "tourney_date": ["NotADate"],
        "surface": ["Hard"],
        "winner_id": [1],
        "loser_id": [2]
    })
    with pytest.raises(ValueError, match="Validation failed:"):
        validate_dataframe(df, mock_config)

def test_validate_null_values(mock_config):
    df = pd.DataFrame({
        "tourney_date": ["20220101"],
        "surface": ["Hard"],
        "winner_id": [None],
        "loser_id": [2]
    })
    with pytest.raises(ValueError, match="contains null values"):
        validate_dataframe(df, mock_config)
        
def test_validate_negative_ranks(mock_config):
    df = pd.DataFrame({
        "tourney_date": ["20220101"],
        "surface": ["Hard"],
        "winner_id": [1],
        "loser_id": [2],
        "winner_rank": [-10],
        "loser_rank": [20]
    })
    with pytest.raises(ValueError, match="Validation failed:"):
        validate_dataframe(df, mock_config)

def test_validate_target_constraints(mock_config):
    df = pd.DataFrame({
        "tourney_date": ["20220101"],
        "surface": ["Hard"],
        "winner_id": [1],
        "loser_id": [2],
        "target_col": [2] # Only 0 or 1 allowed
    })
    with pytest.raises(ValueError, match="Validation failed:"):
        validate_dataframe(df, mock_config)

def test_validate_success(mock_config):
    df = pd.DataFrame({
        "tourney_date": ["20220101"],
        "surface": ["Hard"],
        "winner_id": [1],
        "loser_id": [2],
        "winner_rank": [1],
        "target_col": [1] # valid target
    })
    assert validate_dataframe(df, mock_config) is True
