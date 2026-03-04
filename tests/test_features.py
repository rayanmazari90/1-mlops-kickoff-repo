import pytest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from src.features import build_features, get_feature_preprocessor

@pytest.fixture
def sample_clean_df():
    """Create a minimal mocked valid tennis match dataset after clean_data"""
    data = {
        'tourney_id': ['1', '2', '3', '4'],
        'match_num': [1, 2, 3, 4],
        'surface': ['Hard', 'Clay', 'Grass', 'Hard'],
        'tourney_level': ['G', 'M', 'A', 'G'],
        'round': ['R128', 'R64', 'QF', 'F'],
        'winner_id': [100, 101, 102, 103],
        'loser_id': [200, 201, 202, 203],
        'winner_rank': [1, 10, 5, 2],
        'loser_rank': [50, 20, 15, 8],
        'winner_age': [25.0, 30.1, 22.5, 28.0],
        'loser_age': [20.0, 28.5, 31.0, 24.2],
        'winner_ht': [185.0, 190.0, 180.0, 198.0],
        'loser_ht': [175.0, 188.0, 182.0, 185.0],
        'winner_hand': ['R', 'L', 'R', 'R'],
        'loser_hand': ['R', 'R', 'L', 'R'],
    }
    return pd.DataFrame(data)

def test_build_features_shapes(sample_clean_df):
    """Verify that build_features returns X and y of the exact same length as the input"""
    original_len = len(sample_clean_df)
    X, y = build_features(sample_clean_df)
    
    assert len(X) == original_len, "X must have the same number of rows as input"
    assert len(y) == original_len, "y must have the same number of rows as input"
    assert "player_1_win" == y.name, "Target series must be named 'player_1_win'"

def test_build_features_no_leakage(sample_clean_df):
    """Verify that build_features drops all direct winner/loser columns (including IDs)"""
    X, y = build_features(sample_clean_df)
    
    # Assert IDs are gone
    assert 'winner_id' not in X.columns
    assert 'loser_id' not in X.columns
    assert 'tourney_id' not in X.columns
    assert 'match_num' not in X.columns
    
    # Assert raw stats are gone
    for col in X.columns:
        assert not col.startswith('winner_'), f"Leakage column found: {col}"
        assert not col.startswith('loser_'), f"Leakage column found: {col}"
        assert not col.startswith('w_'), f"Possible match stat leakage found: {col}"
        assert not col.startswith('l_'), f"Possible match stat leakage found: {col}"
        
def test_build_features_symmetric_construction(sample_clean_df):
    """Verify that p1/p2 features are created correctly and difference is computed"""
    X, y = build_features(sample_clean_df)
    
    expected_cols = ['surface', 'tourney_level', 'round', 
                     'p1_rank', 'p2_rank', 'rank_diff',
                     'age_diff', 'ht_diff', 'p1_hand', 'p2_hand']
                     
    for col in expected_cols:
        assert col in X.columns, f"Expected engineered column {col} is missing"

def test_get_feature_preprocessor_unfitted():
    """Verify the ColumnTransformer returns unfitted"""
    numeric_cols = ['p1_rank', 'p2_rank']
    categorical_cols = ['surface']
    
    preprocessor = get_feature_preprocessor(numeric_cols, categorical_cols)
    
    # Check that it's a ColumnTransformer
    assert preprocessor.__class__.__name__ == 'ColumnTransformer'
    
    # Create fake data and attempt to transform WITHOUT fitting. It must fail.
    fake_X = pd.DataFrame({'p1_rank': [1, 2], 'p2_rank': [3, 4], 'surface': ['Hard', 'Clay']})
    
    with pytest.raises(NotFittedError):
        preprocessor.transform(fake_X)
        
def test_get_feature_preprocessor_transformers():
    """Verify the scaling and encoding pipelines are assembled"""
    preprocessor = get_feature_preprocessor(
        numeric_cols=['p1_rank'], 
        categorical_cols=['surface']
    )
    
    transformers = preprocessor.transformers
    # Extract names of active transformers
    names = [t[0] for t in transformers]
    assert 'num' in names
    assert 'cat' in names
