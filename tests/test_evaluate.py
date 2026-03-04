import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.evaluate import evaluate_model

class MockModel:
    def predict(self, X):
        # Predict class 1 if p1_rank < p2_rank, else class 0
        return (X['p1_rank'] < X['p2_rank']).astype(int).values

    def predict_proba(self, X):
        # Fake probability for class 1
        prob_1 = np.where(X['p1_rank'] < X['p2_rank'], 0.8, 0.2)
        prob_0 = 1.0 - prob_1
        return np.column_stack((prob_0, prob_1))

@pytest.fixture
def mock_classification_data():
    X_test = pd.DataFrame({
        'p1_rank': [10, 50, 5, 100],
        'p2_rank': [20, 10, 15, 50]
    })
    # Target: 1 means player 1 wins
    y_test = pd.Series([1, 0, 1, 0])
    return X_test, y_test

def test_evaluate_model_classification(mock_classification_data, tmp_path, monkeypatch):
    """
    Test evaluate_model outputs correct dictionary and writes to JSON artifact.
    """
    # Monkeypatch the reports directory to write into a temporary path
    monkeypatch.chdir(tmp_path)
    
    X_test, y_test = mock_classification_data
    model = MockModel()

    metrics = evaluate_model(model, X_test, y_test, "classification")

    # Assert dictionary keys exist
    expected_keys = [
        "log_loss", "brier_score", "accuracy", "auc",
        "baseline_log_loss", "baseline_brier_score", "baseline_accuracy", "baseline_auc"
    ]
    for k in expected_keys:
        assert k in metrics, f"Missing metric key: {k}"

    # Assert metrics are floats (or None for AUC if undefined, but here defined)
    for k, v in metrics.items():
        assert isinstance(v, float) or v is None

    # Bounds checking
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    if metrics["auc"] is not None:
        assert 0.0 <= metrics["auc"] <= 1.0

    # Ensure artifact was created
    metrics_file = tmp_path / "reports" / "metrics.json"
    assert metrics_file.exists(), "metrics.json artifacts not created"
    
    # Load and ensure it can be parsed
    with open(metrics_file, "r") as f:
        loaded_metrics = json.load(f)
        
    for k in expected_keys:
        assert k in loaded_metrics

def test_evaluate_model_missing_rank_columns():
    """
    Tests evaluation if rank columns for baseline are missing.
    """
    X_test = pd.DataFrame({'other_feat': [1, 2, 3]})
    y_test = pd.Series([1, 0, 1])
    model = MockModel()
    
    # Just setting predict, predict_proba doesn't need 'p1_rank' here just returning dummies
    model.predict = lambda X: np.array([1, 0, 1])
    model.predict_proba = lambda X: np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    
    metrics = evaluate_model(model, X_test, y_test, "classification")
    
    assert "log_loss" in metrics
    assert "baseline_accuracy" not in metrics # Should skip baseline logic silently
