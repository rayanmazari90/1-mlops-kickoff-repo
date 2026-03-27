import json

import numpy as np
import pandas as pd
import pytest
import wandb

from src.evaluate import evaluate_model


class MockModel:
    def predict(self, X):
        return (X["p1_rank"] < X["p2_rank"]).astype(int).values

    def predict_proba(self, X):
        prob_1 = np.where(X["p1_rank"] < X["p2_rank"], 0.8, 0.2)
        prob_0 = 1.0 - prob_1
        return np.column_stack((prob_0, prob_1))


class ModelNoProba:
    """Model without predict_proba."""

    def predict(self, X):
        return np.array([1] * len(X))


class ModelMultiProba:
    """Model that returns >2 probability columns."""

    def predict(self, X):
        return np.array([0] * len(X))

    def predict_proba(self, X):
        return np.column_stack(
            [np.full(len(X), 0.6), np.full(len(X), 0.3), np.full(len(X), 0.1)]
        )


@pytest.fixture
def mock_classification_data():
    X_test = pd.DataFrame({"p1_rank": [10, 50, 5, 100], "p2_rank": [20, 10, 15, 50]})
    y_test = pd.Series([1, 0, 1, 0])
    return X_test, y_test


def test_evaluate_model_classification(mock_classification_data, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.chdir(tmp_path)

    X_test, y_test = mock_classification_data
    model = MockModel()

    wandb.init(mode="offline", project="test")
    metrics = evaluate_model(model, X_test, y_test, "classification")
    wandb.finish()

    expected_keys = [
        "log_loss",
        "brier_score",
        "accuracy",
        "auc",
        "baseline_log_loss",
        "baseline_brier_score",
        "baseline_accuracy",
        "baseline_auc",
    ]
    for k in expected_keys:
        assert k in metrics, f"Missing metric key: {k}"

    for k, v in metrics.items():
        assert isinstance(v, float) or v is None

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    if metrics["auc"] is not None:
        assert 0.0 <= metrics["auc"] <= 1.0

    metrics_file = tmp_path / "reports" / "metrics.json"
    assert metrics_file.exists()

    with open(metrics_file, "r") as f:
        loaded_metrics = json.load(f)

    for k in expected_keys:
        assert k in loaded_metrics


def test_evaluate_model_missing_rank_columns(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.chdir(tmp_path)

    X_test = pd.DataFrame({"other_feat": [1, 2, 3]})
    y_test = pd.Series([1, 0, 1])
    model = MockModel()

    model.predict = lambda X: np.array([1, 0, 1])
    model.predict_proba = lambda X: np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])

    wandb.init(mode="offline", project="test")
    metrics = evaluate_model(model, X_test, y_test, "classification")
    wandb.finish()

    assert "log_loss" in metrics
    assert "baseline_accuracy" not in metrics


def test_evaluate_model_regression(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.chdir(tmp_path)

    class RegressionModel:
        def predict(self, X):
            return np.array([1.0, 2.0, 3.0])

    X_test = pd.DataFrame({"feat": [1, 2, 3]})
    y_test = pd.Series([1.1, 2.2, 2.8])

    wandb.init(mode="offline", project="test")
    metrics = evaluate_model(RegressionModel(), X_test, y_test, "regression")
    wandb.finish()

    assert "rmse" in metrics
    assert metrics["rmse"] >= 0.0


def test_evaluate_model_invalid_problem_type(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.chdir(tmp_path)

    model = MockModel()
    X_test = pd.DataFrame({"p1_rank": [10]})
    y_test = pd.Series([1])

    wandb.init(mode="offline", project="test")
    with pytest.raises(ValueError, match="problem_type"):
        evaluate_model(model, X_test, y_test, "unknown")
    wandb.finish()


def test_evaluate_model_no_predict_proba(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.chdir(tmp_path)

    X_test = pd.DataFrame({"feat": [1, 2, 3]})
    y_test = pd.Series([1, 0, 1])

    wandb.init(mode="offline", project="test")
    metrics = evaluate_model(ModelNoProba(), X_test, y_test, "classification")
    wandb.finish()

    assert "accuracy" in metrics


def test_evaluate_model_multiclass_proba(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.chdir(tmp_path)

    X_test = pd.DataFrame({"feat": [1, 2, 3]})
    y_test = pd.Series([0, 1, 0])

    wandb.init(mode="offline", project="test")
    metrics = evaluate_model(ModelMultiProba(), X_test, y_test, "classification")
    wandb.finish()

    assert "log_loss" in metrics


def test_evaluate_model_constant_y(monkeypatch, tmp_path):
    """When y has only one class, AUC should be None but other metrics should work."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.chdir(tmp_path)

    X_test = pd.DataFrame({"p1_rank": [10, 20, 30], "p2_rank": [30, 40, 50]})
    y_test = pd.Series([1, 0, 1])

    wandb.init(mode="offline", project="test")
    metrics = evaluate_model(MockModel(), X_test, y_test, "classification")
    wandb.finish()

    assert "accuracy" in metrics
    assert "baseline_accuracy" in metrics
