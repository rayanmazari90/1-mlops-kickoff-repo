import pandas as pd
import pytest
import wandb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.train import train_model


@pytest.fixture
def _training_data():
    X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [10, 20, 30, 40, 50]})
    y = pd.Series([0, 1, 0, 1, 0])
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), ["feature1", "feature2"])]
    )
    return X, y, preprocessor


def test_train_logistic_regression(_training_data, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    X, y, preprocessor = _training_data
    config = {"algorithm": "LogisticRegression", "hyperparams": {}, "random_seed": 42}

    wandb.init(mode="offline", project="test")
    pipeline = train_model(X, y, preprocessor, config)
    wandb.finish()

    assert isinstance(pipeline, Pipeline)
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline.named_steps["model"], "classes_")


def test_train_random_forest(_training_data, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    X, y, preprocessor = _training_data
    config = {
        "algorithm": "RandomForestClassifier",
        "hyperparams": {"n_estimators": 10},
        "random_seed": 42,
    }

    wandb.init(mode="offline", project="test")
    pipeline = train_model(X, y, preprocessor, config)
    wandb.finish()

    assert isinstance(pipeline, Pipeline)
    assert pipeline.named_steps["model"].__class__.__name__ == "RandomForestClassifier"


def test_train_ridge(_training_data, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    X, y, preprocessor = _training_data
    y_reg = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    config = {"algorithm": "Ridge", "hyperparams": {}, "random_seed": 42}

    wandb.init(mode="offline", project="test")
    pipeline = train_model(X, y_reg, preprocessor, config)
    wandb.finish()

    assert isinstance(pipeline, Pipeline)
    assert pipeline.named_steps["model"].__class__.__name__ == "Ridge"


def test_train_unsupported_algorithm(_training_data, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    X, y, preprocessor = _training_data
    config = {"algorithm": "MagicModel", "hyperparams": {}, "random_seed": 42}

    wandb.init(mode="offline", project="test")
    with pytest.raises(ValueError, match="not explicitly supported"):
        train_model(X, y, preprocessor, config)
    wandb.finish()


def test_train_default_algorithm(_training_data, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    X, y, preprocessor = _training_data
    config = {}

    wandb.init(mode="offline", project="test")
    pipeline = train_model(X, y, preprocessor, config)
    wandb.finish()

    assert pipeline.named_steps["model"].__class__.__name__ == "LogisticRegression"
