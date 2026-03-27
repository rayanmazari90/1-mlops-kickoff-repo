import pandas as pd
import wandb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.train import train_model


def test_train_model_creates_pipeline(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")

    X_train = pd.DataFrame({"feature1": [1, 2, 3, 4], "feature2": [10, 20, 30, 40]})
    y_train = pd.Series([0, 1, 0, 1])

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), ["feature1", "feature2"])]
    )

    model_config = {
        "algorithm": "LogisticRegression",
        "hyperparams": {},
        "random_seed": 42,
    }

    wandb.init(mode="offline", project="test")
    pipeline = train_model(X_train, y_train, preprocessor, model_config)
    wandb.finish()

    assert isinstance(pipeline, Pipeline)
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline.named_steps["model"], "classes_")
