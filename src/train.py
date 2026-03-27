"""
Module: Model Training
----------------------
Role: Bundle preprocessing and algorithms into a single Pipeline and fit
      on training data.
Input: pandas.DataFrame (Processed) + ColumnTransformer (Recipe).
Output: Serialized scikit-learn Pipeline in `models/`.
"""

import wandb
import pandas as pd
from sklearn.pipeline import Pipeline

from src.logger import get_logger

logger = get_logger(__name__)


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, preprocessor, model_config: dict
):
    """
    Inputs:
    - X_train: The feature DataFrame for training.
    - y_train: The target Series for training.
    - preprocessor: The unfitted ColumnTransformer from features.py.
    - model_config: Dictionary containing algorithm choice and hyperparameters.
    Outputs:
    - A fully fitted scikit-learn Pipeline.
    """
    algorithm_name = model_config.get("algorithm", "LogisticRegression")
    logger.info("Training Pipeline for algorithm: %s ...", algorithm_name)

    hyperparams = model_config.get("hyperparams", {})
    random_seed = model_config.get("random_seed", 42)

    if algorithm_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression(random_state=random_seed, **hyperparams)
    elif algorithm_name == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier(random_state=random_seed, **hyperparams)
    elif algorithm_name == "XGBClassifier":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(random_state=random_seed, **hyperparams)
    elif algorithm_name == "Ridge":
        from sklearn.linear_model import Ridge

        estimator = Ridge(random_state=random_seed, **hyperparams)
    else:
        raise ValueError(f"Algorithm {algorithm_name} is not explicitly supported yet.")

    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    pipeline.fit(X_train, y_train)

    wandb.config.update(
        {"algorithm": algorithm_name, "random_seed": random_seed, **hyperparams},
        allow_val_change=True,
    )

    return pipeline
