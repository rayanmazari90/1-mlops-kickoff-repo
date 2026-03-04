"""
Module: Model Training
----------------------
Role: Bundle preprocessing and algorithms into a single Pipeline and fit on training data.
Input: pandas.DataFrame (Processed) + ColumnTransformer (Recipe).
Output: Serialized scikit-learn Pipeline in `models/`.
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Isolates the model training logic from data processing.
- Responsibility (separation of concerns): Fitting the full Pipeline (preprocessor + model) on training data.
- Pipeline contract (inputs and outputs): Takes training data and an unfitted preprocessor, outputs a fitted Pipeline.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression

def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor, model_config: dict):
    """
    Inputs:
    - X_train: The feature DataFrame for training.
    - y_train: The target Series for training.
    - preprocessor: The unfitted ColumnTransformer from features.py.
    - model_config: Dictionary containing algorithm choice and hyperparameters.
    Outputs:
    - A fully fitted scikit-learn Pipeline.
    Why this contract matters for reliable ML delivery:
    - Bundling the preprocessor and model into a single Pipeline guarantees exact reproduction of logic during inference.
    """
    algorithm_name = model_config.get("algorithm", "LogisticRegression")
    print(f"Training Pipeline for algorithm: {algorithm_name}...") # TODO: replace with logging later
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
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
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator)
    ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline