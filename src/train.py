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

def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor, problem_type: str):
    """
    Inputs:
    - X_train: The feature DataFrame for training.
    - y_train: The target Series for training.
    - preprocessor: The unfitted ColumnTransformer from features.py.
    - problem_type: String ("regression" or "classification") to select the baseline model.
    Outputs:
    - A fully fitted scikit-learn Pipeline.
    Why this contract matters for reliable ML delivery:
    - Bundling the preprocessor and model into a single Pipeline guarantees exact reproduction of logic during inference.
    """
    print(f"Training Pipeline for problem type: {problem_type}...") # TODO: replace with logging later
    
    if problem_type == "regression":
        estimator = Ridge()
    elif problem_type == "classification":
        estimator = LogisticRegression(max_iter=500)
    else:
        raise ValueError("problem_type must be either 'regression' or 'classification'")
        
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: You will want to use more powerful algorithms or hyperparameter tuning here
    # Examples:
    # 1. estimator = RandomForestRegressor(n_estimators=100)
    # 2. estimator = XGBClassifier(learning_rate=0.1)
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator)
    ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline