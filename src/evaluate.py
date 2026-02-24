"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Quantifies model performance on unseen data.
- Responsibility (separation of concerns): Generating metrics to determine if the model is ready for deployment.
- Pipeline contract (inputs and outputs): Takes a fitted model and test data, outputs an evaluation metric float.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> float:
    """
    Inputs:
    - model: The fitted scikit-learn Pipeline.
    - X_test: The feature DataFrame for testing.
    - y_test: The target Series for testing.
    - problem_type: String indicating "regression" or "classification".
    Outputs:
    - A single float representing the primary evaluation metric (RMSE or F1).
    Why this contract matters for reliable ML delivery:
    - Automated metrics allow CI/CD systems to block degraded models from reaching production automatically.
    """
    print("Evaluating model performance on test set...") # TODO: replace with logging later
    
    predictions = model.predict(X_test)
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Business stakeholders may care about different metrics (e.g., Precision/Recall vs Accuracy)
    # Examples:
    # 1. metric = mean_absolute_error(y_test, predictions)
    # 2. metric = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    if problem_type == "regression":
        mse = mean_squared_error(y_test, predictions)
        metric = float(np.sqrt(mse))
        print(f"Test RMSE: {metric:.4f}")
    elif problem_type == "classification":
        metric = float(f1_score(y_test, predictions, average='weighted'))
        print(f"Test Weighted F1 Score: {metric:.4f}")
    else:
        raise ValueError("problem_type must be 'regression' or 'classification'")
        
    return metric