"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Applies the trained model to new, unseen data to generate business value.
- Responsibility (separation of concerns): Loading the pipeline and making predictions on strictly formatted inference data.
- Pipeline contract (inputs and outputs): Takes a fitted model and new data, outputs a DataFrame of predictions matching the input index.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd

def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model: The fitted scikit-learn Pipeline loaded from disk.
    - X_infer: The feature DataFrame for generating new predictions.
    Outputs:
    - A pd.DataFrame containing exactly one column named 'prediction', preserving the original index.
    Why this contract matters for reliable ML delivery:
    - A strict output format ensures downstream engineering systems (like dashboards or databases) can seamlessly consume predictions.
    """
    print("Running inference on new data...") # TODO: replace with logging later
    
    preds = model.predict(X_infer)
    df_preds = pd.DataFrame({"prediction": preds}, index=X_infer.index)
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: You may need to apply post-processing (e.g., thresholding probabilities or mapping IDs to classes)
    # Examples:
    # 1. df_preds['prediction'] = (model.predict_proba(X_infer)[:, 1] > 0.7).astype(int)
    # 2. df_preds['prediction'] = df_preds['prediction'].map({0: 'No', 1: 'Yes'})
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    return df_preds