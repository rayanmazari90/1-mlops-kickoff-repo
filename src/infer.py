"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).

Educational Goal:
- Why this module exists in an MLOps system: Applies the trained model to
  new, unseen data to generate business value.
- Responsibility (separation of concerns): Loading the pipeline and making
  predictions on strictly formatted inference data.
- Pipeline contract (inputs and outputs): Takes a fitted model and new data,
  outputs a DataFrame of predictions matching the input index.
"""

import joblib
from pathlib import Path

import pandas as pd


def run_inference(
    model_or_path, X_infer: pd.DataFrame, save_path: str = None
) -> pd.DataFrame:
    """
    Inputs:
    - model_or_path: The fitted scikit-learn Pipeline, or path to joblib model.
    - X_infer: The feature DataFrame for generating new predictions.
    - save_path: Optional path to save predictions.
    Outputs:
    - A pd.DataFrame containing exactly one column named 'prediction',
      preserving the original index.
    Why this contract matters for reliable ML delivery:
    - A strict output format ensures downstream engineering systems
      (like dashboards or databases) can seamlessly consume predictions.
    """
    if not isinstance(X_infer, pd.DataFrame):
        raise TypeError("Inference data must be a Pandas DataFrame, not numpy array.")

    # Load model if a path is provided
    if isinstance(model_or_path, (str, Path)):
        model = joblib.load(model_or_path)
    else:
        model = model_or_path

    # Duck-typing validation
    if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
        raise TypeError("Model artifact is invalid: missing .predict() method.")

    print("Running inference on new data...")

    # Generate predictions
    preds = model.predict(X_infer)
    df_preds = pd.DataFrame({"prediction": preds}, index=X_infer.index)

    # Add probability column if it's a classifier
    if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba")):
        probas = model.predict_proba(X_infer)
        if probas.shape[1] == 2:
            df_preds["proba"] = probas[:, 1]
        else:
            for i in range(probas.shape[1]):
                df_preds[f"proba_{i}"] = probas[:, i]

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df_preds.to_csv(save_path)
        print(f"Predictions saved to {save_path}")

    return df_preds
