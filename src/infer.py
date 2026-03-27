"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""

import joblib
from pathlib import Path

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


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
    """
    if not isinstance(X_infer, pd.DataFrame):
        raise TypeError("Inference data must be a Pandas DataFrame, not numpy array.")

    if isinstance(model_or_path, (str, Path)):
        model = joblib.load(model_or_path)
    else:
        model = model_or_path

    if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
        raise TypeError("Model artifact is invalid: missing .predict() method.")

    logger.info("Running inference on new data ...")

    preds = model.predict(X_infer)
    df_preds = pd.DataFrame({"prediction": preds}, index=X_infer.index)

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
        logger.info("Predictions saved to %s", save_path)

    return df_preds
