"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_squared_error,
    roc_auc_score,
)

from src.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str
) -> dict:
    """
    Inputs:
    - model: The fitted scikit-learn Pipeline.
    - X_test: The feature DataFrame for testing.
    - y_test: The target Series for testing.
    - problem_type: String indicating "regression" or "classification".
    Outputs:
    - A dictionary representing multiple evaluation metrics.
    """
    logger.info("Evaluating model performance on test set ...")

    metrics = {}

    if problem_type == "regression":
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = float(np.sqrt(mse))
        metrics["rmse"] = rmse
        logger.info("Test RMSE: %.4f", rmse)

    elif problem_type == "classification":
        predictions = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            if probs.shape[1] == 2:
                prob_pos = probs[:, 1]
            else:
                prob_pos = probs[:, 0]
        else:
            prob_pos = predictions.astype(float)

        metrics["log_loss"] = float(log_loss(y_test, prob_pos))
        metrics["brier_score"] = float(brier_score_loss(y_test, prob_pos))
        metrics["accuracy"] = float(accuracy_score(y_test, predictions))

        if len(np.unique(y_test)) > 1:
            try:
                metrics["auc"] = float(roc_auc_score(y_test, prob_pos))
            except ValueError:
                metrics["auc"] = None
        else:
            metrics["auc"] = None

        logger.info(
            "Model Metrics - Log Loss: %.4f, Brier: %.4f, Acc: %.4f",
            metrics["log_loss"],
            metrics["brier_score"],
            metrics["accuracy"],
        )

        if "p1_rank" in X_test.columns and "p2_rank" in X_test.columns:
            logger.info("Evaluating baseline (higher rank wins) ...")
            baseline_probs = (X_test["p1_rank"] < X_test["p2_rank"]).astype(float)
            baseline_preds = baseline_probs.round()

            metrics["baseline_log_loss"] = float(log_loss(y_test, baseline_probs))
            metrics["baseline_brier_score"] = float(
                brier_score_loss(y_test, baseline_probs)
            )
            metrics["baseline_accuracy"] = float(accuracy_score(y_test, baseline_preds))

            if len(np.unique(y_test)) > 1:
                try:
                    metrics["baseline_auc"] = float(
                        roc_auc_score(y_test, baseline_probs)
                    )
                except ValueError:
                    metrics["baseline_auc"] = None
            else:
                metrics["baseline_auc"] = None

            logger.info(
                "Baseline Metrics - Log Loss: %.4f, Brier: %.4f, Acc: %.4f",
                metrics["baseline_log_loss"],
                metrics["baseline_brier_score"],
                metrics["baseline_accuracy"],
            )
    else:
        raise ValueError("problem_type must be 'regression' or 'classification'")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Metrics saved to %s", metrics_path)

    loggable_metrics = {k: v for k, v in metrics.items() if v is not None}
    wandb.log(loggable_metrics)

    return metrics
