"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.

Educational Goal:
- Why this module exists in an MLOps system: Quantifies model performance
  on unseen data.
- Responsibility (separation of concerns): Generating metrics to determine
  if the model is ready for deployment.
- Pipeline contract (inputs and outputs): Takes a fitted model and test
  data, outputs an evaluation metric float.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported
      from config.yml in a later session
"""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    log_loss,
    brier_score_loss,
    accuracy_score,
    roc_auc_score,
)


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
    Why this contract matters for reliable ML delivery:
    - Automated metrics allow CI/CD systems to block degraded models
      from reaching production automatically.
    """
    print(
        "Evaluating model performance on test set..."
    )  # TODO: replace with logging later

    metrics = {}

    if problem_type == "regression":
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = float(np.sqrt(mse))
        metrics["rmse"] = rmse
        print(f"Test RMSE: {rmse:.4f}")

    elif problem_type == "classification":
        # Model predictions
        predictions = model.predict(X_test)

        # Probabilities for log_loss, brier_score, AUC
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            if probs.shape[1] == 2:
                prob_pos = probs[:, 1]
            else:
                # Fallback although sklearn usually outputs class 0, class 1
                prob_pos = probs[:, 0]
        else:
            prob_pos = predictions.astype(
                float
            )  # Fallback if predict_proba is not available

        # Model Metrics
        metrics["log_loss"] = float(log_loss(y_test, prob_pos))
        metrics["brier_score"] = float(brier_score_loss(y_test, prob_pos))
        metrics["accuracy"] = float(accuracy_score(y_test, predictions))

        # Calculate AUC only if there's variance in y_test
        if len(np.unique(y_test)) > 1:
            try:
                metrics["auc"] = float(roc_auc_score(y_test, prob_pos))
            except ValueError:
                metrics["auc"] = None
        else:
            metrics["auc"] = None

        print(
            f"Model Metrics - Log Loss: {metrics['log_loss']:.4f}, "
            f"Brier: {metrics['brier_score']:.4f}, "
            f"Acc: {metrics['accuracy']:.4f}"
        )

        # Baseline: Predict player with higher ATP rank (lower rank number)
        # We need p1_rank and p2_rank in X_test for this.
        if "p1_rank" in X_test.columns and "p2_rank" in X_test.columns:
            print("Evaluating baseline (higher rank wins)...")
            # Lower rank value means higher ranked player
            # If p1_rank < p2_rank, p1 has higher rank -> predict 1 (p1 wins)
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

            print(
                f"Baseline Metrics - Log Loss: {metrics['baseline_log_loss']:.4f}, "
                f"Brier: {metrics['baseline_brier_score']:.4f}, "
                f"Acc: {metrics['baseline_accuracy']:.4f}"
            )
    else:
        raise ValueError("problem_type must be 'regression' or 'classification'")

    # Save metrics to JSON artifact
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {metrics_path}")

    # MLflow Tracking - log all computed metrics (filtering out None)
    loggable_metrics = {k: v for k, v in metrics.items() if v is not None}
    mlflow.log_metrics(loggable_metrics)

    return metrics
