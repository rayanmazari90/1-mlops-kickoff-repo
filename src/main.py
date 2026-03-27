"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow (Load -> Clean -> Validate -> Train ->
      Evaluate).
Usage: python -m src.main

Educational Goal:
- Why this module exists in an MLOps system: The orchestrator script that
  ties all components together.
- Responsibility (separation of concerns): Defining execution order,
  passing data between modules, and configuration management.
- Pipeline contract (inputs and outputs): Execution entry point that reads
  configuration and orchestrates artifacts.

TODO: Any temporary or hardcoded variable or parameter will be imported
      from config.yml in a later session
"""

import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import mlflow
import pandas as pd
import yaml

from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.features import build_features, get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.train import train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe


def main(config_path: str = "config.yaml"):
    logger.info("=== Starting MLOps Pipeline ===")

    # 1. Load config and fail fast
    logger.info("\n--- Load Configuration ---")
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Check for placeholder strings to fail fast
    config_str = yaml.dump(config)
    if "TODO" in config_str or "CHANGEME" in config_str:
        raise ValueError(
            "Configuration contains unresolved placeholders (TODO/CHANGEME). "
            "Please configure them properly."
        )

    # Extract config variables
    dataset_cfg = config.get("dataset", {})
    base_url = dataset_cfg.get("base_url")
    paths_cfg = config.get("paths", {})
    raw_dir = Path(paths_cfg.get("raw_dir", "data/raw"))
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    models_dir = Path(paths_cfg.get("models_dir", "models"))
    reports_dir = Path(paths_cfg.get("reports_dir", "reports"))

    target_column = config.get("schema", {}).get("target", "player_1_win")
    problem_type = config.get("problem_type", "classification")
    feat_cfg = config.get("features", {})
    numeric_pipeline = feat_cfg.get("numeric_pipeline", [])
    categorical_pipeline = feat_cfg.get("categorical_pipeline", [])

    # Check if download is forced or allowed
    download_if_missing = config.get("dataset", {}).get("download_if_missing", True)

    # Create directories
    logger.info("\n--- Setup Directories ---")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Combine all seasons for fetching data
    seasons = []
    for split_type in [
        "seasons_train",
        "seasons_val",
        "seasons_test",
        "seasons_infer",
    ]:
        seasons.extend(dataset_cfg.get(split_type, []))

    if not seasons:
        raise ValueError("No seasons configured in dataset config.")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("tennis_atp_prediction")

    with mlflow.start_run(run_name="batch_pipeline_run"):
        # 2. Load
        logger.info("\n--- Load Data ---")
        df_raw = load_raw_data(
            raw_dir=raw_dir,
            base_url=base_url,
            seasons=seasons,
            download_if_missing=download_if_missing,
        )

        # 3. Clean
        logger.info("\n--- Clean Data ---")
        df_clean = clean_dataframe(df_raw, target_column)

        # Save processed CSV for audit
        clean_path = processed_dir / "clean.csv"
        save_csv(df_clean, clean_path)

        # 4. Validate
        logger.info("\n--- Validate Data ---")
        validate_dataframe(df_clean, config=config)

        # 5. Early Split by seasons
        logger.info("\n--- Early Train/Val/Test Split ---")
        if "tourney_date" in df_clean.columns:
            year_series = pd.to_datetime(
                df_clean["tourney_date"], format="mixed"
            ).dt.year
        else:
            # Fallback if tourney_date not available
            # (should be due to validation)
            raise ValueError("tourney_date column missing from cleaned dataframe!")

        df_clean_train = df_clean[
            year_series.isin(dataset_cfg.get("seasons_train", []))
        ]
        df_clean_val = df_clean[year_series.isin(dataset_cfg.get("seasons_val", []))]
        df_clean_test = df_clean[year_series.isin(dataset_cfg.get("seasons_test", []))]
        df_clean_infer = df_clean[
            year_series.isin(dataset_cfg.get("seasons_infer", []))
        ]

        print(
            f"Split sizes -> Train: {len(df_clean_train)}, "
            f"Val: {len(df_clean_val)}, Test: {len(df_clean_test)}, "
            f"Infer: {len(df_clean_infer)}"
        )

        if len(df_clean_train) == 0:
            raise ValueError(
                "Training split is empty. Check your configured "
                "'seasons_train' and the dataset years."
            )

        # 6. Feature Engineering
        logger.info("\n--- Build Features ---")
        X_train, y_train = build_features(df_clean_train)
        X_val, y_val = (
            build_features(df_clean_val) if not df_clean_val.empty else (None, None)
        )
        X_test, y_test = (
            build_features(df_clean_test) if not df_clean_test.empty else (None, None)
        )
        X_infer, y_infer = (
            build_features(df_clean_infer) if not df_clean_infer.empty else (None, None)
        )

        # Fail-fast feature checks for explicitly configured columns
        # on train split
        for col in numeric_pipeline:
            if col in X_train.columns and not pd.api.types.is_numeric_dtype(
                X_train[col]
            ):
                raise TypeError(
                    f"""Column '{col}' mapped for numeric_pipeline
                    must be numeric."""
                )

        # 7. Build Feature Recipe (PreProcessor)
        logger.info("\n--- Build Preprocessor ---")
        preprocessor = get_feature_preprocessor(
            numeric_cols=numeric_pipeline, categorical_cols=categorical_pipeline
        )

        # 8. Train Model
        logger.info("\n--- Train Pipeline ---")
        pipeline = train_model(X_train, y_train, preprocessor, config.get("model", {}))

        # Save Model Artifact
        model_path = models_dir / "model.joblib"
        save_model(pipeline, model_path)

        # 9. Evaluate
        logger.info("\n--- Evaluate Model ---")
        if X_val is not None and len(X_val) > 0:
            logger.info("Evaluating on Validation Set:")
            val_metrics = evaluate_model(pipeline, X_val, y_val, problem_type)
            logger.info(f"Val Metrics: {val_metrics}")

        if X_test is not None and len(X_test) > 0:
            logger.info("Evaluating on Test Set:")
            test_metrics = evaluate_model(pipeline, X_test, y_test, problem_type)
            logger.info(f"Test Metrics: {test_metrics}")

        # 10. Inference
        logger.info("\n--- Run Inference ---")
        if X_infer is not None and len(X_infer) > 0:
            predictions_path = reports_dir / "predictions.csv"
            run_inference(pipeline, X_infer, save_path=str(predictions_path))
        else:
            print(
                "No inference data provided in the split (seasons_infer). "
                "Skipping inference."
            )

    logger.info("\n=== Pipeline Execution Complete ===")


if __name__ == "__main__":
    main()
