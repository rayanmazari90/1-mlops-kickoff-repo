"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow (Load -> Clean -> Validate -> Train ->
      Evaluate).
Usage: python -m src.main
"""

import os
from pathlib import Path

import pandas as pd
import yaml

from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.features import build_features, get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.logger import get_logger
from src.train import train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe

logger = get_logger(__name__)


def main(config_path: str = "config.yaml"):
    logger.info("=== Starting MLOps Pipeline ===")

    logger.info("--- Load Configuration ---")
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config_str = yaml.dump(config)
    if "TODO" in config_str or "CHANGEME" in config_str:
        raise ValueError(
            "Configuration contains unresolved placeholders (TODO/CHANGEME). "
            "Please configure them properly."
        )

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

    download_if_missing = config.get("dataset", {}).get("download_if_missing", True)

    logger.info("--- Setup Directories ---")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

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

    # --- W&B Tracking (initialized centrally) ---
    import wandb
    from dotenv import load_dotenv

    load_dotenv()

    wandb_cfg = config.get("wandb", {})
    wandb.init(
        project=wandb_cfg.get("project", "tennis-atp-prediction"),
        entity=os.environ.get("WANDB_ENTITY", wandb_cfg.get("entity", "")),
        name="batch_pipeline_run",
        config=config,
        tags=["training", "batch"],
    )

    # 2. Load
    logger.info("--- Load Data ---")
    df_raw = load_raw_data(
        raw_dir=raw_dir,
        base_url=base_url,
        seasons=seasons,
        download_if_missing=download_if_missing,
    )

    # 3. Clean
    logger.info("--- Clean Data ---")
    df_clean = clean_dataframe(df_raw, target_column)

    clean_path = processed_dir / "clean.csv"
    save_csv(df_clean, clean_path)

    # 4. Validate
    logger.info("--- Validate Data ---")
    validate_dataframe(df_clean, config=config)

    # 5. Early Split by seasons
    logger.info("--- Early Train/Val/Test Split ---")
    if "tourney_date" in df_clean.columns:
        year_series = pd.to_datetime(df_clean["tourney_date"], format="mixed").dt.year
    else:
        raise ValueError("tourney_date column missing from cleaned dataframe!")

    df_clean_train = df_clean[year_series.isin(dataset_cfg.get("seasons_train", []))]
    df_clean_val = df_clean[year_series.isin(dataset_cfg.get("seasons_val", []))]
    df_clean_test = df_clean[year_series.isin(dataset_cfg.get("seasons_test", []))]
    df_clean_infer = df_clean[year_series.isin(dataset_cfg.get("seasons_infer", []))]

    logger.info(
        "Split sizes -> Train: %d, Val: %d, Test: %d, Infer: %d",
        len(df_clean_train),
        len(df_clean_val),
        len(df_clean_test),
        len(df_clean_infer),
    )

    if len(df_clean_train) == 0:
        raise ValueError(
            "Training split is empty. Check your configured "
            "'seasons_train' and the dataset years."
        )

    # 6. Feature Engineering
    logger.info("--- Build Features ---")
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

    for col in numeric_pipeline:
        if col in X_train.columns and not pd.api.types.is_numeric_dtype(X_train[col]):
            raise TypeError(
                f"Column '{col}' mapped for numeric_pipeline must be numeric."
            )

    # 7. Build Feature Recipe (PreProcessor)
    logger.info("--- Build Preprocessor ---")
    preprocessor = get_feature_preprocessor(
        numeric_cols=numeric_pipeline, categorical_cols=categorical_pipeline
    )

    # 8. Train Model
    logger.info("--- Train Pipeline ---")
    pipeline = train_model(X_train, y_train, preprocessor, config.get("model", {}))

    model_path = models_dir / "model.joblib"
    save_model(pipeline, model_path)

    # Upload model artifact to W&B and alias as 'prod'
    artifact = wandb.Artifact("tennis-model", type="model")
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact, aliases=["latest", "prod"])
    logger.info("Model artifact uploaded to W&B with alias 'prod'")

    # 9. Evaluate
    logger.info("--- Evaluate Model ---")
    if X_val is not None and len(X_val) > 0:
        logger.info("Evaluating on Validation Set:")
        val_metrics = evaluate_model(pipeline, X_val, y_val, problem_type)
        logger.info("Val Metrics: %s", val_metrics)

    if X_test is not None and len(X_test) > 0:
        logger.info("Evaluating on Test Set:")
        test_metrics = evaluate_model(pipeline, X_test, y_test, problem_type)
        logger.info("Test Metrics: %s", test_metrics)

    # 10. Inference
    logger.info("--- Run Inference ---")
    if X_infer is not None and len(X_infer) > 0:
        predictions_path = reports_dir / "predictions.csv"
        run_inference(pipeline, X_infer, save_path=str(predictions_path))
    else:
        logger.warning(
            "No inference data provided in the split (seasons_infer). "
            "Skipping inference."
        )

    wandb.finish()
    logger.info("=== Pipeline Execution Complete ===")


if __name__ == "__main__":
    main()
