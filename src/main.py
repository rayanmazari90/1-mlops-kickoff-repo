"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow (Load -> Clean -> Validate -> Train -> Evaluate).
Usage: python -m src.main
"""
"""
Educational Goal:
- Why this module exists in an MLOps system: The orchestrator script that ties all components together.
- Responsibility (separation of concerns): Defining execution order, passing data between modules, and configuration management.
- Pipeline contract (inputs and outputs): Execution entry point that reads configuration and orchestrates artifacts.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

# Internal package imports
from src.utils import save_csv, save_model
from src.load_data import load_raw_data
from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.features import get_feature_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import run_inference

# ---------------------------------------------------------
# CONFIGURATION BLOCK (Bridge to YAML)
# LOUD COMMENT: Students MUST map this SETTINGS block to their real dataset!
# ---------------------------------------------------------
SETTINGS = {
    "is_example_config": False,
    "target_column": "player_1_win",
    "problem_type": "classification",
    "features": {
        "numeric_pipeline": ["p1_rank", "p2_rank", "rank_diff", "age_diff", "ht_diff"],
        "categorical_pipeline": ["surface", "tourney_level", "round", "p1_hand", "p2_hand"],
    }
}

def main():
    print("=== Starting MLOps Pipeline ===") # TODO: replace with logging later
    
    # 1. Directory Creation
    print("\n--- Setup Directories ---")
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    
    if SETTINGS.get("is_example_config"):
        print("Note: Running with dummy/example configuration.")
        
    # 2. Load
    print("\n--- Load Data ---")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    dataset_cfg = config.get("dataset", {})
    base_url = dataset_cfg.get("base_url")
    
    # Combine all seasons meant for the entire pipeline
    # The split step will later separate train/val/test/infer using Date/Seasons
    seasons = []
    for split_type in ["seasons_train", "seasons_val", "seasons_test", "seasons_infer"]:
        seasons.extend(dataset_cfg.get(split_type, []))
        
    paths_cfg = config.get("paths", {})
    raw_dir = Path(paths_cfg.get("raw_dir", "data/raw"))
    
    df_raw = load_raw_data(
        raw_dir=raw_dir,
        base_url=base_url,
        seasons=seasons,
        download_if_missing=True
    )
    
    # 3. Clean
    print("\n--- Clean Data ---")
    df_clean = clean_dataframe(df_raw, "player_1_win") # The target doesn't exist yet but the validator might check it later
    
    # 4. Validate
    print("\n--- Validate Data ---")
    validate_dataframe(df_clean, config=config)
    
    # 5. Feature Engineering (NEW STEP)
    print("\n--- Build Features ---")
    # We must run build_features before validating the final dataset
    from src.features import build_features
    X_all, y_all = build_features(df_clean)
    
    # Re-combine temporarily if we want to save the "processed" CSV 
    # MLOps note: we save the engineered features + target here
    df_processed = pd.concat([X_all, y_all], axis=1)
    
    # Save processed CSV
    processed_path = Path("data/processed/clean.csv")
    save_csv(df_processed, processed_path)
    
    # Fail-fast feature checks for explicitly configured columns
    # In a later session, this will also use config instead of SETTINGS.
    for col in SETTINGS["features"]["numeric_pipeline"]:
        if col in df_processed.columns and not pd.api.types.is_numeric_dtype(df_processed[col]):
            raise TypeError(f"Column '{col}' mapped for numeric_pipeline must be numeric.")
            
    # 6. Train / Test Split
    print("\n--- Train Test Split ---")
    X = df_processed.drop(columns=[SETTINGS["target_column"]])
    y = df_processed[SETTINGS["target_column"]]
    
    stratify_col = y if SETTINGS["problem_type"] == "classification" else None
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_col
        )
    except ValueError as e:
        print(f"Stratified split failed ({e}). Falling back to unstratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
    # 7. Build Feature Recipe
    print("\n--- Build Preprocessor ---")
    preprocessor = get_feature_preprocessor(
        numeric_cols=SETTINGS["features"]["numeric_pipeline"],
        categorical_cols=SETTINGS["features"]["categorical_pipeline"]
    )
    
    # 8. Train Model
    print("\n--- Train Pipeline ---")
    pipeline = train_model(X_train, y_train, preprocessor, SETTINGS["problem_type"])
    
    # Save Model Artifact
    model_path = Path("models/model.joblib")
    save_model(pipeline, model_path)
    
    # 9. Evaluate
    print("\n--- Evaluate Model ---")
    metric_value = evaluate_model(pipeline, X_test, y_test, SETTINGS["problem_type"])
    
    # 10. Inference on test set (as an example of scoring new data)
    print("\n--- Run Inference ---")
    # In a real environment, X_infer would be totally new unseen data
    df_predictions = run_inference(pipeline, X_test)
    
    # Save Predictions Artifact
    predictions_path = Path("reports/predictions.csv")
    save_csv(df_predictions, predictions_path)
    
    print("\n=== Pipeline Execution Complete ===")

if __name__ == "__main__":
    main()
