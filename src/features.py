"""
Module: Feature Engineering
---------------------------
Role: Define the transformation "recipe" (binning, encoding, scaling)
      to be bundled with the model.
Input: Configuration (lists of column names).
Output: scikit-learn ColumnTransformer object.

Educational Goal:
- Why this module exists in an MLOps system: Encapsulates all mathematical
  transformations into a deployable artifact.
- Responsibility (separation of concerns): Building a transformation recipe
  (not applying it directly to data here).
- Pipeline contract (inputs and outputs): Takes configuration lists, outputs
  an unfitted scikit-learn ColumnTransformer.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported
      from config.yml in a later session
"""

from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_features(df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs the feature matrix (X) and target vector (y) for tennis match prediction.
    Randomly assigns the winner/loser to player_1/player_2 to prevent target leakage
    and create a perfectly balanced dataset (50% player_1_win).

    Returns:
        X: The feature DataFrame.
        y: The binary target Series (1 if player_1 wins, 0 otherwise).
    """
    df_features = df_clean.copy()

    # 1. Randomly decide if winner is p1 or p2 for each row
    # Ensure determinism for reproducibility via localized generator
    rng = np.random.default_rng(42)
    is_winner_p1 = rng.random(len(df_features)) > 0.5

    # 2. Extract features without leakage
    # We create p1_* and p2_* features based on the random assignment

    # Ranks
    df_features["p1_rank"] = np.where(
        is_winner_p1, df_features["winner_rank"], df_features["loser_rank"]
    )
    df_features["p2_rank"] = np.where(
        is_winner_p1, df_features["loser_rank"], df_features["winner_rank"]
    )
    df_features["rank_diff"] = df_features["p1_rank"] - df_features["p2_rank"]

    # Age (demographics)
    if "winner_age" in df_features.columns and "loser_age" in df_features.columns:
        p1_age = np.where(
            is_winner_p1, df_features["winner_age"], df_features["loser_age"]
        )
        p2_age = np.where(
            is_winner_p1, df_features["loser_age"], df_features["winner_age"]
        )
        df_features["age_diff"] = p1_age - p2_age

    # Height (demographics)
    if "winner_ht" in df_features.columns and "loser_ht" in df_features.columns:
        p1_ht = np.where(
            is_winner_p1, df_features["winner_ht"], df_features["loser_ht"]
        )
        p2_ht = np.where(
            is_winner_p1, df_features["loser_ht"], df_features["winner_ht"]
        )
        df_features["ht_diff"] = p1_ht - p2_ht

    # Hand (categorical demographics)
    if "winner_hand" in df_features.columns and "loser_hand" in df_features.columns:
        df_features["p1_hand"] = np.where(
            is_winner_p1, df_features["winner_hand"], df_features["loser_hand"]
        )
        df_features["p2_hand"] = np.where(
            is_winner_p1, df_features["loser_hand"], df_features["winner_hand"]
        )

    # The target: Did player 1 win?
    y = pd.Series(is_winner_p1.astype(int), name="player_1_win")

    # Context features preserved: surface, tourney_level, round
    # Drop raw winner/loser columns to guarantee no leakage
    leakage_cols = [
        c
        for c in df_features.columns
        if c.startswith("winner_")
        or c.startswith("loser_")
        or c.startswith("w_")
        or c.startswith("l_")
    ]
    df_features = df_features.drop(columns=leakage_cols)

    # Ensure ID columns are dropped if they exist
    # as they are not predictive features
    id_cols = ["tourney_id", "match_num"]
    df_features = df_features.drop(
        columns=[c for c in id_cols if c in df_features.columns]
    )

    # Date should also not be passed directly to
    # the model as a numeric/categorical
    if "tourney_date" in df_features.columns:
        df_features = df_features.drop(columns=["tourney_date"])

    # Also drop tourney_name as it's highly cardinal
    if "tourney_name" in df_features.columns:
        df_features = df_features.drop(columns=["tourney_name"])

    return df_features, y


def get_feature_preprocessor(
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Inputs:
    - numeric_cols: List of numeric columns to impute and scale.
    - categorical_cols: List of categorical columns to impute and
    one-hot encode.
    Outputs:
    - An unfitted scikit-learn ColumnTransformer.
    Why this contract matters for reliable ML delivery:
    - Prevents data leakage by ensuring transformations are fitted ONLY on
      training data, then seamlessly applied to test/production data.
    """
    print(
        "Building feature engineering ColumnTransformer recipe..."
    )  # TODO: replace with logging later

    numeric_cols = numeric_cols or []
    categorical_cols = categorical_cols or []

    transformers = []

    if numeric_cols:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_cols))

    if categorical_cols:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        # Unspecified columns get dropped at transformation time
    )

    return preprocessor
