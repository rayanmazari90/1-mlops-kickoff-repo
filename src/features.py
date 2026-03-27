"""
Module: Feature Engineering
---------------------------
Role: Define the transformation "recipe" (binning, encoding, scaling)
      to be bundled with the model.
Input: Configuration (lists of column names).
Output: scikit-learn ColumnTransformer object.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import get_logger

logger = get_logger(__name__)


def build_features(df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs the feature matrix (X) and target vector (y) for tennis match
    prediction. Randomly assigns the winner/loser to player_1/player_2 to
    prevent target leakage and create a balanced dataset.
    """
    df_features = df_clean.copy()

    rng = np.random.default_rng(42)
    is_winner_p1 = rng.random(len(df_features)) > 0.5

    df_features["p1_rank"] = np.where(
        is_winner_p1, df_features["winner_rank"], df_features["loser_rank"]
    )
    df_features["p2_rank"] = np.where(
        is_winner_p1, df_features["loser_rank"], df_features["winner_rank"]
    )
    df_features["rank_diff"] = df_features["p1_rank"] - df_features["p2_rank"]

    if "winner_age" in df_features.columns and "loser_age" in df_features.columns:
        p1_age = np.where(
            is_winner_p1, df_features["winner_age"], df_features["loser_age"]
        )
        p2_age = np.where(
            is_winner_p1, df_features["loser_age"], df_features["winner_age"]
        )
        df_features["age_diff"] = p1_age - p2_age

    if "winner_ht" in df_features.columns and "loser_ht" in df_features.columns:
        p1_ht = np.where(
            is_winner_p1, df_features["winner_ht"], df_features["loser_ht"]
        )
        p2_ht = np.where(
            is_winner_p1, df_features["loser_ht"], df_features["winner_ht"]
        )
        df_features["ht_diff"] = p1_ht - p2_ht

    if "winner_hand" in df_features.columns and "loser_hand" in df_features.columns:
        df_features["p1_hand"] = np.where(
            is_winner_p1, df_features["winner_hand"], df_features["loser_hand"]
        )
        df_features["p2_hand"] = np.where(
            is_winner_p1, df_features["loser_hand"], df_features["winner_hand"]
        )

    y = pd.Series(is_winner_p1.astype(int), name="player_1_win")

    leakage_cols = [
        c
        for c in df_features.columns
        if c.startswith("winner_")
        or c.startswith("loser_")
        or c.startswith("w_")
        or c.startswith("l_")
    ]
    df_features = df_features.drop(columns=leakage_cols)

    id_cols = ["tourney_id", "match_num"]
    df_features = df_features.drop(
        columns=[c for c in id_cols if c in df_features.columns]
    )

    if "tourney_date" in df_features.columns:
        df_features = df_features.drop(columns=["tourney_date"])

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
    - categorical_cols: List of categorical columns to impute and one-hot encode.
    Outputs:
    - An unfitted scikit-learn ColumnTransformer.
    """
    logger.info("Building feature engineering ColumnTransformer recipe ...")

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
    )

    return preprocessor
