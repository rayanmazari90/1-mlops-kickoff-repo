"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training using Pandera.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.
"""

import pandas as pd
import pandera as pa
from pandera import Check, Column

from src.logger import get_logger

logger = get_logger(__name__)


def validate_dataframe(df: pd.DataFrame, config: dict) -> bool:
    """
    Validates the dataframe against a strictly defined Pandera DataFrameSchema.

    Why this contract matters:
    - Fails fast cleanly without manual iteration. It explicitly documents
      types, value ranges, and constraints.
    """
    logger.info("Validating dataframe schema using Pandera ...")

    if df.empty:
        raise ValueError("Validation failed: DataFrame is completely empty.")

    schema_cfg = config.get("schema", {})
    allowed_surfaces = schema_cfg.get(
        "allowed_surfaces", ["Hard", "Clay", "Grass", "Carpet"]
    )
    target_col = schema_cfg.get("target", "player_1_win")
    config_required_cols = schema_cfg.get("required_columns", [])

    missing_cols = [col for col in config_required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Validation failed: Missing required columns: {missing_cols}")

    schema_dict = {}

    if "tourney_date" in config_required_cols or "tourney_date" in df.columns:
        schema_dict["tourney_date"] = Column(
            (
                pd.DatetimeTZDtype(tz=None)
                if pd.core.dtypes.common.is_datetime64tz_dtype(
                    df.dtypes.get("tourney_date")
                )
                else "datetime64[ns]"
            ),
            nullable=False,
        )

    if "surface" in config_required_cols or "surface" in df.columns:
        schema_dict["surface"] = Column(
            str, Check.isin(allowed_surfaces), nullable=False
        )

    for col in ["winner_id", "loser_id"]:
        if col in config_required_cols or col in df.columns:
            schema_dict[col] = Column(float, nullable=False)

    for col in ["winner_rank", "loser_rank"]:
        if col in config_required_cols or col in df.columns:
            schema_dict[col] = Column(float, Check.greater_than(0), nullable=False)

    if target_col in config_required_cols or target_col in df.columns:
        schema_dict[target_col] = Column(
            float, Check.isin([0.0, 1.0, 0, 1]), nullable=False
        )

    schema = pa.DataFrameSchema(
        columns=schema_dict,
        strict=False,
        coerce=True,
    )

    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        logger.error("Pandera Validation Errors:\n%s", err.failure_cases)
        error_msg = str(err.failure_cases)
        err_str = str(err)

        if "COLUMN_NOT_IN_DATAFRAME" in err_str or "not in dataframe" in error_msg:
            raise ValueError("Missing required columns") from err
        if (
            "not in allowed_values" in error_msg
            or "isin" in error_msg
            and "surface" in error_msg
        ):
            raise ValueError("Invalid surfaces found") from err
        if "not_nullable" in error_msg or "isnull" in error_msg or "NaN" in error_msg:
            raise ValueError("contains null values") from err
        if (
            "greater_than" in error_msg
            or "minimum" in error_msg
            or "less_than" in error_msg
            and ("rank" in error_msg or "rank" in err_str)
        ):
            raise ValueError("contains non-positive ranks") from err
        if (
            "datetime" in error_msg
            or "type" in error_msg
            or "dtype" in error_msg
            and "tourney_date" in err_str
        ):
            raise ValueError("tourney_date contains invalid date formats") from err
        if "isin" in error_msg and "target_col" in error_msg:
            raise ValueError("contains values other than 0 and 1") from err

        raise ValueError(f"Validation failed: {error_msg}") from err

    return True
