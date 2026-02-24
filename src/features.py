"""
Module: Feature Engineering
---------------------------
Role: Define the transformation "recipe" (binning, encoding, scaling) to be bundled with the model.
Input: Configuration (lists of column names).
Output: scikit-learn ColumnTransformer object.
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Encapsulates all mathematical transformations into a deployable artifact.
- Responsibility (separation of concerns): Building a transformation recipe (not applying it directly to data here).
- Pipeline contract (inputs and outputs): Takes configuration lists, outputs an unfitted scikit-learn ColumnTransformer.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from typing import Optional, List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None, 
    categorical_onehot_cols: Optional[List[str]] = None, 
    numeric_passthrough_cols: Optional[List[str]] = None, 
    n_bins: int = 3
):
    """
    Inputs:
    - quantile_bin_cols: List of numeric columns to bin.
    - categorical_onehot_cols: List of categorical columns to encode.
    - numeric_passthrough_cols: List of numeric columns to leave untouched.
    - n_bins: Integer specifying how many bins to use for the discretizer.
    Outputs:
    - An unfitted scikit-learn ColumnTransformer.
    Why this contract matters for reliable ML delivery:
    - Prevents data leakage by ensuring transformations are fitted ONLY on training data, then seamlessly applied to test/production data.
    """
    print("Building feature engineering ColumnTransformer recipe...") # TODO: replace with logging later
    
    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []
    
    transformers = []
    
    if quantile_bin_cols:
        transformers.append(
            ("kbins", KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile'), quantile_bin_cols)
        )
        
    if categorical_onehot_cols:
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        transformers.append(("onehot", ohe, categorical_onehot_cols))
        
    if numeric_passthrough_cols:
        transformers.append(("passthrough", "passthrough", numeric_passthrough_cols))
        
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Feature engineering is highly specific to the signal in your dataset
    # Examples:
    # 1. Add StandardScaler() for numeric columns
    # 2. Add custom text vectorizers for NLP columns
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    
    return preprocessor