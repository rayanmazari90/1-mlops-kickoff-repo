import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import mlflow

from src.train import train_model

def test_train_model_creates_pipeline():
    # Arrange
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [10, 20, 30, 40]
    })
    y_train = pd.Series([0, 1, 0, 1])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['feature1', 'feature2'])
        ]
    )
    
    model_config = {
        "algorithm": "LogisticRegression",
        "hyperparams": {},
        "random_seed": 42
    }
    
    # Act
    with mlflow.start_run():
        pipeline = train_model(X_train, y_train, preprocessor, model_config)
    
    # Assert
    assert isinstance(pipeline, Pipeline)
    assert hasattr(pipeline, "predict")
    
    # Check that it fits properly
    assert hasattr(pipeline.named_steps["model"], "classes_")
