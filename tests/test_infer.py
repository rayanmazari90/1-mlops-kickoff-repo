import pytest
import pandas as pd
import joblib
from pathlib import Path
from src.infer import run_inference


class DummyModel:
    def predict(self, X):
        import numpy as np

        return np.array([0] * len(X))

    def predict_proba(self, X):
        import numpy as np

        return np.array([[0.8, 0.2]] * len(X))


class BadModel:
    pass


def test_run_inference_success(tmp_path):
    model = DummyModel()
    df = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])

    # Save dummy model
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    # Run inference loading from object
    df_preds = run_inference(model, df)

    assert "prediction" in df_preds.columns
    assert "proba" in df_preds.columns
    assert list(df_preds.index) == [10, 20, 30]


def test_run_inference_duck_typing():
    model = BadModel()
    df = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(TypeError, match="predict"):
        run_inference(model, df)


def test_run_inference_numpy_fails():
    model = DummyModel()
    import numpy as np

    arr = np.array([[1], [2], [3]])

    with pytest.raises(TypeError, match="DataFrame"):
        run_inference(model, arr)


def test_run_inference_standalone(tmp_path):
    # Test loading model and saving predictions
    model = DummyModel()
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    df = pd.DataFrame({"a": [1, 2]})
    output_path = tmp_path / "predictions.csv"

    # We alter run_inference to support loading model from path and saving output
    df_preds = run_inference(str(model_path), df, save_path=str(output_path))

    assert "prediction" in df_preds.columns
    assert output_path.exists()

    saved_df = pd.read_csv(output_path, index_col=0)
    assert "prediction" in saved_df.columns
