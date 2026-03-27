import joblib

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from src import api
from src.api import app, _load_model_local


class FakeModel:
    def predict(self, X):
        return np.array([1] * len(X))

    def predict_proba(self, X):
        return np.array([[0.3, 0.7]] * len(X))


class FakeModelNoProba:
    def predict(self, X):
        return np.array([1] * len(X))


@pytest.fixture(autouse=True)
def _set_wandb_offline(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")


@pytest.fixture
def client():
    api.MODEL = FakeModel()
    api.CONFIG = {"problem_type": "classification"}
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    api.MODEL = None
    api.CONFIG = None


def _no_startup_client():
    """Return a TestClient with startup events disabled."""
    original = app.router.on_startup
    app.router.on_startup = []
    c = TestClient(app, raise_server_exceptions=False)
    app.router.on_startup = original
    return c


def test_health_endpoint_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_endpoint_no_model(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    api.MODEL = None

    original_startup = app.router.on_startup
    app.router.on_startup = []
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/health")
    finally:
        app.router.on_startup = original_startup

    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is False


def test_predict_endpoint_success(client):
    payload = {
        "surface": "Hard",
        "tourney_level": "G",
        "round": "F",
        "p1_rank": 1.0,
        "p2_rank": 10.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert "probability" in body
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_auto_computes_rank_diff(client):
    payload = {
        "surface": "Clay",
        "tourney_level": "M",
        "round": "QF",
        "p1_rank": 5.0,
        "p2_rank": 20.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200


def test_predict_missing_required_field(client):
    payload = {"surface": "Hard", "tourney_level": "G"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_rank(client):
    payload = {
        "surface": "Hard",
        "tourney_level": "G",
        "round": "F",
        "p1_rank": -1.0,
        "p2_rank": 10.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_no_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    api.MODEL = None

    original_startup = app.router.on_startup
    app.router.on_startup = []
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            payload = {
                "surface": "Hard",
                "tourney_level": "G",
                "round": "F",
                "p1_rank": 1.0,
                "p2_rank": 10.0,
            }
            resp = c.post("/predict", json=payload)
    finally:
        app.router.on_startup = original_startup

    assert resp.status_code == 503


def test_predict_model_without_predict_proba(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    api.MODEL = FakeModelNoProba()
    api.CONFIG = {}

    original_startup = app.router.on_startup
    app.router.on_startup = []
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            payload = {
                "surface": "Hard",
                "tourney_level": "G",
                "round": "F",
                "p1_rank": 1.0,
                "p2_rank": 10.0,
                "p1_hand": None,
                "p2_hand": None,
                "age_diff": None,
                "ht_diff": None,
            }
            resp = c.post("/predict", json=payload)
    finally:
        app.router.on_startup = original_startup

    assert resp.status_code == 200
    assert resp.json()["probability"] == 1.0


def test_load_model_local_with_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    joblib.dump(model, models_dir / "model.joblib")

    cfg = {"paths": {"models_dir": "models"}}
    with open(tmp_path / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    loaded = _load_model_local()
    assert hasattr(loaded, "predict")


def test_load_model_local_no_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    from sklearn.linear_model import LogisticRegression

    joblib.dump(LogisticRegression(), models_dir / "model.joblib")

    loaded = _load_model_local()
    assert hasattr(loaded, "predict")


def test_load_model_local_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="Local model not found"):
        _load_model_local()


def test_predict_with_explicit_rank_diff(client):
    payload = {
        "surface": "Grass",
        "tourney_level": "G",
        "round": "SF",
        "p1_rank": 3.0,
        "p2_rank": 15.0,
        "rank_diff": -12.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
