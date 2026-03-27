"""
Module: API Serving
-------------------
FastAPI application that serves predictions via a strict Pydantic contract.
No new ML logic lives here — only calls to existing src modules.
"""

import os
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import wandb
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.logger import get_logger

logger = get_logger(__name__)

load_dotenv()

app = FastAPI(
    title="Tennis ATP Match Prediction API",
    version="1.0.0",
    description="Predicts the probability of player 1 winning an ATP tennis match.",
)

MODEL = None
CONFIG = None


class MatchFeatures(BaseModel):
    """Pydantic model enforcing the JSON request contract for /predict."""

    surface: str = Field(..., description="Court surface: Hard, Clay, or Grass")
    tourney_level: str = Field(..., description="Tournament level code (e.g. G, M, A)")
    round: str = Field(..., description="Match round (e.g. F, SF, QF, R16, R32, R64)")
    p1_rank: float = Field(..., gt=0, description="Player 1 ATP ranking")
    p2_rank: float = Field(..., gt=0, description="Player 2 ATP ranking")
    p1_hand: Optional[str] = Field("R", description="Player 1 hand: R, L, or U")
    p2_hand: Optional[str] = Field("R", description="Player 2 hand: R, L, or U")
    rank_diff: Optional[float] = Field(
        None, description="Rank difference (auto-computed if omitted)"
    )
    age_diff: Optional[float] = Field(0.0, description="Age difference (p1 - p2)")
    ht_diff: Optional[float] = Field(0.0, description="Height difference (p1 - p2)")


class PredictionResponse(BaseModel):
    """Pydantic model for the prediction response."""

    prediction: int = Field(..., description="0 or 1 (player 1 win)")
    probability: float = Field(..., description="Win probability for player 1")


class HealthResponse(BaseModel):
    """Pydantic model for the health check response."""

    status: str
    model_loaded: bool


def _load_model_from_wandb():
    """Pull the 'prod' model artifact from W&B."""
    entity = os.environ.get("WANDB_ENTITY", "")
    project = os.environ.get("WANDB_PROJECT", "tennis-atp-prediction")

    wandb.init(project=project, entity=entity, job_type="inference", reinit=True)
    artifact = wandb.use_artifact(f"{entity}/{project}/tennis-model:prod")
    artifact_dir = artifact.download(root="artifacts/model")
    wandb.finish()

    model_files = list(Path(artifact_dir).glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError("No .joblib model found in W&B artifact.")
    return joblib.load(model_files[0])


def _load_model_local():
    """Fallback: load model from local disk."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        model_path = (
            Path(cfg.get("paths", {}).get("models_dir", "models")) / "model.joblib"
        )
    else:
        model_path = Path("models/model.joblib")

    if not model_path.exists():
        raise FileNotFoundError(f"Local model not found at {model_path}")
    return joblib.load(model_path)


@app.on_event("startup")
async def startup_event():
    """Load the model on startup — try W&B first, then fall back to local."""
    global MODEL, CONFIG

    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            CONFIG = yaml.safe_load(f)

    try:
        logger.info("Attempting to load model from W&B artifact (prod) ...")
        MODEL = _load_model_from_wandb()
        logger.info("Model loaded from W&B successfully.")
    except Exception as e:
        logger.warning("W&B model load failed (%s), falling back to local ...", e)
        try:
            MODEL = _load_model_local()
            logger.info("Model loaded from local disk.")
        except FileNotFoundError:
            logger.error("No model available. Run the training pipeline first.")
            MODEL = None


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness / readiness probe."""
    return HealthResponse(status="ok", model_loaded=MODEL is not None)


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: MatchFeatures):
    """Generate a match outcome prediction from the supplied features."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if features.rank_diff is None:
        features.rank_diff = features.p1_rank - features.p2_rank

    row = {
        "surface": features.surface,
        "tourney_level": features.tourney_level,
        "round": features.round,
        "p1_rank": features.p1_rank,
        "p2_rank": features.p2_rank,
        "p1_hand": features.p1_hand or "R",
        "p2_hand": features.p2_hand or "R",
        "rank_diff": features.rank_diff,
        "age_diff": features.age_diff or 0.0,
        "ht_diff": features.ht_diff or 0.0,
    }
    df = pd.DataFrame([row])

    try:
        prediction = int(MODEL.predict(df)[0])
        if hasattr(MODEL, "predict_proba"):
            proba = float(MODEL.predict_proba(df)[0][1])
        else:
            proba = float(prediction)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=422, detail=str(e))

    return PredictionResponse(prediction=prediction, probability=proba)
