# Tennis ATP Match Prediction — Production MLOps Pipeline

[![CI Pipeline](https://github.com/rayanmazari90/1-mlops-kickoff-repo/actions/workflows/ci.yml/badge.svg)](https://github.com/rayanmazari90/1-mlops-kickoff-repo/actions/workflows/ci.yml)

**Group 8** — Rayane Boumediene Mazari, Sacha Huberty, Shreya Jha, Smaragda Apostolou, Marco De Palma, Pipe  
**Course:** MLOps Engineering — MsC in Business Analytics and Data Science  

---

## 1. Business Case

We built an automated, production-grade ML pipeline that predicts the probability of a player winning an ATP tennis match **before the match starts**. Our target user is a sports analyst, bettor, or trading desk operator who needs calibrated pre-match win probabilities to identify value opportunities and manage risk.

**Why this matters:** Manual scouting and gut-feel predictions are inconsistent and unscalable. Our system replaces them with a reproducible, version-controlled, continuously-deployed prediction service that any stakeholder can query via a simple API call or a visual interface — no data-science knowledge required.

**Business KPIs we optimise for:**

- Outperform the "always pick the higher-ranked player" baseline on **Log Loss** and **Brier Score** (calibration).
- Maintain pipeline reliability — every code change is gated by CI, every deployment is tied to a formal GitHub Release.

---

## 2. Architecture

```text
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  config.yaml │────>│   main.py    │────>│    W&B       │
│   .env       │     │ (orchestrator│     │  (tracking,  │
└──────────────┘     │  loads data, │     │   registry)  │
                     │  cleans,     │     └──────┬───────┘
                     │  validates,  │            │ prod
                     │  trains,     │            │ artifact
                     │  evaluates)  │            ▼
                     └──────────────┘     ┌──────────────┐
                                          │   api.py     │
                                          │  (FastAPI)   │
                     ┌──────────────┐     │  /health     │
                     │   app.py     │────>│  /predict    │
                     │ (Streamlit)  │     └──────┬───────┘
                     └──────────────┘            │
                                          ┌──────▼───────┐
                                          │   Render      │
                                          │  (deployed)   │
                                          └──────────────┘
```

**Key components:**

| Module | Responsibility |
|---|---|
| `src/main.py` | Pipeline orchestrator — config, W&B init, data flow |
| `src/load_data.py` | Idempotent data ingestion from upstream CSVs |
| `src/clean_data.py` | Column standardisation, deduplication, rank imputation |
| `src/validate.py` | Pandera schema enforcement (fail-fast quality gate) |
| `src/features.py` | Leakage-free feature construction + ColumnTransformer recipe |
| `src/train.py` | Model fitting inside an sklearn Pipeline |
| `src/evaluate.py` | Metrics computation, baseline comparison, W&B logging |
| `src/infer.py` | Batch inference with strict DataFrame contract |
| `src/api.py` | FastAPI serving layer — Pydantic contract, /health, /predict |
| `src/app.py` | Streamlit UI for interactive predictions |
| `src/logger.py` | Dual-output logger (console + file), zero print() policy |
| `src/utils.py` | I/O helpers (CSV, model serialisation) |

---

## 3. Quick Start

### Local Environment

```bash
# Option A: conda (recommended)
conda env create -f environment.yml
conda activate mlops-student-env

# Option B: pip
pip install -e ".[dev]"
```

### Set Up Secrets

```bash
cp .env.example .env
# Edit .env with your real W&B API key and entity
```

### Run the Full Pipeline

```bash
python -m src.main
```

This will: download data → clean → validate → split → build features → train → evaluate → infer → log everything to W&B.

### Run Tests

```bash
pytest
```

### Start the API

```bash
uvicorn src.api:app --reload
# Health check:  curl http://localhost:8000/health
# Prediction:    curl -X POST http://localhost:8000/predict \
#   -H "Content-Type: application/json" \
#   -d '{"surface":"Hard","tourney_level":"G","round":"F","p1_rank":1,"p2_rank":10}'
```

### Start the UI

```bash
pip install streamlit
streamlit run src/app.py
```

### Docker

```bash
docker build -t tennis-mlops .
docker run -p 8000:8000 --env-file .env tennis-mlops
```

---

## 4. Configuration

All non-secret runtime settings live in `config.yaml`:

- **Dataset:** base URL, season splits (train/val/test/infer)
- **Schema:** required columns, allowed surfaces, target column
- **Model:** algorithm, hyperparameters, random seed
- **Features:** numeric and categorical pipeline columns
- **W&B:** project name, entity

**Secrets** (`WANDB_API_KEY`, `WANDB_ENTITY`) live exclusively in `.env` (gitignored). The `.env.example` file documents the expected variables without exposing values.

---

## 5. Experiment Tracking (W&B)

We use **Weights & Biases** for experiment tracking, metric logging, and model registry.

- **Project:** `tennis-atp-prediction`
- **Entity:** `bmazari-ieu2024-ie-university`
- **What we track:** full pipeline config, algorithm hyperparameters, evaluation metrics (log loss, brier score, accuracy, AUC), baseline comparison metrics, and the trained model artifact.
- **Model registry:** the production model is stored as a W&B artifact aliased `prod`. The API loads this artifact on startup.

> W&B Project link: [https://wandb.ai/bmazari-ieu2024-ie-university/tennis-atp-prediction](https://wandb.ai/bmazari-ieu2024-ie-university/tennis-atp-prediction)

---

## 6. API Contract

### `GET /health`

```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`

**Request body** (Pydantic-validated):

```json
{
  "surface": "Hard",
  "tourney_level": "G",
  "round": "F",
  "p1_rank": 1.0,
  "p2_rank": 10.0,
  "p1_hand": "R",
  "p2_hand": "L",
  "rank_diff": -9.0,
  "age_diff": -2.5,
  "ht_diff": 5.0
}
```

**Response:**

```json
{"prediction": 1, "probability": 0.73}
```

Fields `rank_diff`, `age_diff`, `ht_diff`, `p1_hand`, `p2_hand` are optional — sensible defaults are applied.

### Interactive API Documentation

FastAPI auto-generates interactive documentation:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

When deployed on Render, replace `localhost:8000` with the public Render URL.

---

## 7. Deployment

### Render

Our API is deployed on **Render** as a Docker web service.

- **Public URL:** *(will be provided after first deployment)*
- **Deployment trigger:** Publishing a **GitHub Release** from `main` fires the `deploy.yml` workflow, which hits the Render Deploy Hook.

### Docker

```bash
docker build -t tennis-mlops .
docker run -p 8000:8000 --env-file .env tennis-mlops
```

The Dockerfile uses `python:3.11-slim`, installs only production dependencies, and exposes the FastAPI app via Uvicorn. A strict `.dockerignore` excludes tests, notebooks, data, reports, wandb artifacts, and dev tooling.

---

## 8. Interactive UI (Bonus)

We built a **Streamlit** interface (`src/app.py`) that lets non-technical users interact with the model:

- Select surface, tournament level, round
- Enter both players' ATP rankings, hand, age, and height
- Click "Predict" to see the win probability displayed as a progress gauge
- Colour-coded confidence indicator (green/yellow/blue)

Run it locally:

```bash
API_URL=http://localhost:8000 streamlit run src/app.py
```

---

## 9. Monitoring & Observability

We achieve operational traceability through three complementary channels:

1. **Local log file** — `logs/pipeline.log` captures every pipeline step with timestamps, severity, and module name. Written by `src/logger.py` (dual output: console + file). Zero `print()` statements in production code.
2. **W&B run logs** — every training run, metric, and artifact is tracked in our W&B project with full parameter lineage.
3. **Render service logs** — the deployed API logs requests, model loading status, and prediction errors directly in the Render dashboard.

---

## 10. CI/CD

### Continuous Integration (`ci.yml`)

Runs on every PR and push to `main` or `dev`:

1. **Black** — code formatting check
2. **Flake8** — linting
3. **Pytest** — full test suite with **coverage enforcement (>=70%)**

W&B runs in `offline` mode during CI to avoid needing credentials.

### Continuous Deployment (`deploy.yml`)

Triggered only when a **GitHub Release is published** from `main`. Fires a webhook to Render to redeploy the Docker container.

### Branch Workflow

We follow a clean branching strategy:

- `main` — production-ready code only
- `dev` — integration branch
- `feature/*` — individual feature branches merged into `dev` via PRs
- All PRs must pass CI before merge
- No loose branch clutter

---

## 11. Model Card

| Field | Value |
|---|---|
| **Task** | Binary classification: will Player 1 win the match? |
| **Algorithm** | RandomForestClassifier (configurable via `config.yaml`) |
| **Features** | `p1_rank`, `p2_rank`, `rank_diff`, `age_diff`, `ht_diff`, `surface`, `tourney_level`, `round`, `p1_hand`, `p2_hand` |
| **Training data** | ATP match data 2018–2020 (JeffSackmann dataset) |
| **Validation data** | 2021 season |
| **Test data** | 2022 season |
| **Leakage prevention** | Winner/loser randomly assigned to P1/P2; all post-match stats dropped; features fitted only on train split |
| **Baseline** | Always predict the higher-ranked player wins |
| **Primary metrics** | Log Loss, Brier Score |
| **Secondary metrics** | Accuracy, ROC AUC |
| **Limitations** | No in-match stats, no head-to-head history, no injury data. Model may degrade under concept drift as player strengths shift over time. |

---

## 12. Repository Structure

```text
.
├── .github/workflows/
│   ├── ci.yml              # PR quality gate
│   └── deploy.yml          # Release → Render deploy
├── src/
│   ├── __init__.py
│   ├── api.py              # FastAPI serving (/health, /predict)
│   ├── app.py              # Streamlit UI
│   ├── clean_data.py       # Data cleaning
│   ├── evaluate.py         # Metrics + baseline comparison
│   ├── features.py         # Feature engineering recipe
│   ├── infer.py            # Batch inference
│   ├── load_data.py        # Data ingestion
│   ├── logger.py           # Centralized dual-output logger
│   ├── main.py             # Pipeline orchestrator
│   ├── train.py            # Model training
│   ├── utils.py            # I/O helpers
│   └── validate.py         # Pandera schema validation
├── tests/                  # 45 pytest tests, 80%+ coverage
├── data/                   # Raw + processed (gitignored)
├── models/                 # Model artifacts (gitignored)
├── reports/                # Metrics + predictions (gitignored)
├── logs/                   # Pipeline logs (gitignored)
├── config.yaml             # All runtime configuration
├── .env.example            # Secret template
├── pyproject.toml          # Package + dependency management
├── environment.yml         # Conda environment
├── conda-lock.yml          # Locked reproducible environment
├── Dockerfile              # Lean serving image
├── .dockerignore           # Strict exclusion policy
├── pytest.ini              # Test configuration with coverage
└── README.md               # This file
```

---

## 13. Changelog

| Version | Date | Changes |
|---|---|---|
| **v0.2.0** | 2026-03-27 | Production upgrade: W&B tracking + model registry, FastAPI + Pydantic API (`/health`, `/predict`), Streamlit UI, Docker + `.dockerignore`, `deploy.yml` (GitHub Release → Render), `src/logger.py` (zero `print()`), `conda-lock.yml`, expanded test suite (45 tests, 80%+ coverage), full README rewrite |
| **v0.1.0** | 2026-03-01 | Initial PoC: sklearn pipeline, MLflow tracking, Pandera validation, pytest suite, CI via GitHub Actions |

---

*Built with scikit-learn, FastAPI, Weights & Biases, Streamlit, Docker, and Render.*
