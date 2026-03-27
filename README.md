# Tennis ATP Match Prediction — Production MLOps Pipeline

[![CI Pipeline](https://github.com/rayanmazari90/1-mlops-kickoff-repo/actions/workflows/ci.yml/badge.svg)](https://github.com/rayanmazari90/1-mlops-kickoff-repo/actions/workflows/ci.yml)

**Group 8** — Rayane Boumediene Mazari, Sacha Huberty, Shreya Jha, Smaragda Apostolou, Marco De Palma, Pipe
**Course:** MLOps Engineering — MsC in Business Analytics and Data Science, IE University

| Live Links | |
|---|---|
| **Prediction UI** | [https://mlops-group8.streamlit.app](https://mlops-group8.streamlit.app) |
| **API (Render)** | [https://one-mlops-kickoff-repo-1.onrender.com](https://one-mlops-kickoff-repo-1.onrender.com) |
| **API Docs (Swagger)** | [https://one-mlops-kickoff-repo-1.onrender.com/docs](https://one-mlops-kickoff-repo-1.onrender.com/docs) |
| **W&B Dashboard** | [https://wandb.ai/bmazari-ieu2024-ie-university/tennis-atp-prediction](https://wandb.ai/bmazari-ieu2024-ie-university/tennis-atp-prediction) |

### Business presentation website

We ship a **multipage static pitch** (divide-and-conquer chapters) with animations (scroll reveals, KPI count-ups, hover motion), **business KPI framework** + model scoreboard, narrative arc (notebook vs product), architecture/code map, live proof links, ethics/roadmap, and a **10-minute video script** (`VIDEO_SCRIPT.md` — talking points only, not for reading verbatim on camera).

| Chapter | File | Content |
|---|---|---|
| Start | [`docs/presentation/index.html`](docs/presentation/index.html) | Hero, three promises, chapter cards |
| KPIs | [`docs/presentation/kpi.html`](docs/presentation/kpi.html) | Business KPI table, north-star metrics, model vs baseline |
| Story | [`docs/presentation/story.html`](docs/presentation/story.html) | Persona, pain/gain, notebook vs MLOps |
| Platform | [`docs/presentation/platform.html`](docs/presentation/platform.html) | Architecture ASCII, `src/` map, API soundbite |
| Proof | [`docs/presentation/proof.html`](docs/presentation/proof.html) | Live Swagger, Streamlit, W&amp;B, CI |
| Close | [`docs/presentation/close.html`](docs/presentation/close.html) | Ethics, roadmap, script pointer |
| Script | [`docs/presentation/VIDEO_SCRIPT.md`](docs/presentation/VIDEO_SCRIPT.md) | Minute-by-minute delivery outline |

**Assets:** [`docs/presentation/assets/styles.css`](docs/presentation/assets/styles.css), [`docs/presentation/assets/app.js`](docs/presentation/assets/app.js)

- **Open locally:** open `docs/presentation/index.html` in a browser (or `python -m http.server` from `docs/presentation` for relative asset loading if needed).
- **GitHub Pages (optional):** **Settings → Pages →** branch **`main`**, folder **`/docs`**. Site root: `https://<user>.github.io/1-mlops-kickoff-repo/` — open **`/presentation/`** for this deck.

---

## 1. Business Case

We built an automated, production-grade ML pipeline that predicts the probability of a player winning an ATP tennis match **before the match starts**. Our target user is a sports analyst, bettor, or trading desk operator who needs calibrated pre-match win probabilities to identify value opportunities and manage risk.

**Why this matters:** Manual scouting and gut-feel predictions are inconsistent and unscalable. Our system replaces them with a reproducible, version-controlled, continuously-deployed prediction service that any stakeholder can query via a simple API call or our visual interface — no data-science knowledge required.

**Business KPIs we optimise for:**

- Outperform the "always pick the higher-ranked player" baseline on **Log Loss** and **Brier Score** (calibration).
- Maintain pipeline reliability — every code change is gated by CI, every deployment is tied to a formal GitHub Release.

### Key Results

| Metric | Our Model | Baseline (Rank Heuristic) | Improvement |
|---|---|---|---|
| **Accuracy** | 65.3% | 64.9% | +0.4pp |
| **Log Loss** | 0.637 | 12.665 | **95% better** |
| **Brier Score** | 0.223 | 0.351 | **37% better** |
| **ROC AUC** | 0.706 | 0.649 | +0.057 |

Our model produces dramatically better-calibrated probabilities than the simple rank heuristic, which is the key value for betting and risk-management use cases.

---

## 2. Architecture

```text
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ config.yaml │───>│  main.py    │───>│    W&B      │───>│  api.py     │
│ .env        │    │ orchestrator│    │  tracking   │    │  FastAPI    │
└─────────────┘    │             │    │  registry   │    │ /health     │
                   │ load_data   │    │  artifacts  │    │ /predict    │
                   │ clean_data  │    └──────┬──────┘    └──────┬──────┘
                   │ validate    │           │ prod             │
                   │ features    │           │ alias            │
                   │ train       │           ▼                  │
                   │ evaluate    │    ┌─────────────┐    ┌──────▼──────┐
                   │ infer       │    │  model.     │    │   Render    │
                   └─────────────┘    │  joblib     │    │   Docker    │
                                      └─────────────┘    └──────┬──────┘
                   ┌─────────────┐                              │
                   │  app.py     │──────────────────────────────┘
                   │  Streamlit  │        calls /predict
                   └─────────────┘
```

**Key components:**

| Module | Responsibility |
|---|---|
| `src/main.py` | Pipeline orchestrator — config loading, W&B init, full data flow |
| `src/load_data.py` | Idempotent data ingestion from JeffSackmann ATP CSVs |
| `src/clean_data.py` | Column standardisation, deduplication, date parsing, rank imputation |
| `src/validate.py` | Pandera schema enforcement — fail-fast quality gate |
| `src/features.py` | Leakage-free feature construction + ColumnTransformer recipe |
| `src/train.py` | sklearn Pipeline fitting (RandomForest/LogisticRegression/Ridge) |
| `src/evaluate.py` | Metrics computation, baseline comparison, W&B logging |
| `src/infer.py` | Batch inference with strict DataFrame contract |
| `src/api.py` | FastAPI + Pydantic serving — `/health`, `/predict` |
| `src/app.py` | Streamlit UI with real ATP players, flags, insights |
| `src/logger.py` | Dual-output logger (console + file), zero `print()` policy |
| `src/utils.py` | I/O helpers (CSV read/write, model serialisation) |
| `notebooks/` | EDA notebook reading from production modules |

---

## 3. Quick Start

### 3.1 Local Environment

```bash
# Option A: conda (recommended)
conda env create -f environment.yml
conda activate mlops-student-env

# Option B: pip
pip install -e ".[dev]"
```

### 3.2 Configure Secrets

```bash
cp .env.example .env
# Edit .env with your real W&B API key and entity:
#   WANDB_API_KEY=your_key
#   WANDB_ENTITY=your_entity
#   WANDB_PROJECT=tennis-atp-prediction
```

### 3.3 Run the Full Pipeline

```bash
python -m src.main
```

This downloads ATP data, cleans, validates, splits by season, builds features, trains a RandomForest, evaluates against a rank-based baseline, runs inference, and logs everything to W&B.

### 3.4 Run Tests

```bash
pytest
# 62 tests, 93% coverage, enforced >= 90% threshold
```

### 3.5 Start the API Locally

```bash
uvicorn src.api:app --reload
```

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"surface":"Hard","tourney_level":"G","round":"F","p1_rank":1,"p2_rank":10}'
```

### 3.6 Start the UI Locally

```bash
pip install streamlit
streamlit run src/app.py
```

### 3.7 Docker

```bash
docker build -t tennis-mlops .
docker run -p 8000:8000 --env-file .env tennis-mlops
```

---

## 4. Configuration

All non-secret runtime settings live in **`config.yaml`**:

| Section | What it controls |
|---|---|
| `paths` | Directories for raw data, processed data, models, reports |
| `dataset` | Base URL, season splits (train: 2018-2020, val: 2021, test: 2022, infer: 2023) |
| `schema` | Required columns, allowed surfaces, target column, non-null columns |
| `model` | Algorithm (RandomForestClassifier), hyperparameters, random seed |
| `features` | Numeric pipeline columns, categorical pipeline columns |
| `wandb` | W&B project name and entity |
| `problem_type` | `classification` (supports `regression` too) |

**Secrets** (`WANDB_API_KEY`, `WANDB_ENTITY`) live exclusively in `.env` (gitignored). The `.env.example` file documents the expected variables without exposing values.

---

## 5. Experiment Tracking and Model Registry (W&B)

We use **Weights & Biases** for experiment tracking, metric logging, and the model registry.

- **Project:** [`tennis-atp-prediction`](https://wandb.ai/bmazari-ieu2024-ie-university/tennis-atp-prediction)
- **Entity:** `bmazari-ieu2024-ie-university`

**What we track per run:**
- Full pipeline config (algorithm, hyperparameters, feature lists, split strategy)
- Evaluation metrics: log loss, brier score, accuracy, AUC
- Baseline comparison metrics (rank heuristic)
- Trained model artifact (`.joblib`)

**Model registry:** The production model is stored as a W&B artifact with alias `prod`. The API (`src/api.py`) loads this artifact on startup via `wandb.use_artifact("tennis-model:prod")`, ensuring the deployed model is always the one explicitly promoted — never an unmanaged local file.

**W&B initialisation** happens centrally in `main.py` — no other module calls `wandb.init()`.

---

## 6. API Contract

Our API is built with **FastAPI** and validated by **Pydantic**. The `src/api.py` module contains zero ML logic — it only calls existing `src/` modules for data processing and inference.

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

- **Swagger UI:** [https://one-mlops-kickoff-repo-1.onrender.com/docs](https://one-mlops-kickoff-repo-1.onrender.com/docs)
- **ReDoc:** [https://one-mlops-kickoff-repo-1.onrender.com/redoc](https://one-mlops-kickoff-repo-1.onrender.com/redoc)

---

## 7. Deployment

### Render (Docker)

Our API is deployed on **Render** as a **Docker** web service.

- **Public URL:** [https://one-mlops-kickoff-repo-1.onrender.com](https://one-mlops-kickoff-repo-1.onrender.com)
- **Runtime:** Docker (`python:3.11-slim`)
- **Deploy trigger:** Publishing a **GitHub Release** from `main` fires the `deploy.yml` workflow, which hits the Render Deploy Hook.
- **Environment variables:** `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT` (set in Render dashboard, never committed)
- **Note:** The free-tier Render instance spins down after inactivity. The first request may take ~30 seconds while it wakes up.

### Docker (Local)

```bash
docker build -t tennis-mlops .
docker run -p 8000:8000 --env-file .env tennis-mlops
```

The `Dockerfile` uses `python:3.11-slim`, installs only production dependencies, and serves the FastAPI app via Uvicorn. A strict `.dockerignore` excludes tests, notebooks, data, reports, wandb artifacts, and dev tooling.

The deployment configuration is also version-controlled in `render.yaml` (Render Blueprint).

---

## 8. Interactive UI

We built a **Streamlit** interface (`src/app.py`) that lets non-technical stakeholders interact with the model without any coding knowledge.

**Live URL:** [https://mlops-group8.streamlit.app](https://mlops-group8.streamlit.app)

### Features

- **22 pre-loaded ATP players** (Sinner, Djokovic, Alcaraz, Zverev, Medvedev, Nadal, Federer, ...) with real rankings, hand, age, height, and country flags
- **Custom Player** option for anyone not in the preset list
- Surface selector with visual indicators (Hard/Clay/Grass)
- Tournament level (Grand Slam, Masters, ATP 500/250, Davis Cup, Tour Finals) and round
- **Visual match outcome** — winner/runner-up cards with probability percentages, progress bar
- **Match context pills** — surface, tournament, round, rank gap, edge rating
- **Match insight** — contextual analysis (heavy favourite / moderate edge / coin-flip)
- **Footer** with model performance stats and links to GitHub, W&B, and API docs

---

## 9. Exploratory Data Analysis

The `notebooks/baseline_eda.ipynb` notebook provides a complete exploratory data analysis of the ATP dataset, following the same pipeline modules used in production:

- Dataset overview and shape analysis
- Surface distribution and match counts by year
- Ranking distribution analysis
- Feature correlation exploration
- Target balance verification (leakage-free P1/P2 assignment)
- Baseline model comparison (rank heuristic vs trained model)
- Metric visualizations

The notebook imports directly from `src/` modules, ensuring the analysis stays consistent with the production pipeline.

---

## 10. Monitoring and Observability

We achieve operational traceability through three complementary channels:

1. **Local log file** — `logs/pipeline.log` captures every pipeline step with timestamps, severity, and module name. Written by `src/logger.py` which provides dual output (console + file). Zero `print()` statements in production code.
2. **W&B run logs** — every training run, metric, and artifact is tracked in our [W&B project](https://wandb.ai/bmazari-ieu2024-ie-university/tennis-atp-prediction) with full parameter lineage and run comparison.
3. **Render service logs** — the deployed API logs requests, model loading status, and prediction errors directly in the Render dashboard.

---

## 11. CI/CD

### Continuous Integration (`ci.yml`)

Runs on every PR and push to `main` or `dev`:

1. **Black** — code formatting check
2. **Flake8** — linting (max line length 88)
3. **Pytest** — full test suite with **coverage enforcement (>= 90%)**

W&B runs in `offline` mode during CI (`WANDB_MODE=offline`) to avoid needing credentials on the runner.

### Continuous Deployment (`deploy.yml`)

Triggered only when a **GitHub Release is published** from `main`. Fires a webhook to Render to redeploy the Docker container.

### Branch Workflow

We follow a clean branching strategy:

- `main` — production-ready code only, protected (requires PR + CI pass)
- `dev` — integration branch, protected (requires PR)
- `feature/*` — individual feature branches merged into `dev` via PRs
- All PRs must pass CI before merge
- No loose branch clutter — merged branches are deleted

### Pull Request History

| PR | Feature | Author |
|---|---|---|
| #13 | Centralized logger + zero print() | @sachahuberty |
| #14 | W&B experiment tracking + model registry | @Shreyajha1911 |
| #15 | FastAPI + Pydantic API serving | @mdepalma-tech |
| #16 | Dockerfile + .dockerignore | @smaragdapostolou |
| #17 | CI coverage enforcement + deploy workflow | @Pipe10101 |
| #18 | Test suite (62 tests, 93% coverage) + render.yaml | @rayanmazari90 |
| #19 | Streamlit UI | @sachahuberty |
| #20 | README + conda-lock + API docs | @Shreyajha1911 |
| #21 | Release v0.2.0 (dev -> main) | All |

---

## 12. Testing

We maintain a comprehensive test suite covering all production modules:

| Test file | Tests | What it covers |
|---|---|---|
| `test_api.py` | 14 | /health, /predict, model loading (W&B + local), edge cases |
| `test_evaluate.py` | 7 | Classification, regression, baseline, no-proba, multiclass |
| `test_train.py` | 5 | LogisticRegression, RandomForest, Ridge, unsupported, defaults |
| `test_load_data.py` | 6 | Local files, download, failure cleanup, empty seasons, bad CSV |
| `test_clean_data.py` | 7 | Column names, duplicates, dates, ranks, surface, index |
| `test_features.py` | 5 | Shapes, leakage, symmetric construction, preprocessor |
| `test_validate.py` | 8 | Empty, missing cols, bad surface, dates, nulls, ranks, target |
| `test_infer.py` | 4 | Success, duck-typing, numpy rejection, path loading |
| `test_main.py` | 1 | Full end-to-end pipeline integration |
| `test_utils.py` | 2 | CSV round-trip, model round-trip |
| `test_logger.py` | 3 | Logger type, dual handlers, idempotency |
| **Total** | **62** | **93% coverage** |

Coverage enforcement: `--cov-fail-under=90` in `pytest.ini`. `src/app.py` (Streamlit) is excluded from coverage measurement.

---

## 13. Model Card

| Field | Value |
|---|---|
| **Task** | Binary classification: will Player 1 win the ATP match? |
| **Algorithm** | RandomForestClassifier (100 trees, max_depth=5, configurable via `config.yaml`) |
| **Features** | `p1_rank`, `p2_rank`, `rank_diff`, `age_diff`, `ht_diff`, `surface`, `tourney_level`, `round`, `p1_hand`, `p2_hand` |
| **Training data** | 7,165 ATP matches (2018–2020 seasons, JeffSackmann dataset) |
| **Validation data** | 2,733 matches (2021 season) |
| **Test data** | 2,917 matches (2022 season) |
| **Inference data** | 2,933 matches (2023 season) |
| **Leakage prevention** | Winner/loser randomly assigned to P1/P2; all post-match stats dropped; preprocessing fitted only on train split |
| **Baseline** | Always predict the higher-ranked player wins (rank heuristic) |
| **Accuracy** | 65.3% (test) vs 64.9% (baseline) |
| **Log Loss** | 0.637 (test) vs 12.665 (baseline) |
| **Brier Score** | 0.223 (test) vs 0.351 (baseline) |
| **AUC** | 0.706 (test) vs 0.649 (baseline) |
| **Limitations** | No in-match statistics, no head-to-head history, no injury/fatigue data. Model may degrade under concept drift as player strengths evolve. Performance is strongest on hard court data (majority of training set). |
| **Ethical considerations** | Model is designed for decision support, not autonomous betting. Users should combine model output with domain knowledge. |

---

## 14. Repository Structure

```text
.
├── .github/workflows/
│   ├── ci.yml                  # PR quality gate (black, flake8, pytest --cov)
│   └── deploy.yml              # GitHub Release → Render deploy hook
├── src/
│   ├── __init__.py
│   ├── api.py                  # FastAPI serving (/health, /predict)
│   ├── app.py                  # Streamlit UI with real ATP players
│   ├── clean_data.py           # Data cleaning and standardisation
│   ├── evaluate.py             # Metrics computation + baseline comparison
│   ├── features.py             # Feature engineering recipe
│   ├── infer.py                # Batch inference
│   ├── load_data.py            # Data ingestion from upstream CSVs
│   ├── logger.py               # Centralized dual-output logger
│   ├── main.py                 # Pipeline orchestrator (W&B init here)
│   ├── train.py                # Model training (sklearn Pipeline)
│   ├── utils.py                # I/O helpers
│   └── validate.py             # Pandera schema validation
├── tests/                      # 62 pytest tests, 93% coverage
├── notebooks/
│   └── baseline_eda.ipynb      # Exploratory data analysis notebook
├── data/                       # Raw + processed (gitignored)
├── models/                     # Model artifacts (gitignored)
├── reports/                    # Metrics JSON + predictions CSV (gitignored)
├── logs/                       # Pipeline logs (gitignored)
├── config.yaml                 # All non-secret runtime configuration
├── .env.example                # Secret template (WANDB_API_KEY, etc.)
├── pyproject.toml              # Package definition + dependencies
├── environment.yml             # Conda environment specification
├── conda-lock.yml              # Locked reproducible environment
├── Dockerfile                  # Lean serving image (python:3.11-slim)
├── .dockerignore               # Strict Docker exclusion policy
├── render.yaml                 # Render Blueprint (deployment config)
├── pytest.ini                  # Test configuration with coverage enforcement
└── README.md                   # This file
```

---

## 15. Changelog

| Version | Date | Changes |
|---|---|---|
| **v1.0.0** | 2026-03-27 | **Stable release:** package (`pyproject.toml`) and OpenAPI metadata aligned at 1.0.0; first GitHub Release tag. Includes everything from v0.2.0 below — this is our production milestone for grading and deployment. |
| **v0.2.0** | 2026-03-27 | Production upgrade: W&B tracking + model registry (`prod` alias), FastAPI + Pydantic API (`/health`, `/predict`), Streamlit UI with 22 real ATP players and match insights, Docker + `.dockerignore`, `deploy.yml` (GitHub Release triggers Render), `src/logger.py` (zero `print()`, dual output), `conda-lock.yml`, EDA notebook, expanded test suite (62 tests, 93% coverage), live deployment on Render + Streamlit Cloud, full README rewrite with model card |
| **v0.1.0** | 2026-03-01 | Initial PoC: sklearn pipeline, MLflow tracking, Pandera validation, basic pytest suite, CI via GitHub Actions |

---

*Built with scikit-learn, FastAPI, Weights & Biases, Streamlit, Docker, and Render.*
*Group 8 — IE University MsC Business Analytics & Data Science — MLOps 2026*
