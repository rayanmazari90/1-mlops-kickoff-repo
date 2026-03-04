# Tennis Match Outcome Prediction (ATP) – MLOps Pipeline

**Author:** TODO_STUDENT (Group Name / Number)  
**Course:** MLOps (Master in Business Analytics and Data Science)  
**Status:** Session 1 → Session N (Incremental delivery)

---

## 1. Business Objective

### Client
The primary client is a bettor, sports analyst, or trading bot operator who needs to evaluate betting opportunities.

### The Goal (Business Value)
Build an automated, reliable ML pipeline that predicts the probability of a player winning an ATP match **before the match starts**. This decision-support tool helps identify "value" bets by producing accurately calibrated win probabilities.

### Maturity
The project is currently in the initial **Proof of Concept (PoC)** phase, aiming to establish an end-to-end ML infrastructure and validate predictive value against baselines before further investment.

---

## 2. Success Metrics

### Goal Metric (Business KPI)
Increase decision quality by outperforming a standard heuristic baseline, thus generating potential expected value or strategic advantage. Also measured by pipeline reliability (stable and successful daily runs).

### Technical Metrics
- **Primary**: Log Loss / Cross-Entropy (measures pure probability quality) and Brier Score (measures calibration).
- **Secondary**: ROC AUC, Accuracy vs baseline.

### Baseline
The baseline comparison is a simple model that always predicts the globally higher-ranked ATP player to win the match.

---

## 3. The Data

### Source
Data is sourced from the **JeffSackmann ATP dataset** (CSV files per season). 
- To prevent data leakage, we **restrict features to pre-match data only** (no post-match statistics).

### Privacy & Handling
- There is no obvious Personally Identifiable Information (PII) involved.
- ⚠️ **Dataset files are large: do not commit raw data.** Keep the `data/` directory completely ignored in `.gitignore`.

---

## 4. Requirements & Risks

### Scalability
The pipeline is currently designed to run efficiently on a single machine or CI/CD runner using Pandas. Architecture should allow modular swapping of components (e.g., using Cloud Storage or moving to a specialized orchestration tool if the data grows significantly).

### Risks
- **Data Leakage**: It is highly critical to enforce a strict temporal 3-way split (Train/Val/Test) and fit all feature transformations and scaling on the Train split only.
- **Concept Drift**: Tennis playing conditions and player strengths evolve over time. Predictions may become stale if the model is not regularly retrained with new seasons.

### Cost Estimate
Running the batch pipeline requires minimal compute (e.g., standard GitHub Actions runner or local laptop). Operating a daily cloud job for inference and ad-hoc retraining would cost <$10/month using standard cloud instances.

---

## 5. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── README.md
├── environment.yml
├── config.yaml
├── .env
│
├── notebooks/
│   └── baseline_eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── load_data.py         # Download/read ATP seasons -> raw df
│   ├── clean_data.py        # Clean + standardize + select columns
│   ├── validate.py          # Quality gate (schema + domain checks)
│   ├── features.py          # Build feature recipe (no leakage, fit on Train)
│   ├── train.py             # Train model pipeline + save artifact
│   ├── evaluate.py          # Compute metrics + save report
│   ├── infer.py             # Predict on “new” matches + save output
│   ├── utils.py             # IO helpers, logging helpers, etc.
│   └── main.py              # Orchestrator
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── reports/
└── tests/
```

---

## 6. Artifacts Produced

The pipeline enforces standard artifacts generated on every successful run:
- `data/processed/clean.csv`
- `models/model.joblib`
- `reports/predictions.csv`
- Evaluation reports (e.g., `reports/metrics.json` or `.txt`)

---

## 7. How to Run & Test

Execute the entire ML pipeline from end to end (orchestrated by `config.yaml`):
```bash
python -m src.main
```

Run unit tests to ensure all modules are working perfectly:
```bash
python -m pytest -q
```

---

## 8. Team Split Suggestion
To distribute the workload reasonably across teams, we suggest assigning each member ~2 code modules and their corresponding tests:
- **Member 1**: `load_data.py`, `clean_data.py` + tests
- **Member 2**: `validate.py`, `features.py` + tests
- **Member 3**: `train.py`, `evaluate.py` + tests
- **Member 4**: `infer.py`, `utils.py`, `main.py` + tests