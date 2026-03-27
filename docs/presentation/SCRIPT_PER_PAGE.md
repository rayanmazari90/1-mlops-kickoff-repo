# Read-aloud script — one block per presentation page

**How to use this file:** Each section matches exactly one HTML file in this folder. Read the **Script** paragraphs in order while that page is on screen. Approximate times assume a calm presentation pace (~130 words per minute).

**Note on course rules:** If your assignment still says you must *not* read a script on camera, treat this as your rehearsal text: read it several times, then record while **paraphrasing** so delivery stays natural. If reading aloud is allowed, you can use this verbatim.

---

## Page: `index.html` — Start (~1 minute 30)

**Script:**

Welcome. We are Group Eight, and this is our production MLOps story for ATP tennis match prediction.

Most teams stop at a notebook: they train a model once, save a pickle file, and hope nobody asks for the same result twice. We did not stop there. We built a calibrated prediction service that behaves like a product: versioned code, a promoted model in a registry, a real REST API, a Streamlit interface for non-technical users, and automated tests on every change.

Our headline message is simple: we care about **probability quality**, not only accuracy on a slide. Pre-match analytics for sports and trading-style workflows need **calibrated** win probabilities — numbers you can compare to each other and to reality. That is why our evaluation leads with log loss and Brier score, not just who got more picks right.

On this page we make three promises you can hold us to. First, every prediction goes through the same **Pydantic contract** — valid JSON in, probability out — no “rerun cell seventeen.” Second, production uses one **explicitly promoted** model from Weights and Biases with the `prod` alias. Third, **guardrails**: more than sixty tests and a ninety-percent coverage gate on our core source code, plus Docker so what we run locally matches what runs on Render.

Use the chapter cards to follow the rest of the story: KPIs, narrative, platform, proof, and closing ethics and roadmap.

---

## Page: `kpi.html` — KPIs (~2 minutes 30)

**Script:**

This chapter is the scoreboard — how we talk to business stakeholders, not only to data scientists.

If you cannot measure it, you cannot manage it. In pre-match settings, **accuracy alone is a vanity metric**. Desks and analysts care whether probabilities are **well calibrated**: do predicted sixty-percent chances actually happen six times out of ten over many matches? That is why **log loss** and **Brier score** are our north-star metrics, alongside ROC AUC.

We structured five **business KPIs** on this page. First, **calibration quality**: we need probabilities that match reality, not just directionally correct labels. We deliver that by optimising and reporting log loss and Brier against a strong baseline. Second, **decision latency**: the answer must arrive in seconds, not after someone finds the right notebook. Our REST API and Streamlit app make that possible. Third, **model governance**: there must be exactly one production model, with traceable lineage. Weights and Biases holds our artifact, and the API loads the `prod` alias on startup. Fourth, **change confidence**: when we change code, we must not silently break predictions. Continuous integration runs Black, Flake8, and pytest with a coverage floor. Fifth, **operational audit trail**: when something fails, we know where to look — structured logs, W and B runs, and Render service logs.

Now the numbers on the test set, compared to the simplest serious baseline: always pick the higher-ranked player. Our model reaches about **sixty-five point three percent** accuracy versus **sixty-four point nine** for the baseline — a small edge on accuracy. The real story is calibration: **log loss** drops to about **zero point six three seven** compared to **twelve point six six five** for the baseline. **Brier score** improves to about **zero point two two three** versus **zero point three five one**. **ROC AUC** is about **zero point seven zero six** versus **zero point six four nine**.

In one sentence for the video: we do not only beat the rank heuristic on accuracy by a hair — we crush it on **probability quality**, which is what you need when outcomes are uncertain and stakes are asymmetric.

---

## Page: `story.html` — Story (~2 minutes 15)

**Script:**

This chapter is the business story: from the lab notebook to the operations desk.

Our persona is clear: a sports analyst, a trading-adjacent desk, or a product owner who needs **repeatable** pre-match win probabilities — not a chart from one analyst’s laptop on Tuesday afternoon. The conflict is simple: notebooks **explore**; they do not **govern**.

On the left, the notebook-heavy world: fragile cell order, “works on my machine,” unclear which pickle file is production, no strict JSON contract. Stakeholders cannot self-serve; every question waits on a data scientist. The business cost is latency, key-person risk, and decisions that are hard to audit.

On the right, our MLOps product: one command runs the full pipeline; Pydantic validates every API request; the model is promoted through the registry; Streamlit and Swagger let anyone reproduce the same prediction. The business value is **trust**, **speed**, and a **paper trail** when someone asks why the model said what it said.

Our narrative arc has four beats. Act One: gut feel and ad-hoc notebooks do not scale; calibration matters more than a flashy accuracy slide. Act Two: we engineered the same ATP prediction into modules, tests, and a deployed API — so “model” means one governed artifact. Act Three: live OpenAPI docs, Streamlit, and W and B lineage — evidence, not slides. Act Four: we state limitations honestly and position the system as **decision support**, especially where betting or risk is involved.

Exploration still lives in notebooks — our EDA notebook imports from production modules. Production lives behind an API with tests. We kept the science and added the receipts.

---

## Page: `platform.html` — Platform (~2 minutes)

**Script:**

This chapter is divide and conquer: how the code is organised so one team can evolve without chaos.

Everything starts from **configuration** and **secrets**: `config.yaml` for behaviour, environment variables for API keys. The orchestrator is `main.py`: it runs load, clean, validate, features, train, evaluate, and inference in order, and it initialises Weights and Biases in one place.

Downstream, the registry stores the trained model as an artifact. The **production** alias points to what we trust. The FastAPI application in `api.py` loads that artifact when the container starts — it does **not** reimplement machine learning; it only enforces the input and output contract. The same predictions power Streamlit in `app.py`, which calls the deployed service so there is no second hidden model path.

Walk the module list with me. `load_data` ingests ATP CSVs. `clean_data` standardises names and dates. `validate` uses Pandera to fail fast on bad data. `features` builds leakage-aware features and the sklearn recipe. `train` and `evaluate` fit the model and compare to the rank baseline. `infer` does batch scoring. `logger` gives us dual logging with no print statements in production code.

The API contract in plain language: **POST** to `/predict` with match features such as surface, tournament level, round, ranks, and optional hand and difference fields — you receive a class label and a **probability**. **GET** `/health` tells you whether the model loaded successfully. OpenAPI generates Swagger and ReDoc automatically — that is our technical sales brochure for anyone who wants to integrate.

---

## Page: `proof.html` — Proof (~2 minutes 30, plus live demo time)

**Script:**

Slides claim; URLs prove. This chapter is evidence that the system runs in the wild.

First, open the **Streamlit** application. That is our human-facing layer: real player presets, surfaces, tournament context, and calibrated probabilities — so a stakeholder can stress-test scenarios without writing code.

Second, open **Swagger** at the `/docs` path. This is the same contract a trading tool or internal dashboard would use: **Try it out**, send a JSON body with surface, tournament level, round, and ranks, and read back the probability. That is live inference, not a screenshot.

Third, **ReDoc** gives a clean read-only reference for assessors who prefer documentation over interactive calls.

Fourth, **GET `/health`** confirms whether the model loaded — important when the container wakes up on a free-tier host.

Fifth, the **Weights and Biases** project shows every run: hyperparameters, metrics, and artifacts — including which build is tied to `prod`.

Sixth, **GitHub Actions** shows our continuous integration: formatting, linting, and the full test suite on every push and pull request.

**[PAUSE — LIVE DEMO: execute one POST in Swagger, then show one prediction in Streamlit. If the service is cold-starting, say so briefly and use a short recording.]**

On trust, quote these numbers: **sixty-two** automated tests, coverage enforced at **at least ninety percent** on core `src` modules, **Docker** parity with Render, deployment triggered by a **GitHub Release**, and **three** observability channels: local log file, W and B, and Render logs.

---

## Page: `close.html` — Close (~1 minute 45)

**Script:**

We land the plane with honesty and a forward path.

Our model card is explicit about what we do **not** claim: we do not use in-match dynamics, deep head-to-head history, or injury models. Performance can drift as the tour evolves. Training data is hard-court heavy, so edge cases on clay or grass deserve caution. We position this system as **decision support**: combine outputs with domain knowledge and internal risk policy — especially in betting-adjacent use cases.

The roadmap for **continuous data** is practical, not vaporware. We can schedule the pipeline when new seasons appear. Every promoted model can stay tied to a W and B run and a Git commit. We can monitor input distributions for drift before accuracy collapses. Any system that can POST JSON can consume `/predict` at scale.

Thank you for your attention. Group Eight — Rayane Boumediene Mazari, Sacha Huberty, Shreya Jha, Smaragda Apostolou, Marco De Palma, and Pipe — IE University, Master in Business Analytics and Data Science, MLOps twenty twenty-six. Our repository and full documentation are on GitHub; this presentation walks chapter by chapter from KPIs to proof. We are happy to take questions.

---

## Total timing

| Page            | Approx. read time |
|-----------------|-------------------|
| `index.html`    | ~1:30             |
| `kpi.html`      | ~2:30             |
| `story.html`    | ~2:15             |
| `platform.html` | ~2:00             |
| `proof.html`    | ~2:30 + demo      |
| `close.html`    | ~1:45             |
| **Subtotal**    | **~12:30** read + demo |

If you must fit **ten minutes**, shorten `kpi.html` and `story.html` by skipping repeated numbers, or run `proof.html` demo-only with a shorter spoken intro.
