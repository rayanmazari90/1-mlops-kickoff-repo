# Video script — read aloud, page by page

**Matches these files in order:** `index.html` → `kpi.html` → `story.html` → `platform.html` → `proof.html` → `close.html`

**How to read:** Say each line in order. One line = one breath or one short phrase. Pause at blank lines between blocks if you want a natural rhythm.

**Total:** about twelve to thirteen minutes plus your live demo on `proof.html`. To fit ten minutes, skip the lines marked *(optional — shorten)* in KPI and Story.

---

# PAGE 1 — `index.html` (Start)

**Approx. 1 minute 30 seconds**

```
NARRATOR

Welcome.

We are Group Eight.

This is our production MLOps project for ATP tennis match prediction.

Most teams stop at a notebook.
They train once.
They save a file.
They hope nobody asks for the same answer twice.

We did not stop there.

We built a service that behaves like a product.

Versioned code.

A promoted model in a registry.

A real REST API.

A Streamlit app for people who do not write Python.

And automated tests on every change.

Our headline is simple.

We care about probability quality.
Not only accuracy on a slide.

For sports and trading-style workflows,
you need calibrated probabilities.

Numbers you can compare to each other and to reality.

That is why we lead with log loss and Brier score.

We make three promises.

First.
Every prediction uses the same contract.
Valid JSON in.
Probability out.
No “rerun cell seventeen.”

Second.
Production uses one model.
Promoted in Weights and Biases under the prod alias.

Third.
Guardrails.
More than sixty tests.
At least ninety percent coverage on our core code.
Docker so local matches Render.

On the next pages we go deeper.
KPIs.
Story.
Platform.
Proof.
And we close with ethics and roadmap.

Thank you.
Turn the page to KPIs.
```

---

# PAGE 2 — `kpi.html` (KPIs)

**Approx. 2 minutes 30 seconds**

```
NARRATOR

This page is our scoreboard.

How we talk to business people.
Not only to data scientists.

If you cannot measure it, you cannot manage it.

Accuracy alone is a vanity metric here.

What matters is calibration.

Do sixty-percent predictions happen six times out of ten, over many matches?

That is why log loss and Brier score are our north stars.
Along with ROC AUC.

We track five business KPIs.

One.
Calibration quality.
Probabilities must match reality.
We optimise log loss and Brier against a strong baseline.

Two.
Decision latency.
The answer in seconds.
Not after someone finds a notebook.
The API and Streamlit do that.

Three.
Model governance.
Exactly one production model.
Traceable lineage.
Weights and Biases holds the artifact.
The API loads prod on startup.

Four.
Change confidence.
New code must not silently break predictions.
CI runs Black, Flake8, and pytest with a coverage floor.

Five.
Operational audit trail.
When something fails, we know where to look.
Logs.
W and B.
Render.

Now the numbers.
Test set.
Baseline is simple.
Always pick the higher-ranked player.

Our accuracy is about sixty-five point three percent.

The baseline is about sixty-four point nine.

A small edge.

The real story is calibration.

Log loss for our model is about zero point six three seven.

For the baseline it is about twelve point six six five.

Brier score for us is about zero point two two three.

For the baseline about zero point three five one.

ROC AUC for us is about zero point seven zero six.

For the baseline about zero point six four nine.

So we do not only beat the rank rule on accuracy by a hair.

We crush it on probability quality.

That is what you need when stakes are asymmetric.

Turn the page to Story.
```

*(optional — shorten)* Omit the five KPI bullets and keep only: calibration, latency, governance, CI, audit — one sentence each.

---

# PAGE 3 — `story.html` (Story)

**Approx. 2 minutes 15 seconds**

```
NARRATOR

This page is the business story.

From the lab notebook to the operations desk.

Who is this for?

A sports analyst.

A trading-adjacent desk.

A product owner who needs the same answer every day.

Not a chart from one laptop on Tuesday afternoon.

Here is the conflict.

Notebooks explore.

They do not govern.

On the notebook side.

Cell order is fragile.

It works on my machine.

Which pickle file is production?
Nobody knows.

No strict JSON contract.

Stakeholders cannot self-serve.

Every question waits on a data scientist.

That costs time.
That costs trust.
That is hard to audit.

On our side.

One command runs the full pipeline.

Pydantic validates every API call.

The model is promoted in the registry.

Streamlit and Swagger let anyone get the same prediction.

The value is trust.
Speed.
And a paper trail.

Our story has four acts.

Act one.
Gut feel does not scale.
Calibration beats a flashy accuracy slide.

Act two.
We put the same ATP model into modules, tests, and an API.

One governed artifact.

Act three.
Live docs.
Streamlit.
W and B.
Evidence.
Not slides.

Act four.
We admit limits.
We call this decision support.
Especially where betting or risk matters.

Exploration still lives in notebooks.
Our EDA notebook imports from production code.

Production lives behind an API with tests.

We kept the science.

We added the receipts.

Turn the page to Platform.
```

---

# PAGE 4 — `platform.html` (Platform)

**Approx. 2 minutes**

```
NARRATOR

This page is divide and conquer.

How we organised the code.

Everything starts with config and secrets.

config.yaml drives behaviour.

Environment variables hold API keys.

main.py is the orchestrator.

It runs load, clean, validate, features, train, evaluate, and inference.

Weights and Biases initialises in one place.

The registry stores the trained model.

The prod alias is what we trust.

api.py loads that model when the container starts.

It does not reimplement machine learning.

It only enforces inputs and outputs.

app.py is Streamlit.

It calls the deployed service.

There is no second hidden model.

Module by module.

load_data pulls ATP CSVs.

clean_data standardises names and dates.

validate uses Pandera to fail fast on bad data.

features builds leakage-aware inputs and the sklearn recipe.

train and evaluate fit the model and compare to the rank baseline.

infer does batch scoring.

logger writes to console and file.

No print statements in production.

The API in plain words.

POST to slash predict.

You send surface, tournament level, round, ranks, and optional fields.

You get a label and a probability.

GET slash health tells you if the model loaded.

Swagger and ReDoc come free from OpenAPI.

That is our brochure for integrators.

Turn the page to Proof.
```

---

# PAGE 5 — `proof.html` (Proof)

**Approx. 2 minutes 30 seconds speaking, plus your demo**

```
NARRATOR

This page is proof.

Slides claim.

URLs prove.

First.
Open Streamlit.

That is the human layer.

Real players.
Surfaces.
Tournament context.
Calibrated probabilities.

No code required.

Second.
Open Swagger at slash docs.

That is the same contract a real system would call.

Try it out.

Send JSON.

Read back the probability.

That is live inference.

Third.
ReDoc at slash redoc.

Clean reference for anyone grading the work.

Fourth.
GET slash health.

See if the model loaded.

Useful when the free tier wakes up cold.

Fifth.
Open our Weights and Biases project.

Runs.
Metrics.
Artifacts.

Including what is tied to prod.

Sixth.
Open GitHub Actions.

CI on every push and pull request.

```

**[STOP READING. DO YOUR LIVE DEMO NOW.]**

*Say this while you click:*

```
NARRATOR

I will run one prediction in Swagger.

Then I will show the same idea in Streamlit.

```

*(If the API is slow, say: “Free tier cold start — here is a short clip instead.”)*

**[RESUME READING AFTER DEMO.]**

```
NARRATOR

For trust, remember these numbers.

Sixty-two automated tests.

Coverage at least ninety percent on core src.

Docker matches Render.

Deploy happens when we publish a GitHub Release.

Three places to debug.

Local log file.

W and B.

Render logs.

Turn the page to Close.
```

---

# PAGE 6 — `close.html` (Close)

**Approx. 1 minute 45 seconds**

```
NARRATOR

We close with honesty.

Our model card is clear about limits.

We do not use in-match play-by-play.

We do not model deep head-to-head history.

We do not model injuries.

The tour changes.
Performance can drift.

We trained heavily on hard courts.
Clay and grass need caution.

We position this as decision support.

Combine the model with domain knowledge and risk policy.

Especially in betting-related contexts.

Looking forward.

We can schedule the pipeline when new data arrives.

Every promoted model can link to a W and B run and a Git commit.

We can watch inputs for drift before accuracy collapses.

Anything that can POST JSON can call slash predict at scale.

Thank you for your attention.

We are Group Eight.

Rayane Boumediene Mazari.

Sacha Huberty.

Shreya Jha.

Smaragda Apostolou.

Marco De Palma.

And Pipe.

IE University.

Master in Business Analytics and Data Science.

MLOps twenty twenty-six.

Everything is on GitHub with a full README.

This deck walks page by page with the script you are reading.

We are happy to take questions.
```

---

## Timing cheat sheet

| Page file       | Block above      | ~Time      |
|----------------|------------------|------------|
| `index.html`   | PAGE 1           | ~1:30      |
| `kpi.html`     | PAGE 2           | ~2:30      |
| `story.html`   | PAGE 3           | ~2:15      |
| `platform.html`| PAGE 4           | ~2:00      |
| `proof.html`   | PAGE 5 + demo    | ~2:30 + demo |
| `close.html`   | PAGE 6           | ~1:45      |

**To reach ~10 minutes:** use the *(optional — shorten)* note on PAGE 2, speak PAGE 5 demo section in under one minute, and trim a few repeated lines on PAGE 3.
