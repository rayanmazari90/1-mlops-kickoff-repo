# 10-minute business video — talking points (Group 8)

**Full read-aloud script (per HTML page):** see **[`SCRIPT_PER_PAGE.md`](SCRIPT_PER_PAGE.md)** — continuous lines you can read while each page is displayed.

**Purpose (this file):** Compact **beats and soundbites** if you prefer bullet-style prep. If your course rule says **do not read a script verbatim on camera**, rehearse with [`SCRIPT_PER_PAGE.md`](SCRIPT_PER_PAGE.md) until internalised, then paraphrase on camera.

**Total time:** ~10:00  
**Recommended layout:** Split screen or picture-in-picture: your face + browser (site + Swagger + Streamlit).

---

## 0:00–1:00 — Hook & persona

**Goal:** Answer “who cares?” in one sentence.

**Talking points:**

- Pre-match ATP tennis: operators need **calibrated win probabilities**, not gut feel.
- **Persona:** analyst, trading-adjacent desk, or product owner who must **repeat** decisions daily — a Jupyter notebook doesn’t scale.
- **Thesis:** We didn’t just train a model; we **productised** it — API, UI, tests, registry, deployment.

**Soundbite (paraphrase, don’t read):**  
*“Accuracy is vanity for this use case; calibration is sanity. That’s why we lead with log loss and Brier, not a leaderboard slide.”*

**On screen:** `index.html` — hero + three promises.

---

## 1:00–3:00 — KPIs & business framework

**Goal:** Prove you speak “business KPI,” not only ML jargon.

**Talking points:**

- **North star:** Better **probability quality** than a rank-only baseline (the heuristic everyone already uses).
- Walk through the **business KPI table** on `kpi.html`: calibration, decision latency, governance, change confidence, audit trail.
- Drop the **numbers:** accuracy 65.3% vs 64.9%; log loss 0.637 vs 12.665; Brier 0.223 vs 0.351; AUC 0.706 vs 0.649.
- **Translate:** “Lower log loss means our probabilities are closer to reality — that’s what you need when uncertainty and stakes are asymmetric.”

**Avoid:** Claiming guaranteed profit or “beating the book” without evidence.

**On screen:** `kpi.html` — scroll the business framework table, then the animated KPI tiles.

---

## 3:00–5:30 — Story: notebook vs MLOps

**Goal:** Clear before/after drama.

**Talking points:**

- **Before:** Hidden cell order, unclear artifact, no JSON contract, key-person bottleneck.
- **After:** `python -m src.main`, W&B `prod`, Pydantic on `/predict`, Streamlit for humans.
- **Business translation:** Speed (seconds to answer), trust (same answer every time), audit (logs + W&B + CI).

**Soundbite:**  
*“Exploration lives in notebooks; production lives behind an API with tests. We kept the science and added the receipts.”*

**On screen:** `story.html` — split “pain” vs “gain,” then the comparison table.

---

## 5:30–7:30 — Platform & live proof

**Goal:** Show the architecture in 60s, then **live demo**.

**Talking points:**

- One diagram: config → pipeline → W&B → Docker/Render → API → Streamlit (`platform.html`).
- **Live (must rehearse):**  
  - Open Swagger → `POST /predict` with a simple JSON body → show probability.  
  - Open Streamlit → pick two known players → show the same story visually.
- Optional 20s: W&B project — one run, metrics, artifact lineage.

**On screen:** `platform.html` briefly, then real tabs for Swagger + Streamlit (+ W&B if time).

**Contingency:** If Render is cold-starting, say it honestly (“free tier sleeps”) and cut to recorded clip.

---

## 7:30–9:00 — Trust: CI/CD, Docker, observability

**Goal:** Show engineering maturity.

**Talking points:**

- **62 tests**, coverage gate **≥90%** — changes don’t silently break predictions.
- **GitHub Release → Render** — deployment is explicit, not “someone SSH’d.”
- **Three logs:** file (`logger.py`), W&B, Render — you can answer “what happened?” after the fact.

**Soundbite:**  
*“We’re not asking you to trust our slides; we’re asking you to trust a pipeline that fails in CI before it fails in production.”*

**On screen:** `proof.html` + GitHub Actions tab if useful.

---

## 9:00–10:00 — Ethics, roadmap, close

**Goal:** End mature, not hypey.

**Talking points:**

- **Limitations:** no in-match stats, H2H, injuries; possible surface/drift; hard-court bias in data.
- **Positioning:** **Decision support**, especially in betting-adjacent contexts — combine model with policy and domain knowledge.
- **Roadmap:** scheduled retrains, drift monitoring, more API consumers — **daily data** plugs into what we already built.

**Closing line (your words):**  
Thank the course + link repo + presentation site.

**On screen:** `close.html`.

---

## Delivery checklist (non-negotiables from the brief)

| Requirement | How you satisfy it |
|-------------|-------------------|
| Business value vs former notebook setup | Chapters **KPIs** + **Story** |
| No script reading on camera | Use this file as bullets only; rehearse 3 full runs |
| Real APIs | Swagger live demo + `/health` |
| Continuous / daily use narrative | Roadmap on **Close** + “plumbing is done” |
| Technical credibility | **Platform** + **Proof** |

---

## Taboos (don’t step in these)

- Reading paragraphs verbatim from slides or this file.
- Promising financial returns from the model.
- Hiding Render cold-start or test-set limitations.

---

## Rehearsal plan

1. Dry run with timer — aim for **9:30** to leave buffer.  
2. Record **B-roll** separately: Swagger execute, Streamlit walkthrough (backup if live fails).  
3. **One** teammate does voiceover OR split by chapter — consistent audio levels.

---

*Group 8 — IE University MLOps 2026*
