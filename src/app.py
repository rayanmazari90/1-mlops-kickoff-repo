"""
Streamlit UI for Tennis ATP Match Prediction.
Sends requests to the FastAPI /predict endpoint and displays results.

Run locally:  streamlit run src/app.py
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get(
    "API_URL", "https://one-mlops-kickoff-repo-1.onrender.com"
)

TOP_PLAYERS = {
    "Jannik Sinner": {"rank": 1, "hand": "R", "age": 23.5, "ht": 191},
    "Alexander Zverev": {"rank": 2, "hand": "R", "age": 27.9, "ht": 198},
    "Carlos Alcaraz": {"rank": 3, "hand": "R", "age": 21.9, "ht": 183},
    "Novak Djokovic": {"rank": 7, "hand": "R", "age": 37.8, "ht": 188},
    "Daniil Medvedev": {"rank": 5, "hand": "R", "age": 29.2, "ht": 198},
    "Taylor Fritz": {"rank": 4, "hand": "R", "age": 27.3, "ht": 193},
    "Casper Ruud": {"rank": 6, "hand": "R", "age": 26.2, "ht": 182},
    "Alex de Minaur": {"rank": 8, "hand": "R", "age": 26.1, "ht": 183},
    "Andrey Rublev": {"rank": 9, "hand": "R", "age": 27.3, "ht": 188},
    "Grigor Dimitrov": {"rank": 10, "hand": "R", "age": 33.8, "ht": 191},
    "Tommy Paul": {"rank": 11, "hand": "R", "age": 27.8, "ht": 185},
    "Stefanos Tsitsipas": {"rank": 12, "hand": "R", "age": 26.6, "ht": 193},
    "Holger Rune": {"rank": 13, "hand": "R", "age": 21.9, "ht": 188},
    "Jack Draper": {"rank": 14, "hand": "L", "age": 23.2, "ht": 193},
    "Hubert Hurkacz": {"rank": 15, "hand": "R", "age": 28.1, "ht": 196},
    "Lorenzo Musetti": {"rank": 16, "hand": "R", "age": 23.2, "ht": 185},
    "Frances Tiafoe": {"rank": 17, "hand": "R", "age": 27.1, "ht": 188},
    "Sebastian Korda": {"rank": 18, "hand": "R", "age": 24.6, "ht": 196},
    "Ugo Humbert": {"rank": 19, "hand": "L", "age": 26.7, "ht": 188},
    "Ben Shelton": {"rank": 20, "hand": "L", "age": 22.3, "ht": 193},
    "Custom Player": None,
}

SURFACES = {"Hard": "Hard", "Clay": "Clay", "Grass": "Grass"}

TOURNEY_LEVELS = {
    "Grand Slam": "G",
    "Masters 1000": "M",
    "ATP 500 / 250": "A",
    "Davis Cup": "D",
    "Tour Finals": "F",
}

ROUNDS = {
    "Final": "F",
    "Semi-Final": "SF",
    "Quarter-Final": "QF",
    "Round of 16": "R16",
    "Round of 32": "R32",
    "Round of 64": "R64",
    "Round of 128": "R128",
    "Round Robin": "RR",
}

st.set_page_config(page_title="ATP Match Predictor", page_icon="🎾", layout="wide")

st.markdown(
    """
    <style>
    .main-header {font-size: 2.5rem; font-weight: 700; text-align: center;
                   margin-bottom: 0.2rem;}
    .sub-header  {font-size: 1.1rem; text-align: center; color: #888;
                   margin-bottom: 2rem;}
    .vs-text     {font-size: 2rem; font-weight: 700; text-align: center;
                   color: #e74c3c; padding-top: 2.5rem;}
    .winner-box  {padding: 1.5rem; border-radius: 12px; text-align: center;
                   font-size: 1.3rem; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">ATP Match Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "ML-powered pre-match win probability predictions for ATP tennis"
    "</div>",
    unsafe_allow_html=True,
)

player_names = list(TOP_PLAYERS.keys())

col_left, col_vs, col_right = st.columns([5, 1, 5])

with col_left:
    st.subheader("Player A")
    p1_name = st.selectbox("Select Player", player_names, index=0, key="p1_sel")
    if TOP_PLAYERS[p1_name] is not None:
        p1_defaults = TOP_PLAYERS[p1_name]
        p1_rank = st.number_input(
            "ATP Ranking", min_value=1, max_value=2000,
            value=p1_defaults["rank"], key="p1r",
        )
        p1_hand = st.selectbox(
            "Dominant Hand", ["R", "L", "U"],
            index=["R", "L", "U"].index(p1_defaults["hand"]), key="p1h",
        )
        p1_age = st.number_input(
            "Age", min_value=14.0, max_value=50.0,
            value=float(p1_defaults["age"]), step=0.1, key="p1a",
        )
        p1_ht = st.number_input(
            "Height (cm)", min_value=150, max_value=220,
            value=p1_defaults["ht"], key="p1ht",
        )
    else:
        p1_rank = st.number_input(
            "ATP Ranking", min_value=1, max_value=2000, value=50, key="p1r"
        )
        p1_hand = st.selectbox("Dominant Hand", ["R", "L", "U"], key="p1h")
        p1_age = st.number_input(
            "Age", min_value=14.0, max_value=50.0, value=25.0, step=0.1, key="p1a"
        )
        p1_ht = st.number_input(
            "Height (cm)", min_value=150, max_value=220, value=185, key="p1ht"
        )

with col_vs:
    st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)

with col_right:
    st.subheader("Player B")
    p2_name = st.selectbox("Select Player", player_names, index=3, key="p2_sel")
    if TOP_PLAYERS[p2_name] is not None:
        p2_defaults = TOP_PLAYERS[p2_name]
        p2_rank = st.number_input(
            "ATP Ranking", min_value=1, max_value=2000,
            value=p2_defaults["rank"], key="p2r",
        )
        p2_hand = st.selectbox(
            "Dominant Hand", ["R", "L", "U"],
            index=["R", "L", "U"].index(p2_defaults["hand"]), key="p2h",
        )
        p2_age = st.number_input(
            "Age", min_value=14.0, max_value=50.0,
            value=float(p2_defaults["age"]), step=0.1, key="p2a",
        )
        p2_ht = st.number_input(
            "Height (cm)", min_value=150, max_value=220,
            value=p2_defaults["ht"], key="p2ht",
        )
    else:
        p2_rank = st.number_input(
            "ATP Ranking", min_value=1, max_value=2000, value=50, key="p2r"
        )
        p2_hand = st.selectbox("Dominant Hand", ["R", "L", "U"], key="p2h")
        p2_age = st.number_input(
            "Age", min_value=14.0, max_value=50.0, value=25.0, step=0.1, key="p2a"
        )
        p2_ht = st.number_input(
            "Height (cm)", min_value=150, max_value=220, value=185, key="p2ht"
        )

st.markdown("---")

mcol1, mcol2, mcol3 = st.columns(3)
with mcol1:
    surface_label = st.selectbox("Surface", list(SURFACES.keys()))
with mcol2:
    tourney_label = st.selectbox("Tournament Level", list(TOURNEY_LEVELS.keys()))
with mcol3:
    round_label = st.selectbox("Round", list(ROUNDS.keys()))

st.markdown("")
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    predict_btn = st.button("Predict Match Outcome", use_container_width=True, type="primary")

if predict_btn:
    p1_display = p1_name if TOP_PLAYERS[p1_name] else f"Custom (Rank {p1_rank})"
    p2_display = p2_name if TOP_PLAYERS[p2_name] else f"Custom (Rank {p2_rank})"

    payload = {
        "surface": SURFACES[surface_label],
        "tourney_level": TOURNEY_LEVELS[tourney_label],
        "round": ROUNDS[round_label],
        "p1_rank": float(p1_rank),
        "p2_rank": float(p2_rank),
        "p1_hand": p1_hand,
        "p2_hand": p2_hand,
        "rank_diff": float(p1_rank - p2_rank),
        "age_diff": float(p1_age - p2_age),
        "ht_diff": float(p1_ht - p2_ht),
    }

    try:
        with st.spinner("Querying prediction model ..."):
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()

        prob_a = result["probability"]
        prob_b = 1.0 - prob_a
        pred = result["prediction"]
        winner = p1_display if pred == 1 else p2_display
        loser = p2_display if pred == 1 else p1_display
        win_prob = prob_a if pred == 1 else prob_b

        st.markdown("---")
        st.markdown("### Match Prediction")

        res_left, res_mid, res_right = st.columns([2, 1, 2])

        with res_left:
            if pred == 1:
                st.success(f"**{p1_display}**  \nWin probability: **{prob_a:.1%}**")
            else:
                st.error(f"**{p1_display}**  \nWin probability: **{prob_a:.1%}**")

        with res_mid:
            st.markdown(
                f'<div style="text-align:center; padding-top:1rem; '
                f'font-size:1.5rem; font-weight:700;">'
                f'{prob_a:.0%} – {prob_b:.0%}</div>',
                unsafe_allow_html=True,
            )

        with res_right:
            if pred == 0:
                st.success(f"**{p2_display}**  \nWin probability: **{prob_b:.1%}**")
            else:
                st.error(f"**{p2_display}**  \nWin probability: **{prob_b:.1%}**")

        st.progress(prob_a)

        st.markdown("### Match Context")
        ctx1, ctx2, ctx3, ctx4 = st.columns(4)
        ctx1.metric("Surface", surface_label)
        ctx2.metric("Tournament", tourney_label)
        ctx3.metric("Round", round_label)
        ctx4.metric("Predicted Winner", winner)

        st.markdown("### Insight")
        rank_gap = abs(p1_rank - p2_rank)
        if win_prob >= 0.7:
            st.info(
                f"**Strong favourite:** {winner} is heavily favoured with "
                f"**{win_prob:.1%}** win probability."
            )
        elif win_prob >= 0.55:
            st.warning(
                f"**Slight edge:** {winner} has a moderate advantage "
                f"(**{win_prob:.1%}**). This could go either way."
            )
        else:
            st.success(
                f"**Coin-flip match:** Our model sees this as extremely close "
                f"(**{win_prob:.1%}** for {winner}). Expect a battle."
            )

        if rank_gap > 50:
            st.caption(
                f"Ranking gap of {rank_gap} positions — significant disparity."
            )

    except requests.ConnectionError:
        st.error(
            f"Cannot reach the API at **{API_URL}**. "
            "The Render free-tier service may be sleeping — wait 30 seconds and retry."
        )
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")

st.markdown("---")
with st.expander("About this app"):
    st.markdown(
        """
        **Tennis ATP Match Predictor** is a production MLOps project built by Group 8
        (IE University, MsC Business Analytics & Data Science).

        - **Model:** RandomForestClassifier trained on ATP match data (2018–2020)
        - **Features:** Player rankings, hand dominance, age/height differentials,
          surface, tournament level, and round
        - **Tracking:** Weights & Biases (model registry with `prod` alias)
        - **API:** FastAPI with Pydantic validation, deployed on Render via Docker
        - **Accuracy:** 65.3% (vs 64.9% rank-only baseline)
        - **Log Loss:** 0.637 (vs 12.67 baseline) — dramatically better calibration

        [GitHub Repository](https://github.com/rayanmazari90/1-mlops-kickoff-repo) |
        [W&B Dashboard](https://wandb.ai/bmazari-ieu2024-ie-university/tennis-atp-prediction) |
        [API Docs](https://one-mlops-kickoff-repo-1.onrender.com/docs)
        """
    )
