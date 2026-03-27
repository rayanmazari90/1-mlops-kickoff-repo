"""
Streamlit UI for Tennis ATP Match Prediction.
Sends requests to the FastAPI /predict endpoint and displays results.

Run locally:  streamlit run src/app.py
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ATP Match Predictor", page_icon="🎾", layout="centered")

st.title("Tennis ATP Match Predictor")
st.markdown(
    "Enter the pre-match details below and click **Predict** to see "
    "the estimated probability of Player 1 winning."
)

with st.form("match_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Player 1")
        p1_rank = st.number_input("ATP Rank", min_value=1, value=5, key="p1_rank")
        p1_hand = st.selectbox("Hand", ["R", "L", "U"], key="p1_hand")
        p1_age = st.number_input(
            "Age", min_value=14.0, max_value=50.0, value=25.0, step=0.1, key="p1_age"
        )
        p1_ht = st.number_input(
            "Height (cm)", min_value=150, max_value=220, value=185, key="p1_ht"
        )

    with col2:
        st.subheader("Player 2")
        p2_rank = st.number_input("ATP Rank", min_value=1, value=20, key="p2_rank")
        p2_hand = st.selectbox("Hand", ["R", "L", "U"], key="p2_hand")
        p2_age = st.number_input(
            "Age", min_value=14.0, max_value=50.0, value=28.0, step=0.1, key="p2_age"
        )
        p2_ht = st.number_input(
            "Height (cm)", min_value=150, max_value=220, value=180, key="p2_ht"
        )

    st.markdown("---")
    surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
    tourney_level = st.selectbox("Tournament Level", ["G", "M", "A", "D", "F"])
    match_round = st.selectbox(
        "Round", ["F", "SF", "QF", "R16", "R32", "R64", "R128", "RR"]
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "surface": surface,
        "tourney_level": tourney_level,
        "round": match_round,
        "p1_rank": float(p1_rank),
        "p2_rank": float(p2_rank),
        "p1_hand": p1_hand,
        "p2_hand": p2_hand,
        "rank_diff": float(p1_rank - p2_rank),
        "age_diff": float(p1_age - p2_age),
        "ht_diff": float(p1_ht - p2_ht),
    }

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()

        prob = result["probability"]
        pred = result["prediction"]

        st.markdown("---")
        st.subheader("Prediction Result")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Winner", "Player 1" if pred == 1 else "Player 2")
        with col_b:
            st.metric("P1 Win Probability", f"{prob:.1%}")

        st.progress(prob)

        if prob >= 0.6:
            st.success("High confidence in Player 1 winning.")
        elif prob >= 0.4:
            st.warning("Close match — either player could win.")
        else:
            st.info("Player 2 is favored to win.")

    except requests.ConnectionError:
        st.error(
            f"Cannot reach the API at **{API_URL}**. "
            "Make sure the server is running (`uvicorn src.api:app`)."
        )
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")
