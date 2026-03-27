"""
Streamlit UI for Tennis ATP Match Prediction.
Connects to the live FastAPI backend on Render.
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "https://one-mlops-kickoff-repo-1.onrender.com")

TOP_PLAYERS = {
    "Jannik Sinner": {"rank": 1, "hand": "R", "age": 23.5, "ht": 191, "country": "ITA"},
    "Alexander Zverev": {
        "rank": 2,
        "hand": "R",
        "age": 27.9,
        "ht": 198,
        "country": "GER",
    },
    "Carlos Alcaraz": {
        "rank": 3,
        "hand": "R",
        "age": 21.9,
        "ht": 183,
        "country": "ESP",
    },
    "Taylor Fritz": {"rank": 4, "hand": "R", "age": 27.3, "ht": 193, "country": "USA"},
    "Daniil Medvedev": {
        "rank": 5,
        "hand": "R",
        "age": 29.2,
        "ht": 198,
        "country": "RUS",
    },
    "Casper Ruud": {"rank": 6, "hand": "R", "age": 26.2, "ht": 182, "country": "NOR"},
    "Novak Djokovic": {
        "rank": 7,
        "hand": "R",
        "age": 37.8,
        "ht": 188,
        "country": "SRB",
    },
    "Alex de Minaur": {
        "rank": 8,
        "hand": "R",
        "age": 26.1,
        "ht": 183,
        "country": "AUS",
    },
    "Andrey Rublev": {"rank": 9, "hand": "R", "age": 27.3, "ht": 188, "country": "RUS"},
    "Grigor Dimitrov": {
        "rank": 10,
        "hand": "R",
        "age": 33.8,
        "ht": 191,
        "country": "BUL",
    },
    "Tommy Paul": {"rank": 11, "hand": "R", "age": 27.8, "ht": 185, "country": "USA"},
    "Stefanos Tsitsipas": {
        "rank": 12,
        "hand": "R",
        "age": 26.6,
        "ht": 193,
        "country": "GRE",
    },
    "Holger Rune": {"rank": 13, "hand": "R", "age": 21.9, "ht": 188, "country": "DEN"},
    "Jack Draper": {"rank": 14, "hand": "L", "age": 23.2, "ht": 193, "country": "GBR"},
    "Hubert Hurkacz": {
        "rank": 15,
        "hand": "R",
        "age": 28.1,
        "ht": 196,
        "country": "POL",
    },
    "Lorenzo Musetti": {
        "rank": 16,
        "hand": "R",
        "age": 23.2,
        "ht": 185,
        "country": "ITA",
    },
    "Frances Tiafoe": {
        "rank": 17,
        "hand": "R",
        "age": 27.1,
        "ht": 188,
        "country": "USA",
    },
    "Sebastian Korda": {
        "rank": 18,
        "hand": "R",
        "age": 24.6,
        "ht": 196,
        "country": "USA",
    },
    "Ugo Humbert": {"rank": 19, "hand": "L", "age": 26.7, "ht": 188, "country": "FRA"},
    "Ben Shelton": {"rank": 20, "hand": "L", "age": 22.3, "ht": 193, "country": "USA"},
    "Rafael Nadal": {
        "rank": 250,
        "hand": "L",
        "age": 38.7,
        "ht": 185,
        "country": "ESP",
    },
    "Roger Federer": {
        "rank": 500,
        "hand": "R",
        "age": 43.5,
        "ht": 185,
        "country": "SUI",
    },
}

SURFACES = {"Hard Court": "Hard", "Clay Court": "Clay", "Grass Court": "Grass"}
SURFACE_EMOJI = {"Hard Court": "🔵", "Clay Court": "🟠", "Grass Court": "🟢"}

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

HAND_LABELS = {"R": "Right", "L": "Left", "U": "Unknown"}

FLAG = {
    "ITA": "🇮🇹",
    "GER": "🇩🇪",
    "ESP": "🇪🇸",
    "USA": "🇺🇸",
    "RUS": "🇷🇺",
    "NOR": "🇳🇴",
    "SRB": "🇷🇸",
    "AUS": "🇦🇺",
    "BUL": "🇧🇬",
    "GRE": "🇬🇷",
    "DEN": "🇩🇰",
    "GBR": "🇬🇧",
    "POL": "🇵🇱",
    "FRA": "🇫🇷",
    "SUI": "🇨🇭",
}


def get_flag(name):
    info = TOP_PLAYERS.get(name)
    if info and info.get("country"):
        return FLAG.get(info["country"], "🎾")
    return "🎾"


st.set_page_config(page_title="ATP Match Predictor", page_icon="🎾", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    padding: 2.5rem 2rem 2rem 2rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 70%);
}
.hero h1 {
    font-size: 2.8rem; font-weight: 900; color: #fff;
    letter-spacing: -1px; margin: 0;
}
.hero p {
    font-size: 1.1rem; color: rgba(255,255,255,0.7);
    margin-top: 0.5rem; font-weight: 400;
}
.hero .badge {
    display: inline-block; background: rgba(76, 175, 80, 0.2);
    color: #4CAF50; padding: 4px 14px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600; margin-top: 0.8rem;
    border: 1px solid rgba(76, 175, 80, 0.3);
}

.player-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.8rem;
    transition: all 0.3s ease;
}
.player-card:hover {
    border-color: rgba(255,255,255,0.15);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.player-card h3 {
    color: #fff; font-weight: 700; font-size: 1.3rem;
    margin-bottom: 1rem; letter-spacing: -0.5px;
}

.vs-badge {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
    color: white; font-weight: 900; font-size: 1.6rem;
    width: 70px; height: 70px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 2.5rem auto; box-shadow: 0 4px 20px rgba(231,76,60,0.4);
    letter-spacing: -1px;
}

.result-card {
    border-radius: 16px; padding: 2rem; text-align: center;
    border: 2px solid transparent;
}
.result-winner {
    background: linear-gradient(145deg, rgba(76,175,80,0.1), rgba(76,175,80,0.05));
    border-color: #4CAF50;
}
.result-loser {
    background: linear-gradient(145deg, rgba(239,83,80,0.1), rgba(239,83,80,0.05));
    border-color: rgba(239,83,80,0.3);
}
.result-card .name { font-size: 1.5rem; font-weight: 800; color: #fff; }
.result-card .prob { font-size: 2.5rem; font-weight: 900; margin: 0.5rem 0; }
.result-winner .prob { color: #4CAF50; }
.result-loser .prob { color: #EF5350; }
.result-card .label { font-size: 0.85rem; color: rgba(255,255,255,0.5);
                       text-transform: uppercase; letter-spacing: 2px; font-weight: 600; }

.stat-pill {
    background: rgba(255,255,255,0.05); border-radius: 12px;
    padding: 1rem 1.5rem; text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}
.stat-pill .value { font-size: 1.4rem; font-weight: 800; color: #fff; }
.stat-pill .label { font-size: 0.75rem; color: rgba(255,255,255,0.5);
                     text-transform: uppercase; letter-spacing: 1.5px;
                     margin-top: 0.3rem; font-weight: 600; }

.insight-box {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border-left: 4px solid #FFC107; border-radius: 12px;
    padding: 1.5rem; margin-top: 1.5rem;
}
.insight-box p { color: rgba(255,255,255,0.85); margin: 0; line-height: 1.6; }
.insight-box strong { color: #FFC107; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
    <h1>🎾 ATP Match Predictor</h1>
    <p>ML-powered pre-match win probability predictions for professional tennis</p>
    <div class="badge">✦ LIVE MODEL — Powered by RandomForest + W&B</div>
</div>
""",
    unsafe_allow_html=True,
)

player_names = list(TOP_PLAYERS.keys()) + ["Custom Player"]

col_a, col_vs, col_b = st.columns([5, 1, 5])

with col_a:
    st.markdown(
        '<div class="player-card"><h3>🟢 Player A</h3></div>', unsafe_allow_html=True
    )
    p1_name = st.selectbox(
        "Player", player_names, index=0, key="p1_sel", label_visibility="collapsed"
    )
    if p1_name != "Custom Player":
        d = TOP_PLAYERS[p1_name]
        flag = get_flag(p1_name)
        st.caption(f"{flag} {p1_name} — ATP #{d['rank']}")
        c1, c2 = st.columns(2)
        p1_rank = c1.number_input("Rank", 1, 2000, d["rank"], key="p1r")
        p1_hand = c2.selectbox(
            "Hand", ["R", "L", "U"], ["R", "L", "U"].index(d["hand"]), key="p1h"
        )
        c3, c4 = st.columns(2)
        p1_age = c3.number_input("Age", 14.0, 50.0, float(d["age"]), 0.1, key="p1a")
        p1_ht = c4.number_input("Height (cm)", 150, 220, d["ht"], key="p1ht")
    else:
        c1, c2 = st.columns(2)
        p1_rank = c1.number_input("Rank", 1, 2000, 50, key="p1r")
        p1_hand = c2.selectbox("Hand", ["R", "L", "U"], key="p1h")
        c3, c4 = st.columns(2)
        p1_age = c3.number_input("Age", 14.0, 50.0, 25.0, 0.1, key="p1a")
        p1_ht = c4.number_input("Height (cm)", 150, 220, 185, key="p1ht")

with col_vs:
    st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)

with col_b:
    st.markdown(
        '<div class="player-card"><h3>🔴 Player B</h3></div>', unsafe_allow_html=True
    )
    p2_name = st.selectbox(
        "Player", player_names, index=6, key="p2_sel", label_visibility="collapsed"
    )
    if p2_name != "Custom Player":
        d = TOP_PLAYERS[p2_name]
        flag = get_flag(p2_name)
        st.caption(f"{flag} {p2_name} — ATP #{d['rank']}")
        c1, c2 = st.columns(2)
        p2_rank = c1.number_input("Rank", 1, 2000, d["rank"], key="p2r")
        p2_hand = c2.selectbox(
            "Hand", ["R", "L", "U"], ["R", "L", "U"].index(d["hand"]), key="p2h"
        )
        c3, c4 = st.columns(2)
        p2_age = c3.number_input("Age", 14.0, 50.0, float(d["age"]), 0.1, key="p2a")
        p2_ht = c4.number_input("Height (cm)", 150, 220, d["ht"], key="p2ht")
    else:
        c1, c2 = st.columns(2)
        p2_rank = c1.number_input("Rank", 1, 2000, 50, key="p2r")
        p2_hand = c2.selectbox("Hand", ["R", "L", "U"], key="p2h")
        c3, c4 = st.columns(2)
        p2_age = c3.number_input("Age", 14.0, 50.0, 25.0, 0.1, key="p2a")
        p2_ht = c4.number_input("Height (cm)", 150, 220, 185, key="p2ht")

st.markdown("")
st.markdown("#### Match Setup")
mc1, mc2, mc3 = st.columns(3)
with mc1:
    surface_label = st.selectbox("Surface", list(SURFACES.keys()))
    st.caption(f"{SURFACE_EMOJI[surface_label]} {surface_label}")
with mc2:
    tourney_label = st.selectbox("Tournament", list(TOURNEY_LEVELS.keys()))
with mc3:
    round_label = st.selectbox("Round", list(ROUNDS.keys()))

st.markdown("")
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_btn = st.button(
        "⚡ Predict Match Outcome",
        use_container_width=True,
        type="primary",
    )

if predict_btn:
    p1_label = p1_name if p1_name != "Custom Player" else f"Player A (#{p1_rank})"
    p2_label = p2_name if p2_name != "Custom Player" else f"Player B (#{p2_rank})"

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
        with st.spinner("Consulting the model ..."):
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()

        prob_a = result["probability"]
        prob_b = round(1.0 - prob_a, 4)
        pred = result["prediction"]
        winner = p1_label if pred == 1 else p2_label
        win_prob = prob_a if pred == 1 else prob_b

        st.markdown("---")

        r1, r2 = st.columns(2)
        with r1:
            cls = "result-winner" if pred == 1 else "result-loser"
            tag = "PREDICTED WINNER" if pred == 1 else "RUNNER-UP"
            st.markdown(
                f"""
            <div class="result-card {cls}">
                <div class="label">{tag}</div>
                <div class="name">{get_flag(p1_name)} {p1_label}</div>
                <div class="prob">{prob_a:.1%}</div>
                <div class="label">Win Probability</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with r2:
            cls = "result-winner" if pred == 0 else "result-loser"
            tag = "PREDICTED WINNER" if pred == 0 else "RUNNER-UP"
            st.markdown(
                f"""
            <div class="result-card {cls}">
                <div class="label">{tag}</div>
                <div class="name">{get_flag(p2_name)} {p2_label}</div>
                <div class="prob">{prob_b:.1%}</div>
                <div class="label">Win Probability</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.progress(prob_a)

        st.markdown("")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.markdown(
            f"""<div class="stat-pill">
            <div class="value">{SURFACE_EMOJI[surface_label]} {surface_label.split()[0]}</div>
            <div class="label">Surface</div></div>""",
            unsafe_allow_html=True,
        )
        s2.markdown(
            f"""<div class="stat-pill">
            <div class="value">{tourney_label}</div>
            <div class="label">Tournament</div></div>""",
            unsafe_allow_html=True,
        )
        s3.markdown(
            f"""<div class="stat-pill">
            <div class="value">{round_label}</div>
            <div class="label">Round</div></div>""",
            unsafe_allow_html=True,
        )
        rank_gap = abs(p1_rank - p2_rank)
        s4.markdown(
            f"""<div class="stat-pill">
            <div class="value">{rank_gap}</div>
            <div class="label">Rank Gap</div></div>""",
            unsafe_allow_html=True,
        )
        edge = abs(prob_a - prob_b)
        edge_label = "Tight" if edge < 0.1 else "Clear" if edge < 0.3 else "Dominant"
        s5.markdown(
            f"""<div class="stat-pill">
            <div class="value">{edge_label}</div>
            <div class="label">Edge</div></div>""",
            unsafe_allow_html=True,
        )

        if win_prob >= 0.7:
            insight = (
                f"<strong>{winner}</strong> is the heavy favourite at "
                f"<strong>{win_prob:.1%}</strong>. The ranking gap of "
                f"{rank_gap} positions and surface conditions strongly favour them."
            )
        elif win_prob >= 0.55:
            insight = (
                f"<strong>{winner}</strong> has a moderate edge "
                f"(<strong>{win_prob:.1%}</strong>). This match could produce "
                f"an upset — watch for momentum swings on {surface_label.lower()}."
            )
        else:
            insight = (
                f"This is a <strong>coin-flip match</strong> at "
                f"<strong>{win_prob:.1%}</strong> for {winner}. Expect a "
                f"tight contest where mental toughness will decide the outcome."
            )

        st.markdown(
            f"""
        <div class="insight-box">
            <p>💡 {insight}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    except requests.ConnectionError:
        st.error(
            f"Cannot reach the API at **{API_URL}**. "
            "The free-tier Render service may be sleeping — wait 30-60 seconds and retry."
        )
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")

st.markdown("---")
st.markdown("")

f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("**Model**")
    st.caption("RandomForest trained on 7,165 ATP matches (2018-2020)")
with f2:
    st.markdown("**Performance**")
    st.caption("65.3% accuracy | 0.637 log loss | 0.706 AUC")
with f3:
    st.markdown("**Links**")
    st.caption(
        "[GitHub](https://github.com/rayanmazari90/1-mlops-kickoff-repo) · "
        "[W&B Dashboard](https://wandb.ai/bmazari-ieu2024-ie-university/tennis-atp-prediction) · "
        "[API Docs](https://one-mlops-kickoff-repo-1.onrender.com/docs)"
    )

st.markdown(
    "<div style='text-align:center; color:rgba(255,255,255,0.3); "
    "font-size:0.75rem; margin-top:2rem;'>"
    "Group 8 — IE University MsC Business Analytics & Data Science — MLOps 2026"
    "</div>",
    unsafe_allow_html=True,
)
