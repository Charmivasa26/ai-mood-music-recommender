"""
AI Mood-Based Music Recommender — Streamlit Web App
Run: streamlit run app.py

Tabs:
  1. 🎵 Mood Explorer   — Select a mood → get song recommendations
  2. 🎛️ Feature Predictor — Adjust sliders → predict mood
  3. 🔍 Song Inspector   — Pick a song → see its predicted mood
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px

# ──────────────────────────────────────────────────────────────
# Page Configuration (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Mood Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────────────
# Custom CSS — Dark Modern Theme
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background: #0d0d1a; color: #e0e0f0; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #1a0533 0%, #0d1a33 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid #3a1a6a;
    }
    .main-header h1 { color: #c084fc; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #94a3b8; margin: 0.4rem 0 0 0; font-size: 1rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #13131f;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #2a2a3f;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
    }

    /* Song cards */
    .song-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d2d4a;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        transition: border-color 0.2s;
    }
    .song-card:hover { border-color: #7c3aed; }
    .song-title { color: #e2e8f0; font-weight: 700; font-size: 0.95rem; }
    .song-artist { color: #94a3b8; font-size: 0.82rem; margin-top: 2px; }
    .song-meta { color: #64748b; font-size: 0.78rem; margin-top: 6px; }

    /* Mood badge */
    .mood-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-left: 8px;
    }

    /* Metric cards */
    .metric-card {
        background: #13131f;
        border: 1px solid #2a2a3f;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #c084fc; }
    .metric-label { font-size: 0.8rem; color: #64748b; margin-top: 2px; }

    /* Confidence bar */
    .conf-row { display: flex; align-items: center; margin: 4px 0; gap: 10px; }
    .conf-label { width: 140px; font-size: 0.82rem; color: #94a3b8; text-align: right; }
    .conf-bar-bg { flex: 1; background: #1e1e30; border-radius: 4px; height: 14px; }
    .conf-bar-fill { height: 100%; border-radius: 4px; }
    .conf-pct { width: 45px; font-size: 0.82rem; color: #c084fc; font-weight: 700; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.55rem 1.5rem;
    }
    .stButton > button:hover { opacity: 0.9; }

    /* Sliders */
    .stSlider > div > div > div > div { background: #7c3aed !important; }

    /* Select boxes */
    .stSelectbox > div > div { background: #13131f; border-color: #2a2a3f; color: #e0e0f0; }

    /* Section title */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #c084fc;
        margin: 1rem 0 0.6rem 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #2a2a3f;
    }

    /* Divider */
    hr { border-color: #2a2a3f; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

MOOD_COLORS = {
    "High Energy":         "#ef4444",
    "Chill/Acoustic":      "#22d3ee",
    "Groovy":              "#a78bfa",
    "Focus/Instrumental":  "#34d399",
    "Mixed Vibe":          "#f59e0b",
}

MOOD_EMOJI = {
    "High Energy":         "⚡",
    "Chill/Acoustic":      "🌊",
    "Groovy":              "🕺",
    "Focus/Instrumental":  "🎯",
    "Mixed Vibe":          "🎲",
}

ACTIVE_FEATURE_COLS = FEATURE_COLS  # will be overwritten by load_artifacts

FEATURE_RANGES = {
    'danceability':      (0.0, 1.0, 0.5),
    'energy':            (0.0, 1.0, 0.5),
    'loudness':          (-60.0, 0.0, -10.0),
    'speechiness':       (0.0, 1.0, 0.1),
    'acousticness':      (0.0, 1.0, 0.3),
    'instrumentalness':  (0.0, 1.0, 0.1),
    'liveness':          (0.0, 1.0, 0.2),
    'valence':           (0.0, 1.0, 0.5),
    'tempo':             (50.0, 250.0, 120.0),
}

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# ──────────────────────────────────────────────────────────────
# Load Artifacts (cached)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model…")
def load_artifacts():
    import tensorflow as tf
    model = tf.keras.models.load_model(f"{ARTIFACTS_DIR}/cnn_lstm_model.keras")
    with open(f"{ARTIFACTS_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{ARTIFACTS_DIR}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    df = pd.read_csv(f"{ARTIFACTS_DIR}/processed_dataset.csv")
    feat_path = f"{ARTIFACTS_DIR}/feature_cols.pkl"
    if os.path.exists(feat_path):
        with open(feat_path, "rb") as f2:
            feature_cols = pickle.load(f2)
    else:
        feature_cols = FEATURE_COLS
    return model, scaler, le, df, feature_cols

@st.cache_resource(show_spinner=False)
def load_raw_dataset():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset.csv"))

# ──────────────────────────────────────────────────────────────
# Helper: Engineer extra features (mirrors train_model_v3.py)
# ──────────────────────────────────────────────────────────────
def engineer_features(fv: dict) -> dict:
    fv = dict(fv)
    fv['energy_sq']           = fv['energy'] ** 2
    fv['dance_energy']        = fv['danceability'] * fv['energy']
    fv['acoustic_inv_energy'] = fv['acousticness'] * (1 - fv['energy'])
    fv['valence_dance']       = fv['valence'] * fv['danceability']
    fv['high_energy_flag']    = 1.0 if fv['energy'] > 0.8 else 0.0
    fv['acoustic_flag']       = 1.0 if fv['acousticness'] > 0.7 else 0.0
    fv['groovy_flag']         = 1.0 if fv['danceability'] > 0.7 else 0.0
    fv['instr_flag']          = 1.0 if fv['instrumentalness'] > 0.5 else 0.0
    return fv

# ──────────────────────────────────────────────────────────────
# Helper: Predict mood from raw feature values
# ──────────────────────────────────────────────────────────────
def predict_mood(feature_values, model, scaler, le, feat_cols=None):
    """Always engineers all 17 features before calling scaler."""
    if feat_cols is None:
        feat_cols = ACTIVE_FEATURE_COLS
    feature_values = engineer_features(feature_values)
    arr = np.array([[feature_values.get(c, 0.0) for c in feat_cols]])
    arr_scaled = scaler.transform(arr)
    arr_reshaped = arr_scaled.reshape(1, len(feat_cols), 1)
    probs = model.predict(arr_reshaped, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_label = le.inverse_transform([pred_idx])[0]
    prob_dict = {le.inverse_transform([i])[0]: float(probs[i]) for i in range(len(probs))}
    return pred_label, prob_dict

def song_card(row, show_genre=True):
    mood = row.get('manual_vibe', 'Unknown')
    color = MOOD_COLORS.get(mood, '#888')
    emoji = MOOD_EMOJI.get(mood, '🎵')
    genre_tag = f"<span style='color:#64748b;'>· {row.get('track_genre','')}</span>" if show_genre else ""
    popularity = row.get('popularity', 0)
    stars = "★" * min(5, max(1, int(popularity / 20)))
    return f"""
    <div class="song-card">
        <div class="song-title">
            {emoji} {row.get('track_name','Unknown')}
            <span class="mood-badge" style="background:{color}22;color:{color};border:1px solid {color}55;">
                {mood}
            </span>
        </div>
        <div class="song-artist">{row.get('artists','Unknown Artist')} {genre_tag}</div>
        <div class="song-meta">
            <span style="color:#f59e0b;">{stars}</span>
            <span style="margin-left:8px;">Popularity: {popularity}</span>
        </div>
    </div>
    """

# ──────────────────────────────────────────────────────────────
# Helper: Render confidence bars
# ──────────────────────────────────────────────────────────────
def confidence_bars(prob_dict):
    sorted_moods = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    bars_html = ""
    for mood, prob in sorted_moods:
        color = MOOD_COLORS.get(mood, '#888')
        pct = prob * 100
        emoji = MOOD_EMOJI.get(mood, '🎵')
        bars_html += f"""
        <div class="conf-row">
            <div class="conf-label">{emoji} {mood}</div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
            </div>
            <div class="conf-pct">{pct:.1f}%</div>
        </div>
        """
    return f"""
    <div style="background:#13131f;border:1px solid #2a2a3f;border-radius:12px;padding:1.2rem;margin-top:0.8rem;">
        <div style="color:#94a3b8;font-size:0.85rem;font-weight:700;margin-bottom:10px;">
            CONFIDENCE SCORES
        </div>
        {bars_html}
    </div>
    """

# ──────────────────────────────────────────────────────────────
# App Header
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎵 AI Mood-Based Music Recommender</h1>
    <p>Hybrid CNN-LSTM Deep Learning Model · 2,000 Songs · 4 Mood Classes</p>
</div>
""", unsafe_allow_html=True)

# Try loading artifacts; show error if not yet trained
try:
    model, scaler, le, df_processed, ACTIVE_FEATURE_COLS = load_artifacts()
    df_raw = load_raw_dataset()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(f"""
    **Model artifacts not found.**
    Please run the training script first:
    ```
    cd mood_music_project
    python train_model.py
    ```
    Error: `{e}`
    """)
    st.stop()

MOODS = list(le.classes_)

# ──────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🎵  Mood Explorer",
    "🎛️  Feature Predictor",
    "🔍  Song Inspector"
])

# ══════════════════════════════════════════════════════════════
# TAB 1: Mood Explorer
# ══════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<div class="section-title">Choose Your Mood</div>', unsafe_allow_html=True)
        selected_mood = st.selectbox(
            "Mood", MOODS, label_visibility="collapsed",
            format_func=lambda m: f"{MOOD_EMOJI.get(m,'')} {m}"
        )
        num_songs = st.slider("Number of recommendations", 3, 20, 8)
        sort_by = st.selectbox("Sort by", ["Popularity (High→Low)", "Random"])
        st.markdown("<br>", unsafe_allow_html=True)
        recommend_btn = st.button("🎵 Get Recommendations", use_container_width=True)

    with col_right:
        color = MOOD_COLORS.get(selected_mood, '#888')
        emoji = MOOD_EMOJI.get(selected_mood, '🎵')
        st.markdown(f"""
        <div style="background:{color}18;border:1px solid {color}44;border-radius:14px;
                    padding:1.2rem 1.5rem;margin-bottom:1rem;">
            <div style="font-size:2rem;">{emoji}</div>
            <div style="font-size:1.4rem;font-weight:800;color:{color};margin:4px 0;">{selected_mood}</div>
            <div style="color:#94a3b8;font-size:0.85rem;">
                {df_raw[df_raw['manual_vibe']==selected_mood].shape[0]} songs in this mood category
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Distribution mini-chart
        mood_counts = df_raw['manual_vibe'].value_counts().reset_index()
        mood_counts.columns = ['Mood', 'Count']
        fig = px.bar(
            mood_counts, x='Mood', y='Count',
            color='Mood',
            color_discrete_map=MOOD_COLORS,
            template='plotly_dark',
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False, height=200,
            margin=dict(l=0, r=0, t=10, b=30),
            font=dict(color='#94a3b8', size=11)
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    if recommend_btn or True:  # show on load
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{emoji} Recommended Songs — {selected_mood}</div>',
                    unsafe_allow_html=True)

        mood_df = df_raw[df_raw['manual_vibe'] == selected_mood].copy()

        if sort_by == "Popularity (High→Low)":
            mood_df = mood_df.sort_values('popularity', ascending=False)
        else:
            mood_df = mood_df.sample(frac=1, random_state=None)

        recs = mood_df.head(num_songs)

        cols = st.columns(2)
        for idx, (_, row) in enumerate(recs.iterrows()):
            with cols[idx % 2]:
                st.markdown(song_card(row), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2: Feature Predictor
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Adjust Audio Features → Predict Mood</div>',
                unsafe_allow_html=True)

    FEATURE_HELP = {
        'danceability':     "How suitable for dancing (rhythm, beat strength)",
        'energy':           "Intensity and activity level",
        'loudness':         "Overall loudness in dB (typically -60 to 0)",
        'speechiness':      "Presence of spoken words",
        'acousticness':     "Whether the track is acoustic",
        'instrumentalness': "Predicts absence of vocals",
        'liveness':         "Presence of audience/live performance",
        'valence':          "Musical positiveness / happiness",
        'tempo':            "Beats per minute (BPM)",
    }

    col_sliders, col_result = st.columns([1.2, 1])

    feature_values = {}
    with col_sliders:
        left_feats = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness']
        right_feats = ['instrumentalness', 'liveness', 'valence', 'tempo']

        c1, c2 = st.columns(2)
        with c1:
            for feat in left_feats:
                mn, mx, default = FEATURE_RANGES[feat]
                val = st.slider(
                    f"**{feat.capitalize()}**",
                    min_value=float(mn), max_value=float(mx), value=float(default),
                    step=0.01 if mx <= 1.0 else 1.0,
                    help=FEATURE_HELP[feat]
                )
                feature_values[feat] = val

        with c2:
            for feat in right_feats:
                mn, mx, default = FEATURE_RANGES[feat]
                val = st.slider(
                    f"**{feat.capitalize()}**",
                    min_value=float(mn), max_value=float(mx), value=float(default),
                    step=0.01 if mx <= 1.0 else 1.0,
                    help=FEATURE_HELP[feat]
                )
                feature_values[feat] = val

        predict_btn = st.button("🧠 Predict Mood", use_container_width=True)

    with col_result:
        if predict_btn or True:
            pred_label, prob_dict = predict_mood(feature_values, model, scaler, le, ACTIVE_FEATURE_COLS)
            color = MOOD_COLORS.get(pred_label, '#888')
            emoji = MOOD_EMOJI.get(pred_label, '🎵')
            confidence = prob_dict[pred_label] * 100

            st.markdown(f"""
            <div style="background:{color}18;border:2px solid {color}55;border-radius:16px;
                        padding:1.5rem;text-align:center;margin-bottom:1rem;">
                <div style="font-size:3rem;">{emoji}</div>
                <div style="font-size:1.8rem;font-weight:900;color:{color};margin:6px 0;">
                    {pred_label}
                </div>
                <div style="color:#94a3b8;">
                    Confidence: <span style="color:{color};font-weight:700;">{confidence:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(confidence_bars(prob_dict), unsafe_allow_html=True)

            # Show 3 matching songs
            st.markdown('<div class="section-title" style="margin-top:1rem;">Matching Songs</div>',
                        unsafe_allow_html=True)
            matches = df_raw[df_raw['manual_vibe'] == pred_label].nlargest(3, 'popularity')
            for _, row in matches.iterrows():
                st.markdown(song_card(row, show_genre=False), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3: Song Inspector
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Select a Song → Test Model Prediction</div>',
                unsafe_allow_html=True)

    col_pick, col_details = st.columns([1, 1.5])

    with col_pick:
        search_query = st.text_input("🔎 Search song / artist", placeholder="e.g. Zara Zara…")

        if search_query:
            mask = (
                df_raw['track_name'].str.contains(search_query, case=False, na=False) |
                df_raw['artists'].str.contains(search_query, case=False, na=False)
            )
            display_df = df_raw[mask].head(50)
        else:
            display_df = df_raw.nlargest(100, 'popularity')

        song_options = [
            f"{row['track_name']} — {str(row['artists'])[:30]}"
            for _, row in display_df.iterrows()
        ]

        if not song_options:
            st.warning("No songs found. Try a different search.")
        else:
            selected_song_str = st.selectbox("Select song", song_options, label_visibility="collapsed")
            selected_idx = song_options.index(selected_song_str)
            selected_row = display_df.iloc[selected_idx]

            st.markdown(song_card(selected_row), unsafe_allow_html=True)
            inspect_btn = st.button("🔍 Inspect with AI", use_container_width=True)

    with col_details:
        if not song_options:
            st.info("Search for a song on the left.")
        else:
            # Get features for selected song
            song_feats = {col: selected_row[col] for col in FEATURE_COLS}

            # Predict
            pred_label, prob_dict = predict_mood(song_feats, model, scaler, le, ACTIVE_FEATURE_COLS)
            true_label = selected_row['manual_vibe']

            pred_color = MOOD_COLORS.get(pred_label, '#888')
            true_color = MOOD_COLORS.get(true_label, '#888')
            match = pred_label == true_label

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{true_color};">
                        {MOOD_EMOJI.get(true_label,'')}
                    </div>
                    <div style="color:{true_color};font-weight:700;font-size:0.85rem;">{true_label}</div>
                    <div class="metric-label">Labeled Mood</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{pred_color};">
                        {MOOD_EMOJI.get(pred_label,'')}
                    </div>
                    <div style="color:{pred_color};font-weight:700;font-size:0.85rem;">{pred_label}</div>
                    <div class="metric-label">AI Predicted</div>
                </div>
                """, unsafe_allow_html=True)

            match_txt = "✅ Correct Prediction" if match else "⚠️ Different from Label"
            match_col = "#34d399" if match else "#f59e0b"
            st.markdown(f"""
            <div style="text-align:center;color:{match_col};font-weight:700;
                        font-size:0.95rem;margin:0.8rem 0;">
                {match_txt}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(confidence_bars(prob_dict), unsafe_allow_html=True)

            # Radar chart of audio features (normalized)
            feats_display = ['danceability', 'energy', 'speechiness',
                             'acousticness', 'instrumentalness', 'liveness', 'valence']

            song_feats_eng = engineer_features(song_feats)
            norm_vals = scaler.transform([[song_feats_eng.get(c, 0.0) for c in ACTIVE_FEATURE_COLS]])[0]
            feat_idx = [ACTIVE_FEATURE_COLS.index(f) for f in feats_display if f in ACTIVE_FEATURE_COLS]
            radar_vals = [norm_vals[i] for i in feat_idx]

            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=feats_display + [feats_display[0]],
                fill='toself',
                fillcolor=pred_color + '33',
                line=dict(color=pred_color, width=2),
                name=pred_label
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], color='#555'),
                    angularaxis=dict(color='#94a3b8'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8', size=11),
                showlegend=False,
                height=280,
                margin=dict(l=30, r=30, t=20, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="text-align:center;color:#374151;font-size:0.78rem;padding:0.5rem 0;">
    AI Mood-Based Music Recommender · ECSCI24305 Deep Learning Project ·
    Hybrid CNN-LSTM · Dataset: Kaggle + Custom Mood Labels
</div>
""", unsafe_allow_html=True)