import base64
import io
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import librosa
import requests
from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context, no_update
import xgboost as xgb
from pipeline import get_youtube_stats
# =========================================================
# LOAD & CLEAN DATA  (SUMMARY TAB)
# =========================================================

data = pd.read_csv("merged_final_data.csv")

# Clean tempo values
data["tempo"] = (
    data["tempo"]
    .astype(str)
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .astype(float)
)

# Ensure numeric fields
data["energy"] = pd.to_numeric(data["energy"], errors="coerce")

# Convert duration_ms → minutes
data["duration_min"] = (data["duration_ms"] / 60000).round(2)

# Remove missing rows
clean_data = data.dropna(subset=["tempo", "energy"])

# Dropdown options for SUMMARY tab
artist_options = [
    {"label": artist, "value": artist}
    for artist in sorted(clean_data["Artist"].dropna().unique())
]

# Precompute recommendation matrix (log space for Views/Likes/Comments)
rec_feature_cols = [c for c in ["Views", "Likes", "Comments"] if c in clean_data.columns]
rec_meta_cols = [c for c in ["Artist", "Track"] if c in clean_data.columns]
rec_matrix = None
rec_meta = None
if rec_feature_cols:
    rec_matrix = np.log1p(clean_data[rec_feature_cols].fillna(0).to_numpy())
    rec_meta = clean_data[rec_meta_cols] if rec_meta_cols else None

# =========================================================
# LOAD MODEL DATA (PREDICTIONS FOR DEEP LEARNING TAB)
# =========================================================

# CSV without embeddings, but WITH Predicted_Popularity & Predicted_Marketability
model_data = pd.read_csv("merged_no_embeddings.csv")

# Dropdown options for MODEL tab (artists)
model_artist_options = [
    {"label": artist, "value": artist}
    for artist in sorted(model_data["Artist"].dropna().unique())
]

# Dropdown options for MODEL tab (albums) — only if column exists
if "Album" in model_data.columns:
    model_album_options = [
        {"label": album, "value": album}
        for album in sorted(model_data["Album"].dropna().unique())
    ]
else:
    model_album_options = []

# XGB feature columns derived from model_data
xgb_target_cols = ["Predicted_Popularity", "Predicted_Marketability"]
numeric_cols = model_data.select_dtypes(include=["float", "int"]).columns.tolist()
drop_cols = set(xgb_target_cols + ["Spotify_Track_Popularity", "Marketability"])
xgb_feature_cols = [c for c in numeric_cols if c not in drop_cols]

# =========================================================
# DASH APP SETUP
# =========================================================

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # Required for Render
app.title = "The Science of Song Success"

# =========================================================
# MODEL LOADING (PyTorch)
# =========================================================

XGB_MODEL_POP_PATH = "xgb_popularity.pkl"
XGB_MODEL_MARKET_PATH = "xgb_marketability.pkl"
XGB_SCALER_PATH = "xgb_scaler.pkl"
xgb_pop_model = None
xgb_market_model = None
xgb_scaler = None
GENIUS_TOKEN = os.environ.get("GENIUS_API_TOKEN")
SR = 16000  # sample rate for lightweight audio feature extraction

# =========================================================
# MODEL LOADING (XGB artifacts)
# =========================================================

try:
    if Path(XGB_MODEL_POP_PATH).exists() and Path(XGB_MODEL_MARKET_PATH).exists() and Path(XGB_SCALER_PATH).exists():
        xgb_pop_model = joblib.load(XGB_MODEL_POP_PATH)
        xgb_market_model = joblib.load(XGB_MODEL_MARKET_PATH)
        xgb_scaler = joblib.load(XGB_SCALER_PATH)
except Exception:
    xgb_pop_model = None
    xgb_market_model = None
    xgb_scaler = None
# XGB artifacts (if present)
try:
    if Path(XGB_MODEL_POP_PATH).exists() and Path(XGB_MODEL_MARKET_PATH).exists() and Path(XGB_SCALER_PATH).exists():
        xgb_pop_model = joblib.load(XGB_MODEL_POP_PATH)
        xgb_market_model = joblib.load(XGB_MODEL_MARKET_PATH)
        xgb_scaler = joblib.load(XGB_SCALER_PATH)
except Exception:
    xgb_pop_model = None
    xgb_market_model = None
    xgb_scaler = None


# Basic audio feature extraction for XGB fallback
def compute_basic_audio_features(raw_bytes, max_duration=10.0):
    """
    Decode audio and extract lightweight features.
    Limits decode to max_duration seconds to avoid long runtimes on large files.
    """
    y, sr = librosa.load(
        io.BytesIO(raw_bytes),
        sr=SR,
        mono=True,
        duration=max_duration,
    )
    if y is None or y.size == 0:
        raise ValueError("Failed to decode audio")
    tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.mean(tempo_arr)) if np.ndim(tempo_arr) else float(tempo_arr)
    rms_scalar = float(np.mean(librosa.feature.rms(y=y)))
    loudness = float(20 * np.log10(rms_scalar + 1e-9))
    duration_ms = int(len(y) / sr * 1000)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    return {
        "tempo": tempo,
        "loudness": loudness,
        "duration_ms": duration_ms,
        "zcr": zcr,
        "spec_centroid": spec_centroid,
        "spec_bandwidth": spec_bandwidth,
        "spectral_contrast": spectral_contrast,
    }


def build_xgb_feature_row(base_row, audio_feats, yt_stats):
    """
    Build a feature dict aligned to xgb_feature_cols.
    Start from base_row (e.g., a matched catalog row) then overlay audio/yt stats.
    """
    row = dict(base_row) if base_row else {k: 0.0 for k in xgb_feature_cols}
    if audio_feats:
        for k, v in audio_feats.items():
            if k in row:
                row[k] = v
    if yt_stats and "data" in yt_stats and not yt_stats.get("error"):
        d = yt_stats["data"]
        views = float(d.get("views", 0.0))
        likes = float(d.get("likes", 0.0))
        comments = float(d.get("comments", 0.0))
        if "Views" in row: row["Views"] = views
        if "Likes" in row: row["Likes"] = likes
        if "Comments" in row: row["Comments"] = comments
        if "log_views" in row: row["log_views"] = np.log1p(views)
        if "log_likes" in row: row["log_likes"] = np.log1p(likes)
        if "log_comments" in row: row["log_comments"] = np.log1p(comments)
        if "like_view_ratio" in row: row["like_view_ratio"] = likes / (views + 1)
        if "comment_view_ratio" in row: row["comment_view_ratio"] = comments / (views + 1)
        if "engagement_rate" in row: row["engagement_rate"] = (likes + comments) / (views + 1)
    return row




def match_catalog_row(title, artist):
    """
    Try to find a matching row in model_data using title (and optional artist).
    Returns a dict of feature values or None.
    """
    if not title or model_data is None:
        return None
    if "Track" not in model_data.columns:
        return None
    df = model_data
    title_lower = title.strip().lower()
    mask = df["Track"].astype(str).str.lower() == title_lower
    if artist and "Artist" in df.columns:
        artist_lower = artist.strip().lower()
        mask = mask & (df["Artist"].astype(str).str.lower() == artist_lower)
    matches = df[mask]
    if matches.empty:
        mask = df["Track"].astype(str).str.lower().str.contains(title_lower, na=False)
        if artist and "Artist" in df.columns:
            artist_lower = artist.strip().lower()
            mask = mask & df["Artist"].astype(str).str.lower().str.contains(artist_lower, na=False)
        matches = df[mask]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return {col: row[col] for col in xgb_feature_cols if col in row}


# Genius lyrics helper
def fetch_lyrics_url(title, artist=None):
    if not GENIUS_TOKEN:
        return "GENIUS_API_TOKEN not set"
    query = f"{title} {artist}" if artist else title
    try:
        resp = requests.get(
            "https://api.genius.com/search",
            params={"q": query},
            headers={"Authorization": f"Bearer {GENIUS_TOKEN}"},
            timeout=10,
        )
        resp.raise_for_status()
        hits = resp.json().get("response", {}).get("hits", [])
        if not hits:
            return "No lyrics found"
        top = hits[0]["result"]
        return top.get("url", "No URL returned")
    except Exception as e:
        return f"Genius lookup failed: {e}"


# XGB feature selection helper (based on model_data)
xgb_target_cols = ["Predicted_Popularity", "Predicted_Marketability"]
xgb_feature_cols = []
if "model_data" in globals():
    numeric_cols = model_data.select_dtypes(include=["float", "int"]).columns.tolist()
    drop_cols = set(xgb_target_cols + ["Spotify_Track_Popularity", "Marketability"])
    xgb_feature_cols = [c for c in numeric_cols if c not in drop_cols]


def xgb_preprocess(df):
    """
    Select and scale features for XGB models. Expects columns present in xgb_feature_cols.
    """
    if not xgb_feature_cols:
        raise ValueError("No XGB feature columns available.")
    missing = [c for c in xgb_feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing XGB feature columns: {', '.join(missing)}")
    feats = df[xgb_feature_cols].astype(float).fillna(0)
    if xgb_scaler is None:
        raise ValueError("XGB scaler not loaded.")
    return xgb_scaler.transform(feats)

# Helper: Team member card style
def card():
    return {
        "backgroundColor": "white",
        "padding": "15px",
        "borderRadius": "10px",
        "margin": "10px",
        "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
        "width": "250px",
    }

# =========================================================
# LAYOUT
# =========================================================

app.layout = html.Div(
    style={
        "backgroundColor": "#faf7f7",
        "fontFamily": "Georgia, serif",
        "padding": "25px",
        "color": "#1a1a1a",
    },
    children=[
        # ---------------- HEADER ----------------
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "flex-start",
                "gap": "20px",
            },
            children=[
                html.Img(
                    src="/assets/ucdavis_logo.png",
                    style={"height": "65px", "marginRight": "15px"},
                ),
                html.H1(
                    "The Science of Song Success",
                    style={
                        "color": "#0b2f59",
                        "fontSize": "40px",
                        "marginBottom": "5px",
                    },
                ),
            ],
        ),

        html.Div(
            style={
                "borderBottom": "3px solid #d4a72c",
                "width": "180px",
                "marginTop": "5px",
                "marginBottom": "20px",
            }
        ),

        html.P(
            """
            For our capstone project, we built an interactive dashboard to explore how Spotify audio 
            features and YouTube engagement metrics relate to one another — and how these patterns 
            help explain a song's success. This dashboard includes summary statistics, a deep learning 
            model overview, key visualizations, and a polished final report summary.
            """,
            style={
                "fontSize": "18px",
                "textAlign": "center",
                "lineHeight": "1.7",
                "maxWidth": "1100px",
                "margin": "auto",
                "marginBottom": "30px",
            },
        ),

        # ---------------- TABS ----------------
        dcc.Tabs(
            id="tabs",
            value="intro",
            colors={"border": "#0b2f59", "primary": "#0b2f59", "background": "#faf7f7"},
            children=[
                dcc.Tab(label="Introduction", value="intro"),
                dcc.Tab(label="Summary Statistics", value="summary"),
                dcc.Tab(label="Model", value="model"),
                dcc.Tab(label="Visuals", value="visuals"),
                dcc.Tab(label="Final Report Summary", value="report"),
                dcc.Tab(label="Team & Acknowledgments", value="team"),




            ],
        ),
        dcc.Store(id="selected-visual"),
        html.Div(id="tabs-content"),
        html.Br(),

        # --------------- FOOTER ----------------
        html.Footer(
            "Developed by Team 13 — STA 160 Capstone | UC Davis, Fall 2025",
            style={
                "textAlign": "center",
                "fontSize": "14px",
                "color": "#555",
                "marginTop": "40px",
                "paddingTop": "15px",
                "borderTop": "1px solid #ccc",
            },
        ),
    ],
)

# =========================================================
# TAB CALLBACK — CONTENT FOR EACH TAB
# =========================================================

@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value")
)
def render_tab(selected):
    if selected == "intro":
        return html.Div(
            style={
                "maxWidth": "1000px",
                "margin": "auto",
                "lineHeight": "1.8",
            },
            children=[
                html.H2("Introduction", style={"color": "#0b2f59"}),
                html.P(
                    """
                    In today’s music industry, success is increasingly shaped by digital platforms such as 
                    Spotify and YouTube, where millions of songs compete for listener attention. While artistic 
                    creativity remains central to music production, a song’s popularity is often influenced by 
                    measurable audio characteristics, listener behavior, and platform-driven dynamics. 
                    Understanding what drives popularity has become both a creative and analytical challenge 
                    for artists, producers, and record labels.
                    """,
                ),
                html.P(
                    """
                    The purpose of this project is to examine which song-level features best predict popularity 
                    and to assess whether data-driven models can reliably explain success in modern streaming 
                    environments. By integrating Spotify audio features with YouTube engagement metrics, we 
                    investigate how musical structure and listener response interact across platforms.
                    """,
                ),
                html.P(
                    """
                    Our approach combines exploratory data analysis, statistical modeling, and machine learning. 
                    We first clean and merge large-scale Spotify and YouTube datasets, then evaluate relationships 
                    between musical attributes such as tempo, energy, loudness, and duration and audience 
                    engagement measures including views, likes, and comments. We apply regression techniques, 
                    regularization methods, and deep learning models to assess predictive performance, using 
                    cross-validation metrics to evaluate accuracy.
                    """,
                ),
                html.P(
                    """
                    The results indicate that while certain audio features are consistently associated with 
                    engagement, musical characteristics alone explain only a limited portion of popularity. 
                    This finding highlights the importance of external factors such as artist visibility, 
                    promotion, and platform algorithms, which are not captured in audio signals. To make these 
                    insights accessible, we developed an interactive dashboard that allows users to explore 
                    data patterns, compare songs, and understand the strengths and limitations of predictive models.
                    """
                ),
            ],
        )

    # ------------------------------------------------------
    # SUMMARY STATISTICS TAB
    # ------------------------------------------------------
    if selected == "summary":
        return html.Div(
            [
                html.H2("Summary Statistics", style={"color": "#0b2f59"}),

                html.P(
                    """
                    Below are descriptive summaries of key Spotify and YouTube variables:
                     • tempo — beats per minute (BPM), describing rhythmic speed  
                     • energy — musical intensity on a 0–1 scale  
                     • loudness — average decibel level of the track  
                     • duration_min — track length in minutes  
                     • views, likes, comments — listener engagement on YouTube  

                    Use the filters below to explore trends across 5,000+ songs.
                    """,
                    style={"whiteSpace": "pre-line", "fontSize": "16px"},
                ),

                html.Label("Filter by Artist:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="artist-dropdown",
                    options=artist_options,
                    placeholder="Select an artist...",
                ),
                html.Br(),

                html.Label("Search Songs or Albums:", style={"fontWeight": "bold"}),
                dcc.Input(
                    id="search-input",
                    type="text",
                    placeholder="Type song, album, or keyword...",
                    style={"width": "100%", "padding": "10px"},
                ),
                html.Br(), html.Br(),

                dash_table.DataTable(
                    id="summary-table",
                    columns=[
                        {"name": "Artist", "id": "Artist"},
                        {"name": "Views", "id": "Views"},
                        {"name": "Likes", "id": "Likes"},
                        {"name": "Comments", "id": "Comments"},
                        {"name": "tempo", "id": "tempo"},
                        {"name": "energy", "id": "energy"},
                        {"name": "duration_min", "id": "duration_min"},
                    ],
                    data=clean_data.to_dict("records"),
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#0b2f59",
                        "color": "white",
                        "fontWeight": "bold",
                    },
                ),
            ]
        )

    # ------------------------------------------------------
    # MODEL TAB  (XGB ONLY)
    # ------------------------------------------------------
    elif selected == "model":
        return html.Div(
            [
                html.H2("Model", style={"color": "#0b2f59"}),
                html.P(
                    "View precomputed popularity and marketability scores from our XGBoost models or run them on your own CSV.",
                    style={"fontSize": "17px"},
                ),
                html.P(
                    "Tip: include the song title (and artist) so we can fetch YouTube stats and lyrics for better context.",
                    style={"fontSize": "13px", "color": "#444"},
                ),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))", "gap": "10px"},
                    children=[
                        dcc.Dropdown(
                            id="model-artist-dropdown",
                            options=model_artist_options,
                            placeholder="Filter by artist...",
                        ),
                        dcc.Dropdown(
                            id="model-album-dropdown",
                            options=model_album_options,
                            placeholder="Filter by album...",
                        ),
                        dcc.Input(
                            id="model-search-input",
                            type="text",
                            placeholder="Search track/artist...",
                        ),
                    ],
                ),
                html.Br(),
                dash_table.DataTable(
                    id="model-table",
                    columns=[
                        {"name": "Artist", "id": "Artist"},
                        {"name": "Track", "id": "Track"},
                        {"name": "Album", "id": "Album"},
                        {"name": "Predicted_Popularity", "id": "Predicted_Popularity"},
                        {"name": "Predicted_Marketability", "id": "Predicted_Marketability"},
                    ],
                    data=model_data.head(50).to_dict("records"),
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#0b2f59",
                        "color": "white",
                        "fontWeight": "bold",
                    },
                ),
                html.Br(),
                html.H4("Run XGB on your CSV", style={"color": "#0b2f59"}),
                html.P(
                    "Upload a CSV containing the numeric feature columns from merged_no_embeddings.csv to score songs locally. "
                    "Ensure xgb_popularity.pkl, xgb_marketability.pkl, and xgb_scaler.pkl are present in the repo root.",
                    style={"fontSize": "14px", "color": "#444"},
                ),
                dcc.Upload(
                    id="xgb-upload",
                    children=html.Div(
                        ["Drag and drop CSV for XGB inference, or ", html.A("select a file", style={"color": "#0b2f59"})]
                    ),
                    style={
                        "width": "100%",
                        "height": "90px",
                        "lineHeight": "90px",
                        "borderWidth": "2px",
                        "borderStyle": "dashed",
                        "borderColor": "#0b2f59",
                        "textAlign": "center",
                        "borderRadius": "10px",
                        "marginBottom": "12px",
                        "backgroundColor": "white",
                    },
                    multiple=False,
                ),
                html.Div(id="xgb-upload-output"),
                html.Br(),
                html.H4("Lyrics lookup (Genius)", style={"color": "#0b2f59"}),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))", "gap": "10px"},
                    children=[
                        dcc.Input(id="lyrics-title", type="text", placeholder="Song title"),
                        dcc.Input(id="lyrics-artist", type="text", placeholder="Artist (optional)"),
                    ],
                ),
                html.Button(
                    "Find Lyrics",
                    id="lyrics-button",
                    n_clicks=0,
                    style={
                        "marginTop": "10px",
                        "backgroundColor": "#0b2f59",
                        "color": "white",
                        "padding": "8px 14px",
                        "borderRadius": "8px",
                        "border": "none",
                        "cursor": "pointer",
                    },
                ),
                html.Div(id="lyrics-output", style={"marginTop": "8px"}),
                html.Br(),
                html.H4("End-to-end XGB (Audio + Title)", style={"color": "#0b2f59"}),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))", "gap": "10px"},
                    children=[
                        dcc.Input(id="xgb-song-title", type="text", placeholder="Song title (recommended)"),
                        dcc.Input(id="xgb-song-artist", type="text", placeholder="Artist (optional)"),
                    ],
                ),
                html.Br(),
                dcc.Upload(
                    id="xgb-audio",
                    children=html.Div(
                        ["Drag and drop audio for XGB scoring, or ", html.A("select a file", style={"color": "#0b2f59"})]
                    ),
                    style={
                        "width": "100%",
                        "height": "90px",
                        "lineHeight": "90px",
                        "borderWidth": "2px",
                        "borderStyle": "dashed",
                        "borderColor": "#0b2f59",
                        "textAlign": "center",
                        "borderRadius": "10px",
                        "marginBottom": "12px",
                        "backgroundColor": "white",
                    },
                    multiple=False,
                ),
                html.Div(id="xgb-audio-status", style={"fontSize": "12px", "color": "#0b2f59", "marginTop": "4px"}),
                html.Button(
                    "Run XGB End-to-End",
                    id="xgb-e2e-button",
                    n_clicks=0,
                    style={
                        "marginTop": "8px",
                        "backgroundColor": "#0b2f59",
                        "color": "white",
                        "padding": "8px 14px",
                        "borderRadius": "8px",
                        "border": "none",
                        "cursor": "pointer",
                    },
                ),
                html.Div(id="xgb-e2e-output", style={"marginTop": "10px"}),
            ]
        )

    # ------------------------------------------------------
    # VISUALS TAB
    # ------------------------------------------------------
    elif selected == "visuals":
        return html.Div(
            [
                html.H2("Visualizations", style={"color": "#0b2f59"}),
    
                html.Div(
                    style={
                        "display": "flex",
                        "gap": "20px",
                        "marginTop": "20px"
                    },
                    children=[
    
                        # ================= LEFT CATEGORY MENU =================
                        html.Div(
                            style={
                                "width": "260px",
                                "backgroundColor": "white",
                                "padding": "20px",
                                "borderRadius": "10px",
                                "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                                "height": "fit-content"
                            },
                            children=[
    
                                html.H3("Choose a Visualization", 
                                        style={"color": "#0b2f59", "marginBottom": "15px"}),
    
                                # ---- AUDIO FEATURE VISUALS ----
                                html.H4("Audio Feature Visuals"),
                                dcc.RadioItems(
                                    id="vis-select",
                                    options=[
                                        {"label": "Correlation Heatmap", "value": "heatmap"},
                                        {"label": "Scatter Matrix", "value": "matrix"},
                                        {"label": "Audio Map (2D)", "value": "pca"},
                                    ],
                                    value="heatmap",
                                    style={"marginBottom": "20px"}
                                ),
    
                                # ---- ENGAGEMENT VISUALS ----
                                html.H4("Engagement Visuals"),
                                dcc.RadioItems(
                                    id="vis-select-eng",
                                    options=[
                                        {"label": "Engagement vs Audio", "value": "eng"},
                                    ],
                                    value=None,
                                    style={"marginBottom": "20px"}
                                ),
    
                                # ---- PREDICTION VISUALS ----
                                html.H4("Prediction Visuals"),
                                dcc.RadioItems(
                                    id="vis-select-pred",
                                    options=[
                                        {"label": "Predicted vs Actual Relationship (Scatter Plot)", "value": "pred_pop"},
                                        {"label": "Distribution — Popularity & Marketability (Histogram)", "value": "pred_mark"},
                                    ],
                                    value=None,
                                    style={"marginBottom": "10px"}
                                ),
                            ]
                        ),
    
                        # ================= RIGHT DISPLAY PANEL =================
                        html.Div(
                            style={
                                "flexGrow": 1,
                                "backgroundColor": "white",
                                "padding": "20px",
                                "borderRadius": "10px",
                                "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"
                            },
                            children=[
                                dcc.Loading(
                                    type="circle",
                                    children=html.Iframe(
                                        id="vis-frame",
                                        src="/assets/triangular_heatmap.html",
                                        style={
                                            "width": "100%",
                                            "height": "900px",
                                            "border": "none"
                                        }
                                    )
                                ),
                                html.Br(),

                                
                                html.P(
                                    id="vis-caption",
                                    style={
                                        "fontSize": "16px",
                                        "fontStyle": "italic",
                                        "color": "#444",
                                        "marginTop": "10px"
                                    }
                                )
                            ]
                        )
                    ]
                ),
            ]
        )
    # ------------------------------------------------------
    # FINAL REPORT SUMMARY TAB
    # ------------------------------------------------------
    elif selected == "report":
        return html.Div(
            style={
                "maxWidth": "1000px",
                "margin": "auto",
                "lineHeight": "1.8",
            },
            children=[
                html.H2("Final Report Summary", style={"color": "#0b2f59"}),
                html.P(
                    [
                        html.Strong("Abstract — "),
                        "This project analyzes a combined Spotify–YouTube dataset to identify musical and engagement features that best predict song popularity. By integrating audio attributes such as tempo, energy, and loudness with listener engagement metrics, we conduct both exploratory analysis and predictive modeling. While several features show meaningful associations with popularity, the results demonstrate that audio characteristics alone provide limited predictive power, highlighting the role of external social and platform-driven factors.",
                    ]
                ),
                html.P(
                    [
                        html.Strong("Motivation — "),
                        "With the rapid growth of streaming platforms, understanding what drives song popularity has become increasingly important. Millions of tracks are released each year, yet only a small fraction achieve widespread success. This project seeks to identify which quantifiable song-level characteristics contribute to popularity and how predictive modeling can support decision-making in the music industry.",
                    ]
                ),
                html.P(
                    [
                        html.Strong("Methodology — "),
                        "We merged Spotify audio features with YouTube engagement data and conducted extensive data cleaning and preprocessing. Our analysis included exploratory visualization, regression modeling, regularization techniques, similarity analysis, and deep learning models using learned audio embeddings. Model performance was evaluated through cross-validation to assess generalizability.",
                    ]
                ),
                html.P(
                    [
                        html.Strong("Key Findings — "),
                        "Tracks with consistent tempo, moderate loudness, higher energy, and specific stylistic characteristics tend to exhibit greater engagement. However, even advanced audio-based models show limited predictive accuracy, indicating that popularity is strongly influenced by non-audio factors such as artist recognition, marketing, and algorithmic promotion.",
                    ]
                ),
                html.Div(
                    html.A(
                    "📄 View Full Final Report (Google Doc)",
                    href="https://docs.google.com/document/d/1wYoX1HOKK6wEcU_JRXqQMJ0NLWfyZAwa5gW9crADFms/edit?usp=sharing",
                    target="_blank",
                    style={
                        "backgroundColor": "#0b2f59",
                        "color": "white",
                        "padding": "12px 24px",
                        "borderRadius": "8px",
                        "textDecoration": "none",
                        "fontWeight": "bold",
                        "display": "inline-block",
                        "marginTop": "12px",
                    },
                ),
                style={"textAlign": "center"},
            ),
        ],
    )
    # ------------------------------------------------------
    # TEAM & ACKNOWLEDGMENTS TAB
    # ------------------------------------------------------
    elif selected == "team":
        return html.Div(
            [
                html.H2("Team & Acknowledgments", style={"color": "#0b2f59"}),

                html.Div(
                    style={"display": "flex", "flexWrap": "wrap"},
                    children=[
                        html.Div(
                            [html.H4("Capri Gallo"), html.P("B.S. Statistical Data Science, UC Davis (2026)")],
                            style=card(),
                        ),
                        html.Div(
                            [html.H4("Alex Garcia"), html.P("B.S. Statistical Data Science, UC Davis (2026)")],
                            style=card(),
                        ),
                        html.Div(
                            [html.H4("Rohan Pillay"), html.P("B.S. Statistical Data Science, UC Davis (2026)")],
                            style=card(),
                        ),
                        html.Div(
                            [html.H4("Edward Ron"), html.P("B.S. Statistical Data Science, UC Davis (2026)")],
                            style=card(),
                        ),
                        html.Div(
                            [html.H4("Yuxiao Tan"), html.P("B.S. Statistical Data Science, UC Davis (2026)")],
                            style=card(),
                        ),
                    ],
                ),

                html.Br(),
                html.H3("Acknowledgments", style={"color": "#0b2f59"}),
                html.P(
                    "Special thanks to Professor Lingfei Cui and the UC Davis Statistics Department "
                    "for their guidance throughout this project."
                ),

                html.Br(),
                html.H3("References", style={"color": "#0b2f59"}),
                html.Ul(
                    [
                        html.Li(
                            "Li, Y. et al. (2024). MERT: Acoustic Music Understanding Model with Large-Scale "
                            "Self-Supervised Training. arXiv:2306.00107."
                        ),
                        html.Li(
                            "Grewal, R. (2025). Spotify–YouTube Data. Kaggle. "
                            "https://www.kaggle.com/datasets/rohitgrewal/spotify-youtube-data"
                        ),
                    ]
                ),
            ]
        )


# =========================================================
# CALLBACK: FILTERING FOR SUMMARY TABLE
# =========================================================

@app.callback(
    Output("summary-table", "data"),
    [Input("artist-dropdown", "value"), Input("search-input", "value")]
)
def update_table(selected_artist, search_text):
    df = clean_data.copy()

    # Filter by selected artist
    if selected_artist:
        df = df[df["Artist"] == selected_artist]

    # Global text search across all string columns
    if search_text and search_text.strip():
        search = search_text.lower()
        string_cols = df.select_dtypes(include="object").columns
        mask = df[string_cols].apply(
            lambda col: col.str.lower().str.contains(search, na=False)
        )
        df = df[mask.any(axis=1)]

    return df.to_dict("records")


# =========================================================
# CALLBACKS: MODEL INFERENCE
# =========================================================


@app.callback(
    Output("xgb-upload-output", "children"),
    Input("xgb-upload", "contents"),
    State("xgb-upload", "filename"),
    prevent_initial_call=True,
)
def run_xgb_upload(contents, filename):
    if contents is None:
        return "No file uploaded."
    if xgb_pop_model is None or xgb_market_model is None or xgb_scaler is None:
        return "XGB artifacts not loaded. Add xgb_popularity.pkl, xgb_marketability.pkl, xgb_scaler.pkl to the repo."
    try:
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.BytesIO(decoded))
        X = xgb_preprocess(df)
        pop_pred = xgb_pop_model.predict(X)
        market_pred = xgb_market_model.predict(X)
        out = df.copy()
        out["xgb_pred_popularity"] = pop_pred
        out["xgb_pred_marketability"] = market_pred
        return dash_table.DataTable(
            data=out.head(200).to_dict("records"),
            page_size=10,
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#0b2f59",
                "color": "white",
                "fontWeight": "bold",
            },
        )
    except Exception as e:
        return f"Error running XGB inference on {filename}: {e}"


@app.callback(
    Output("lyrics-output", "children"),
    Input("lyrics-button", "n_clicks"),
    State("lyrics-title", "value"),
    State("lyrics-artist", "value"),
    prevent_initial_call=True,
)
def get_lyrics_url(n_clicks, title, artist):
    if not title or not title.strip():
        return "Please enter a song title."
    url = fetch_lyrics_url(title.strip(), artist.strip() if artist else None)
    if url.startswith("http"):
        return html.A("View lyrics on Genius", href=url, target="_blank")
    return url


@app.callback(
    [Output("xgb-e2e-output", "children"), Output("xgb-audio-status", "children")],
    [Input("xgb-e2e-button", "n_clicks"), Input("xgb-audio", "contents")],
    State("xgb-song-title", "value"),
    State("xgb-song-artist", "value"),
    State("xgb-audio", "filename"),
    prevent_initial_call=True,
)
def run_xgb_e2e(n_clicks, contents, title, artist, filename):
    try:
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        # Handle pure upload event: show a success message without running model
        if triggered == "xgb-audio":
            if contents and filename:
                return no_update, f"Uploaded file: {filename}"
            if contents:
                return no_update, "File uploaded."
            return no_update, ""

        # Handle run button click
        if xgb_pop_model is None or xgb_market_model is None or xgb_scaler is None:
            return "XGB artifacts not loaded. Ensure xgb_popularity.pkl, xgb_marketability.pkl, xgb_scaler.pkl are in the repo root.", "Artifacts missing."
        title_val = title.strip() if title else ""
        artist_val = artist.strip() if artist else ""
        if not xgb_feature_cols:
            return "XGB feature columns not available.", "Missing feature columns."

        yt_stats = {"error": "Skipped (no title/artist)", "data": {}} if not title_val else get_youtube_stats(title_val, artist_val or None)
        lyrics_link = None
        if title_val:
            lyrics_link = fetch_lyrics_url(title_val, artist_val or None)

        base_row = match_catalog_row(title_val, artist_val)
        audio_feats = {}
        audio_err = None
        status_text = "Analyzing..."
        if contents:
            try:
                _, content_string = contents.split(",")
                decoded = base64.b64decode(content_string)
                audio_feats = compute_basic_audio_features(decoded)
                status_text = f"Analyzed audio from {filename or 'upload'}."
            except Exception as e:
                audio_err = f"Audio feature extraction failed: {e}"
                audio_feats = {}
                status_text = "Audio upload processed but feature extraction failed."
        else:
            audio_err = "No audio provided; using defaults for audio features."
            status_text = "No audio uploaded; using metadata only."

        row = build_xgb_feature_row(base_row, audio_feats, yt_stats)
        df_row = pd.DataFrame([[row.get(col, 0.0) for col in xgb_feature_cols]], columns=xgb_feature_cols)
        df_scaled = xgb_scaler.transform(df_row)
        pop_pred = xgb_pop_model.predict(df_scaled)[0]
        mkt_pred = xgb_market_model.predict(df_scaled)[0]

        details = [
            html.P(f"Predicted Popularity: {pop_pred:.2f}"),
            html.P(f"Predicted Marketability: {mkt_pred:.2f}"),
        ]
        if yt_stats and not yt_stats.get("error"):
            d = yt_stats["data"]
            details.append(html.P(f"YouTube stats - Views: {d.get('views')}, Likes: {d.get('likes')}, Comments: {d.get('comments')}"))
        elif yt_stats and yt_stats.get("error"):
            details.append(html.P(f"YouTube stats error: {yt_stats['error']}"))
        if audio_err:
            details.append(html.P(audio_err, style={"color": "red"}))
        if isinstance(lyrics_link, str) and lyrics_link.startswith("http"):
            details.append(html.A("View lyrics on Genius", href=lyrics_link, target="_blank"))
        elif lyrics_link:
            details.append(html.P(f"Lyrics: {lyrics_link}"))

        # Percentiles and nearest neighbors for context
        pct_parts = []
        nn_lines = []
        if "Predicted_Popularity" in model_data.columns and "Predicted_Marketability" in model_data.columns:
            pop_series = model_data["Predicted_Popularity"].dropna()
            mkt_series = model_data["Predicted_Marketability"].dropna()
            if len(pop_series) > 0:
                pop_pct = float((pop_series < pop_pred).mean() * 100)
                pct_parts.append(f"Popularity is around the {pop_pct:.1f}th percentile of the catalog.")
                pop_diff = (pop_series - pop_pred).abs()
                nn_idx = pop_diff.nsmallest(3).index
                for i in nn_idx:
                    row_nn = model_data.loc[i]
                    nn_lines.append(f"Similar popularity: {row_nn.get('Artist','?')} - {row_nn.get('Track','?')} ({row_nn.get('Predicted_Popularity',0):.2f})")
            if len(mkt_series) > 0:
                mkt_pct = float((mkt_series < mkt_pred).mean() * 100)
                pct_parts.append(f"Marketability is around the {mkt_pct:.1f}th percentile of the catalog.")
                mkt_diff = (mkt_series - mkt_pred).abs()
                nn_idx = mkt_diff.nsmallest(3).index
                for i in nn_idx:
                    row_nn = model_data.loc[i]
                    nn_lines.append(f"Similar marketability: {row_nn.get('Artist','?')} - {row_nn.get('Track','?')} ({row_nn.get('Predicted_Marketability',0):.2f})")

        # Per-sample contributions (popularity)
        try:
            dm = xgb.DMatrix(df_scaled, feature_names=xgb_feature_cols)
            contrib = xgb_pop_model.get_booster().predict(dm, pred_contribs=True)[0][:-1]
            top_idx = np.argsort(np.abs(contrib))[-3:][::-1]
            feat_names = [xgb_feature_cols[i] for i in top_idx]
            feat_vals = contrib[top_idx]
            colors = ["#0b2f59" if v >= 0 else "#c0392b" for v in feat_vals]
            contrib_fig = dcc.Graph(
                figure=go.Figure(
                    data=go.Bar(x=feat_vals, y=feat_names, orientation="h", marker_color=colors),
                    layout=go.Layout(title="Top feature contributions (popularity)", margin=dict(l=80, r=20, t=40, b=40), height=260),
                ),
                style={"height": "260px"},
            )
            details.append(contrib_fig)
        except Exception:
            pass

        if pct_parts:
            details.append(html.P(" ".join(pct_parts), style={"color": "#0b2f59"}))
        if nn_lines:
            details.append(html.Ul([html.Li(t) for t in nn_lines]))

        # Per-sample contributions (if available)
        contrib_fig = None
        try:
            dm = xgb.DMatrix(df_scaled, feature_names=xgb_feature_cols)
            contrib = xgb_pop_model.get_booster().predict(dm, pred_contribs=True)[0]
            contrib = contrib[:-1]  # drop bias term
            top_idx = np.argsort(np.abs(contrib))[-3:][::-1]
            feat_names = [xgb_feature_cols[i] for i in top_idx]
            feat_vals = contrib[top_idx]
            colors = ["#0b2f59" if v >= 0 else "#c0392b" for v in feat_vals]
            contrib_fig = dcc.Graph(
                figure=go.Figure(
                    data=go.Bar(
                        x=feat_vals,
                        y=feat_names,
                        orientation="h",
                        marker_color=colors,
                    ),
                    layout=go.Layout(
                        title="Top feature contributions (popularity)",
                        margin=dict(l=80, r=20, t=40, b=40),
                        height=260,
                    ),
                ),
                style={"height": "260px"},
            )
        except Exception:
            contrib_fig = None

        if contrib_fig:
            details.append(contrib_fig)

        if not status_text:
            status_text = "Analysis complete."
        return html.Div(details), status_text
    except Exception as e:
        return f"End-to-end XGB error: {e}", "Error during processing."
# =========================================================
# CALLBACK: FILTERING FOR MODEL TABLE (PREDICTIONS)
# =========================================================

@app.callback(
    Output("model-table", "data"),
    [
        Input("model-artist-dropdown", "value"),
        Input("model-album-dropdown", "value"),
        Input("model-search-input", "value"),
    ]
)
def update_model_table(selected_artist, selected_album, search_text):
    df = model_data.copy()

    if selected_artist:
        df = df[df["Artist"] == selected_artist]

    if selected_album and "Album" in df.columns:
        df = df[df["Album"] == selected_album]

    if search_text and search_text.strip():
        search = search_text.lower()
        string_cols = df.select_dtypes(include="object").columns
        mask = df[string_cols].apply(
            lambda col: col.str.lower().str.contains(search, na=False)
        )
        df = df[mask.any(axis=1)]

    cols_to_keep = [
        "Artist",
        "Track",
        "Album",
        "Predicted_Popularity",
        "Predicted_Marketability",
    ]
    existing = [c for c in cols_to_keep if c in df.columns]
    return df[existing].to_dict("records")


# =========================================================
# CALLBACK: COMPARISON
# =========================================================

@app.callback(
    [
        Output("xgb-compare-output", "children"),
        Output("xgb-compare-status-a", "children"),
        Output("xgb-compare-status-b", "children"),
    ],
    [
        Input("xgb-compare-button", "n_clicks"),
        Input("xgb-compare-audio-a", "contents"),
        Input("xgb-compare-audio-b", "contents"),
    ],
    [
        State("compare-title-a", "value"),
        State("compare-artist-a", "value"),
        State("compare-title-b", "value"),
        State("compare-artist-b", "value"),
        State("xgb-compare-audio-a", "filename"),
        State("xgb-compare-audio-b", "filename"),
    ],
    prevent_initial_call=True,
)
def run_xgb_compare(n_clicks, contents_a, contents_b, title_a, artist_a, title_b, artist_b, fname_a, fname_b):
    try:
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        status_a = ""
        status_b = ""
        if triggered == "xgb-compare-audio-a":
            if contents_a and fname_a:
                status_a = f"Uploaded A: {fname_a}"
            elif contents_a:
                status_a = "File A uploaded."
        if triggered == "xgb-compare-audio-b":
            if contents_b and fname_b:
                status_b = f"Uploaded B: {fname_b}"
            elif contents_b:
                status_b = "File B uploaded."
        if triggered != "xgb-compare-button":
            return no_update, status_a, status_b

        details_a, status_a_run, preds_a = single_xgb_prediction(title_a.strip() if title_a else "", artist_a.strip() if artist_a else "", contents_a, fname_a)
        details_b, status_b_run, preds_b = single_xgb_prediction(title_b.strip() if title_b else "", artist_b.strip() if artist_b else "", contents_b, fname_b)
        if status_a_run:
            status_a = status_a_run
        if status_b_run:
            status_b = status_b_run

        comparison = []
        if preds_a and preds_b:
            if preds_a["pop"] > preds_b["pop"]:
                comparison.append(html.H4("Song A predicted higher popularity", style={"color": "#0b2f59"}))
            elif preds_b["pop"] > preds_a["pop"]:
                comparison.append(html.H4("Song B predicted higher popularity", style={"color": "#0b2f59"}))
            else:
                comparison.append(html.H4("Popularity scores are equal", style={"color": "#0b2f59"}))
            comparison.append(html.P(f"Song A - Popularity: {preds_a['pop']:.2f} | Marketability: {preds_a['mkt']:.2f}"))
            comparison.append(html.P(f"Song B - Popularity: {preds_b['pop']:.2f} | Marketability: {preds_b['mkt']:.2f}"))

        output = html.Div(
            [
                html.H3("Comparison Result", style={"color": "#0b2f59"}),
                *(comparison if comparison else [html.P("Could not compare (missing predictions).")]),
                html.Br(),
                html.H4("Song A Details", style={"color": "#0b2f59"}),
                html.Div(details_a),
                html.Br(),
                html.H4("Song B Details", style={"color": "#0b2f59"}),
                html.Div(details_b),
            ]
        )
        return output, status_a, status_b
    except Exception as e:
        return f"Comparison error: {e}", "Error", "Error"


# =========================================================
# CALLBACK: Visualization Selector (IFRAME)
# =========================================================

@app.callback(
    [
        Output("vis-select", "value"),
        Output("vis-select-eng", "value"),
        Output("vis-select-pred", "value"),
    ],
    [
        Input("vis-select", "value"),
        Input("vis-select-eng", "value"),
        Input("vis-select-pred", "value"),
    ],
    prevent_initial_call=True
)
def sync_radio_groups(audio_choice, eng_choice, pred_choice):
    ctx = callback_context
    clicked = ctx.triggered[0]["prop_id"].split(".")[0]

    if clicked == "vis-select":
        return audio_choice, None, None
    if clicked == "vis-select-eng":
        return None, eng_choice, None
    if clicked == "vis-select-pred":
        return None, None, pred_choice
    return "heatmap", None, None


@app.callback(
    Output("selected-visual", "data"),
    [
        Input("vis-select", "value"),
        Input("vis-select-eng", "value"),
        Input("vis-select-pred", "value"),
    ],
    prevent_initial_call=True
)
def store_visual_choice(audio_choice, eng_choice, pred_choice):
    if audio_choice:
        return audio_choice
    if eng_choice:
        return eng_choice
    if pred_choice:
        return pred_choice
    return "heatmap"


@app.callback(
    Output("vis-frame", "src"),
    Input("selected-visual", "data")
)
def update_visual_iframe(choice):
    if choice == "heatmap":
        return "/assets/triangular_heatmap.html"
    if choice == "eng":
        return "/assets/engagement_vs_audio.html"
    if choice == "pca":
        return "/assets/audio_pca.html"
    if choice == "matrix":
        return "/assets/matrix.html"
    if choice == "pred_pop":
        return "/assets/predicted_relationship.html"
    if choice == "pred_mark":
        return "/assets/histogram.html"
    return "/assets/triangular_heatmap.html"


@app.callback(
    Output("vis-caption", "children"),
    Input("selected-visual", "data")
)
def update_caption(choice):
    captions = {
        "heatmap": "This triangular acoustic heatmap displays correlations among audio features such as energy, danceability, valence, and tempo.",
        "eng": "This plot shows how different audio features relate to engagement metrics like engagement rate.",
        "pca": "This 2D audio map uses audio embeddings to group similar songs. The dropdown shows how popular or marketable a song is.",
        "matrix": "This scatter matrix plots pairwise relationships between key audio features to highlight linear and nonlinear trends.",
        "pred_pop": "This scatter plot shows how closely the model’s predicted values align with the actual values.",
        "pred_mark": "This histogram shows the distributions of actual and model-predicted scores for marketability and popularity.",
    }
    return captions.get(choice, "")
# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)
