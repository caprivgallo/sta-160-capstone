import base64
import io
import os
import tempfile
import math
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import librosa
import requests
from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context
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

# Dropdown options for MODEL tab (albums) – only if column exists
if "Album" in model_data.columns:
    model_album_options = [
        {"label": album, "value": album}
        for album in sorted(model_data["Album"].dropna().unique())
    ]
else:
    model_album_options = []

# =========================================================
# DASH APP SETUP
# =========================================================

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # Required for Render
app.title = "The Science of Song Success"

# =========================================================
# MODEL LOADING (PyTorch)
# =========================================================

MODEL_PATH = "best_model_robust.pt"
DEVICE = "cpu"
checkpoint = None
target_frames = 768
target_names = []
target_stats = {}
XGB_MODEL_POP_PATH = "xgb_popularity.pkl"
XGB_MODEL_MARKET_PATH = "xgb_marketability.pkl"
XGB_SCALER_PATH = "xgb_scaler.pkl"
xgb_pop_model = None
xgb_market_model = None
xgb_scaler = None
GENIUS_TOKEN = os.environ.get("GENIUS_API_TOKEN")
SR = 16000
N_FFT = 512
HOP = 160
WIN = 400
N_MELS = 128
FMIN = 20
FMAX = 8000
POWER = 2.0
EPS = 1e-10

try:
    # Try TorchScript first
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
except Exception:
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict):
            target_names = checkpoint.get("targets") or []
            target_stats = checkpoint.get("target_stats") or {}
            cfg_frames = checkpoint.get("config", {}).get("frames")
            if cfg_frames:
                target_frames = int(cfg_frames)
        if hasattr(checkpoint, "eval") and not isinstance(checkpoint, dict):
            model = checkpoint
            model.eval()
        else:
            model = None
    except Exception as e:  # pragma: no cover - startup guard
        model = None
        model_load_error = str(e)
    else:
        model_load_error = None
else:
    model_load_error = None

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


# Model architecture (from training)
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=(3, 7), s=(1, 1), p=None):
        super().__init__()
        p = p or (k[0] // 2, k[1] // 2)
        self.conv = nn.Conv2d(cin, cout, k, s, p)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels, k=(3, 3)):
        super().__init__()
        p = (k[0] // 2, k[1] // 2)
        self.conv1 = nn.Conv2d(channels, channels, k, padding=p)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, k, padding=p)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.SiLU()

    def forward(self, x):
        identity = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.act2(out)


class MultiTaskModel_v2_with_residuals(nn.Module):
    def __init__(self, n_targets):
        super().__init__()
        C = [32, 64, 128, 256, 512]
        self.stem = ConvBlock(1, C[0], (3, 7))
        self.res_stem = ResidualBlock(C[0], (3, 3))
        self.b1 = ConvBlock(C[0], C[1], (3, 5))
        self.res1 = ResidualBlock(C[1], (3, 3))
        self.b2 = ConvBlock(C[1], C[2], (3, 5))
        self.res2 = ResidualBlock(C[2], (3, 3))
        self.b3 = ConvBlock(C[2], C[3], (3, 3))
        self.res3 = ResidualBlock(C[3], (3, 3))
        self.b4 = ConvBlock(C[3], C[4], (3, 3))
        self.res4 = ResidualBlock(C[4], (3, 3))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C[4], 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_targets),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_stem(x)
        x = self.b1(x)
        x = self.res1(x)
        x = self.b2(x)
        x = self.res2(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.b3(x)
        x = self.res3(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.b4(x)
        x = self.res4(x)
        x = self.pool(x)
        return self.head(x)


# If checkpoint has a state dict, build the model now
if model is None and isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    if target_names:
        try:
            model = MultiTaskModel_v2_with_residuals(n_targets=len(target_names))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            model_load_error = None
        except Exception as e:  # pragma: no cover
            model_load_error = f"Failed to load state_dict into model: {e}"
    else:
        model_load_error = "Checkpoint has state_dict but no targets list to size the model."


def preprocess_features(df):
    """
    Basic preprocessing placeholder.
    Replace/extend with the exact steps used during training.
    """
    cols = ["tempo", "energy", "loudness", "duration_ms"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    return torch.tensor(df[cols].values, dtype=torch.float32, device=DEVICE)


def prepare_logmel_tensor(raw_bytes, target_frames=768, n_mels=128):
    """
    Load an NPZ log-mel file from bytes and return a tensor of shape [1, 1, n_mels, target_frames].
    Crops/pads to target_frames and applies winsorization + per-file standardization
    similar to the training pipeline.
    """
    np_obj = np.load(io.BytesIO(raw_bytes))
    if isinstance(np_obj, np.lib.npyio.NpzFile):
        if "logmel" in np_obj.files:
            arr = np_obj["logmel"]
        elif np_obj.files:
            arr = np_obj[np_obj.files[0]]
        else:
            raise ValueError("NPZ file is empty.")
    else:
        arr = np_obj
    if arr.ndim != 2:
        raise ValueError(f"Expected logmel 2D array, got shape {arr.shape}")
    if arr.shape[0] != n_mels:
        raise ValueError(f"Expected {n_mels} mel bins, got {arr.shape[0]}")

    # time axis is arr.shape[1]; crop/pad to target_frames
    T = arr.shape[1]
    if T >= target_frames:
        start = (T - target_frames) // 2
        arr = arr[:, start:start + target_frames]
    else:
        pad_before = (target_frames - T) // 2
        pad_after = target_frames - T - pad_before
        arr = np.pad(arr, ((0, 0), (pad_before, pad_after)), mode="constant")

    # winsorize and standardize per sample
    lo, hi = np.percentile(arr, [0.5, 99.5]).astype(np.float32)
    arr = np.clip(arr, lo, hi)
    m = float(np.mean(arr, dtype=np.float64))
    s = float(np.std(arr, dtype=np.float64))
    if not np.isfinite(s) or s < 1e-6:
        raise ValueError("Invalid std in logmel array.")
    arr = (arr - m) / s
    arr = np.clip(arr, -10, 10)

    tensor = torch.tensor(arr, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
    return tensor


def logmel_from_audio_bytes(file_bytes, target_frames=768):
    """
    Decode audio bytes (mp3/mp4/wav) -> log-mel -> standardized tensor [1,1,n_mels,target_frames].
    Mirrors the training pipeline params.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=SR, mono=True)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if y is None or y.size == 0:
        raise ValueError("Failed to decode audio.")

    S = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=POWER,
    )
    logS = np.log10(np.maximum(S, EPS)).astype(np.float32)

    # crop/pad to target_frames
    T = logS.shape[1]
    if T >= target_frames:
        start = (T - target_frames) // 2
        logS = logS[:, start:start + target_frames]
    else:
        pad_before = (target_frames - T) // 2
        pad_after = target_frames - T - pad_before
        logS = np.pad(logS, ((0, 0), (pad_before, pad_after)), mode="constant")

    # winsorize + standardize
    lo, hi = np.percentile(logS, [0.5, 99.5]).astype(np.float32)
    logS = np.clip(logS, lo, hi)
    m = float(np.mean(logS, dtype=np.float64))
    s = float(np.std(logS, dtype=np.float64))
    if not np.isfinite(s) or s < 1e-6:
        raise ValueError("Invalid std in logmel array.")
    logS = (logS - m) / s
    logS = np.clip(logS, -10, 10)

    return torch.tensor(logS, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)


def denormalize_predictions(pred_array):
    """Apply target mean/std from checkpoint if shapes align."""
    if not target_names or not target_stats:
        return pred_array
    if pred_array.shape[-1] != len(target_names):
        return pred_array
    pred = np.array(pred_array, dtype=float)
    for i, name in enumerate(target_names):
        stats = target_stats.get(name)
        if stats and "mean" in stats and "std" in stats:
            pred[..., i] = pred[..., i] * stats["std"] + stats["mean"]
    return pred


def expm1_for_logs(pred_array):
    """
    Convert *_log targets back to real scale using expm1.
    Returns a dict mapping target name -> array of converted values when applicable.
    """
    if not target_names:
        return {}
    pred_array = np.array(pred_array, dtype=float)
    outputs = {}
    for i, name in enumerate(target_names):
        if name.endswith("_log"):
            base = name.replace("_log", "")
            outputs[base] = np.expm1(pred_array[..., i])
    return outputs


def clamp_logs(pred_flat):
    """
    Clamp *_log predictions to mean ± 2.5 std using target_stats.
    Mutates and returns the array.
    """
    if not target_stats:
        return pred_flat
    for i, name in enumerate(target_names):
        if name.endswith("_log") and name in target_stats:
            mu = target_stats[name].get("mean")
            sd = target_stats[name].get("std")
            if mu is not None and sd is not None:
                pred_flat[i] = np.clip(pred_flat[i], mu - 2.5 * sd, mu + 2.5 * sd)
    return pred_flat


def normal_percentile(z):
    """Approximate percentile from z-score using error function."""
    return 50 * (1 + math.erf(z / math.sqrt(2)))


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
            value="summary",
            colors={"border": "#0b2f59", "primary": "#0b2f59", "background": "#faf7f7"},
            children=[
                dcc.Tab(label="Summary Statistics", value="summary"),
                dcc.Tab(label="Deep Learning Model", value="model"),
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
    # MODEL TAB  (OVERVIEW + DROPDOWNS + TABLE OF PREDICTIONS)
    # ------------------------------------------------------
    elif selected == "model":
        return html.Div(
            [
                html.H2("Deep Learning Model", style={"color": "#0b2f59"}),

                html.Label("Choose model engine:", style={"fontWeight": "bold"}),
                dcc.RadioItems(
                    id="model-engine",
                    options=[
                        {"label": "Audio CNN (log-mel, NPZ/audio)", "value": "cnn"},
                        {"label": "XGBoost (precomputed table)", "value": "xgb"},
                    ],
                    value="cnn",
                    labelStyle={"display": "block", "marginBottom": "4px"},
                ),
                html.Br(),

                html.Div(
                    id="cnn-section",
                    children=[
                        html.P(
                            """
                            Provide feature values to generate predictions, or upload a CSV/NPZ/audio for batch
                            scoring with the CNN. Features should match the training schema used for this model.
                            """,
                            style={"fontSize": "17px"},
                        ),
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                                "gap": "12px",
                                "marginTop": "10px",
                            },
                            children=[
                                html.Div(
                                    [
                                        html.Label("Tempo (BPM)"),
                                        dcc.Slider(
                                            id="tempo",
                                            min=50,
                                            max=220,
                                            step=1,
                                            value=120,
                                            marks=None,
                                            tooltip={"placement": "bottom"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Energy (0-1)"),
                                        dcc.Slider(
                                            id="energy",
                                            min=0,
                                            max=1,
                                            step=0.01,
                                            value=0.6,
                                            marks=None,
                                            tooltip={"placement": "bottom"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Loudness (dB)"),
                                        dcc.Slider(
                                            id="loudness",
                                            min=-40,
                                            max=5,
                                            step=0.5,
                                            value=-8,
                                            marks=None,
                                            tooltip={"placement": "bottom"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("Duration (ms)"),
                                        dcc.Slider(
                                            id="duration_ms",
                                            min=60000,
                                            max=480000,
                                            step=5000,
                                            value=180000,
                                            marks=None,
                                            tooltip={"placement": "bottom"},
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        html.Button(
                            "Predict Single",
                            id="predict-btn",
                            n_clicks=0,
                            style={
                                "marginTop": "12px",
                                "backgroundColor": "#0b2f59",
                                "color": "white",
                                "padding": "10px 18px",
                                "borderRadius": "8px",
                                "border": "none",
                                "cursor": "pointer",
                            },
                        ),
                        html.Div(id="single-output", style={"marginTop": "12px", "fontWeight": "bold"}),

                        html.Hr(),
                        html.H3("Batch Prediction", style={"color": "#0b2f59"}),
                        html.P(
                            "Upload a CSV (tempo, energy, loudness, duration_ms), an NPZ log-mel file, or audio.",
                            style={"fontSize": "15px"},
                        ),
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div(
                                ["Drag and drop CSV/NPZ here, or ", html.A("select a file", style={"color": "#0b2f59"})]
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
                        html.Div(id="batch-output"),

                        html.Br(),
                        html.H3("Upload Audio (mp3/mp4/wav)", style={"color": "#0b2f59"}),
                        html.P("We will compute log-mel features on the server and run the CNN.", style={"fontSize": "15px"}),
                        dcc.Upload(
                            id="upload-audio",
                            children=html.Div(
                                ["Drag and drop audio here, or ", html.A("select a file", style={"color": "#0b2f59"})]
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
                        html.Div(id="audio-output"),

                        html.Br(),
                        html.Div(
                            [
                                html.H4("Notes", style={"color": "#0b2f59"}),
                                html.Ul(
                                    [
                                        html.Li("CNN model best_model_robust.pt is loaded once at startup (CPU)."),
                                        html.Li("CSV expects columns: tempo, energy, loudness, duration_ms; NPZ uploads should contain the feature matrix as the first array."),
                                        html.Li("If predictions fail, a descriptive error will appear below. Checkpoint with only state_dict requires the model class (e.g., MultiTaskModel_v2_with_residuals) to be defined in code."),
                                    ]
                                ),
                            ]
                        ),
                    ],
                ),

                html.Div(
                    id="xgb-section",
                    children=[
                        html.P(
                            "View the XGBoost marketability/popularity predictions (precomputed) with filters.",
                            style={"fontSize": "17px"},
                        ),
                        html.P(
                            "Upload a CSV containing the same numeric feature columns as merged_no_embeddings.csv to run XGB locally "
                            "if xgb_popularity.pkl/xgb_marketability.pkl/xgb_scaler.pkl are present.",
                            style={"fontSize": "14px", "color": "#444"},
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
                    ],
                ),

                html.Br(),
                html.H3("Explore XGBoost Predictions", style={"color": "#0b2f59"}),
                html.P( 
                    """
                     What do these scores mean?

                     • Predicted Popularity is a 0–100 score that approximates Spotify’s track popularity, 
                     where higher values indicate songs that look more like widely streamed hits.

                     • Predicted Marketability is a 0–100 index we designed that combines streaming popularity, 
                        YouTube views and likes, engagement rate, and a small contribution from energy and danceability. 
                        Higher values indicate songs that appear more attractive from a commercial/marketing standpoint.

                     Use the filters bleow or click through the drop menu below to explore our predicted scores across 5000+ songs.
                    """,
                    style={"fontSize": "17px", "whiteSpace": "pre-line"},
                ),
                html.Br(),

                html.Label("Filter by Artist:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="model-artist-dropdown",
                    options=model_artist_options,
                    placeholder="Select an artist...",
                ),
                html.Br(),

                html.Label("Filter by Album:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="model-album-dropdown",
                    options=model_album_options,
                    placeholder="Select an album...",
                ) if len(model_album_options) > 0 else html.Div(
                    "(Album metadata not available in this dataset.)",
                    style={"fontStyle": "italic", "marginBottom": "10px"},
                ),
                html.Br(),

                html.Label("Search Songs or Albums:", style={"fontWeight": "bold"}),
                dcc.Input(
                    id="model-search-input",
                    type="text",
                    placeholder="Type song, album, or keyword...",
                    style={"width": "100%", "padding": "10px"},
                ),
                html.Br(), html.Br(),

                dash_table.DataTable(
                    id="model-table",
                    columns=[
                        {"name": "Artist", "id": "Artist"},
                        {"name": "Track", "id": "Track"},
                        {"name": "Album", "id": "Album"},
                        {"name": "Predicted Popularity", "id": "Predicted_Popularity"},
                        {
                            "name": "Predicted Marketability",
                            "id": "Predicted_Marketability",
                        },
                    ],
                    data=model_data.to_dict("records"),
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
            [
                html.H2("Final Report Summary", style={"color": "#0b2f59"}),

                # ABSTRACT
                html.H3("Abstract", style={"color": "#0b2f59"}),
                html.P(
                    """
                    Our project analyzes a combined Spotify–YouTube dataset to identify the musical 
                    and engagement features that best predict song popularity. The dataset integrates 
                    Spotify audio attributes (such as danceability, energy, tempo, and valence) with 
                    YouTube metrics (views, likes, comments), enabling both exploratory and predictive modeling. 
                    Our final models demonstrate strong predictive power, and the dashboard provides 
                    actionable insights for artists, producers, and record labels.
                    """,
                    style={"fontSize": "16px"},
                ),

                # MOTIVATION
                html.H3("Motivation", style={"color": "#0b2f59"}),
                html.P(
                    """
                    In an industry driven by streaming algorithms, understanding the components of 
                    song popularity is essential. By integrating musical structure with listener 
                    behavior, we aimed to identify what characteristics help songs succeed across 
                    major streaming platforms. Our core research question was:

                    “What song-level features best predict popularity, and how can predictive modeling 
                    support decision-making in the music industry?”
                    """,
                    style={"fontSize": "16px", "whiteSpace": "pre-line"},
                ),

                # METHODOLOGY
                html.H3("Methodology", style={"color": "#0b2f59"}),
                html.P(
                    """
                    We combined Spotify audio features with YouTube engagement data and performed:

                    • Extensive data cleaning  
                    • Exploratory data analysis  
                    • Regression, LASSO, and model selection  
                    • Deep learning model development  
                    • Feature engineering (interactions, embeddings, scaled variables)  
                    • Similarity and clustering analysis  

                    This framework ensured interpretability, predictive accuracy, and scalability.
                    """,
                    style={"fontSize": "16px", "whiteSpace": "pre-line"},
                ),

                # RESULTS
                html.H3("Results", style={"color": "#0b2f59"}),
                html.P(
                    """
                    Two major predictive models were developed:

                    • Popularity Model → R² ≈ 0.656, MAE ≈ 5.02  
                    • Marketability Model → R² ≈ 0.698, MAE ≈ 2.02  

                    Engagement metrics produced the strongest signals, but audio features—especially 
                    energy, tempo, and danceability—were still significant contributors.

                    The hybrid similarity matrix revealed structured clusters of songs that share both 
                    acoustic and engagement characteristics.
                    """,
                    style={"fontSize": "16px", "whiteSpace": "pre-line"},
                ),

                # INTERPRETATION
                html.H3("Interpretation", style={"color": "#0b2f59"}),
                html.P(
                    """
                    Engagement signals dominated prediction, demonstrating the power of consumer behavior 
                    in shaping music success. However, audio features added valuable nuance, showing that 
                    musical structure influences how listeners interact with songs.

                    Songs with consistent rhythm, moderate loudness, and high energy performed noticeably 
                    better across metrics.
                    """,
                    style={"fontSize": "16px", "whiteSpace": "pre-line"},
                ),

                # CONCLUSION
                html.H3("Conclusion", style={"color": "#0b2f59"}),
                html.P(
                    """
                    Our modeling framework effectively captures both musical characteristics and listener 
                    behavior. The models we developed provide practical insights for forecasting song 
                    performance and strengthening data-driven decision-making in the modern music industry.

                    A full version of the report—including extended methodology, visualizations, and 
                    discussion—is available below.
                    """,
                    style={"fontSize": "16px", "whiteSpace": "pre-line"},
                ),

                html.Br(),

                # FULL REPORT LINK
                html.Div(
                    html.A(
                        "📄 View Full Final Report (Google Doc)",
                        href="https://docs.google.com/document/d/1wYoX1HOKK6wEcU_JRXqQMJ0NLWfyZAwa5gW9crADFms/edit?usp=sharing",
                        target="_blank",
                        style={
                            "backgroundColor": "#0b2f59",
                            "color": "white",
                            "padding": "12px 20px",
                            "borderRadius": "8px",
                            "textDecoration": "none",
                            "fontWeight": "bold",
                        },
                    ),
                    style={"textAlign": "center"},
                ),
            ]
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
    Output("single-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("tempo", "value"),
    State("energy", "value"),
    State("loudness", "value"),
    State("duration_ms", "value"),
    prevent_initial_call=True,
)
def predict_single(n_clicks, tempo, energy, loudness, duration_ms):
    if model is None:
        return f"Model not loaded: {model_load_error}"
    if target_names:
        return "This model expects log-mel NPZ input. Use the batch upload below to score an NPZ file."

    try:
        df = pd.DataFrame(
            [
                {
                    "tempo": tempo,
                    "energy": energy,
                    "loudness": loudness,
                    "duration_ms": duration_ms,
                }
            ]
        )
        features = preprocess_features(df)
        with torch.no_grad():
            pred = model(features).cpu().numpy()
        pred_vals = pred.squeeze()
        if pred_vals.shape == ():
            return f"Prediction: {float(pred_vals):.2f}"
        return html.Div(
            [
                html.Div("Prediction:", style={"marginBottom": "6px"}),
                html.Ul([html.Li(f"{i + 1}: {v:.2f}") for i, v in enumerate(pred_vals.tolist())]),
            ]
        )
    except Exception as e:
        return f"Error generating prediction: {e}"


@app.callback(
    Output("batch-output", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def predict_batch(contents, filename):
    if contents is None:
        return "No file uploaded."
    if model is None:
        return f"Model not loaded: {model_load_error}"

    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        is_npz = filename.lower().endswith(".npz") if filename else False

        if is_npz:
            features_tensor = prepare_logmel_tensor(
                decoded, target_frames=target_frames, n_mels=128
            )
            with torch.no_grad():
                preds = model(features_tensor).cpu().numpy()
            preds = denormalize_predictions(preds)
            preds_flat = preds.squeeze()

            # For NPZ we show predictions with target names when available
            if preds_flat.shape == ():
                data_rows = [{"prediction": float(preds_flat)}]
            elif preds_flat.ndim == 0:
                data_rows = [{"prediction": float(preds_flat)}]
            elif preds_flat.ndim == 1:
                preds_flat = clamp_logs(preds_flat)
                row = {}
                for i, v in enumerate(preds_flat.tolist()):
                    name = target_names[i] if i < len(target_names) else f"prediction_{i + 1}"
                    row[name] = float(v)
                    if name.endswith("_log"):
                        base = name.replace("_log", "")
                        row[base] = float(np.expm1(v))
                        stats = target_stats.get(name, {})
                        mu, sd = stats.get("mean"), stats.get("std")
                        if mu is not None and sd is not None and sd > 0:
                            z = (v - mu) / sd
                            row[f"{base}_percentile"] = round(normal_percentile(z), 1)
                data_rows = [row]
            else:
                data_rows = []
                for row_vals in preds:
                    row_vals = clamp_logs(np.array(row_vals))
                    row = {}
                    for i, v in enumerate(row_vals.tolist()):
                        name = target_names[i] if i < len(target_names) else f"prediction_{i + 1}"
                        row[name] = float(v)
                        if name.endswith("_log"):
                            base = name.replace("_log", "")
                            row[base] = float(np.expm1(v))
                            stats = target_stats.get(name, {})
                            mu, sd = stats.get("mean"), stats.get("std")
                            if mu is not None and sd is not None and sd > 0:
                                z = (v - mu) / sd
                                row[f"{base}_percentile"] = round(normal_percentile(z), 1)
                    data_rows.append(row)
            return dash_table.DataTable(
                data=data_rows[:200],
                page_size=10,
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#0b2f59",
                    "color": "white",
                    "fontWeight": "bold",
                },
            )
        else:
            df = pd.read_csv(io.BytesIO(decoded))
            features = preprocess_features(df)
            with torch.no_grad():
                preds = model(features).cpu().numpy()
            preds = denormalize_predictions(preds)
            preds_flat = preds.squeeze()

            df_out = df.copy()
            if preds_flat.shape == ():
                df_out["prediction"] = float(preds_flat)
            elif preds_flat.ndim == 0:
                df_out["prediction"] = float(preds_flat)
            elif preds_flat.ndim == 1:
                preds_flat = clamp_logs(preds_flat)
                for i, v in enumerate(preds_flat.tolist()):
                    name = target_names[i] if i < len(target_names) else f"prediction_{i + 1}"
                    df_out[name] = float(v)
                    if name.endswith("_log"):
                        base = name.replace("_log", "")
                        df_out[base] = float(np.expm1(v))
                        stats = target_stats.get(name, {})
                        mu, sd = stats.get("mean"), stats.get("std")
                        if mu is not None and sd is not None and sd > 0:
                            z = (v - mu) / sd
                            df_out[f"{base}_percentile"] = round(normal_percentile(z), 1)
            else:
                for i in range(preds.shape[1]):
                    name = target_names[i] if i < len(target_names) else f"prediction_{i + 1}"
                    col_vals = clamp_logs(preds[:, i])
                    df_out[name] = col_vals
                    if name.endswith("_log"):
                        base = name.replace("_log", "")
                        df_out[base] = np.expm1(col_vals)
                        stats = target_stats.get(name, {})
                        mu, sd = stats.get("mean"), stats.get("std")
                        if mu is not None and sd is not None and sd > 0:
                            z = (col_vals - mu) / sd
                            df_out[f"{base}_percentile"] = normal_percentile(z)

            return dash_table.DataTable(
                data=df_out.head(200).to_dict("records"),
                page_size=10,
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#0b2f59",
                    "color": "white",
                    "fontWeight": "bold",
                },
            )
    except Exception as e:
        return f"Error processing {filename}: {e}"


@app.callback(
    Output("audio-output", "children"),
    Input("upload-audio", "contents"),
    State("upload-audio", "filename"),
    prevent_initial_call=True,
)
def predict_audio(contents, filename):
    if contents is None:
        return "No audio file uploaded."
    if model is None:
        return f"Model not loaded: {model_load_error}"
    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        features_tensor = logmel_from_audio_bytes(decoded, target_frames=target_frames)
        with torch.no_grad():
            preds = model(features_tensor).cpu().numpy()
        preds = denormalize_predictions(preds)
        preds_flat = preds.squeeze()

        pred_row = {}
        if preds_flat.ndim == 0:
            pred_row["prediction"] = float(preds_flat)
        elif preds_flat.ndim == 1:
            preds_flat = clamp_logs(preds_flat)
            for i, v in enumerate(preds_flat.tolist()):
                name = target_names[i] if i < len(target_names) else f"prediction_{i + 1}"
                pred_row[name] = float(v)
                if name.endswith("_log"):
                    base = name.replace("_log", "")
                    pred_row[base] = float(np.expm1(v))
                    stats = target_stats.get(name, {})
                    mu, sd = stats.get("mean"), stats.get("std")
                    if mu is not None and sd is not None and sd > 0:
                        z = (v - mu) / sd
                        pred_row[f"{base}_percentile"] = round(normal_percentile(z), 1)

        rec_table = None
        if rec_matrix is not None and target_names:
            sim_cols = []
            sim_vec = []
            for col in ["Views", "Likes", "Comments"]:
                log_name = f"{col.lower()}_log"
                if log_name in target_names:
                    idx = target_names.index(log_name)
                    sim_cols.append(col)
                    sim_vec.append(preds_flat[idx])
            if sim_vec and len(sim_vec) == rec_matrix.shape[1]:
                sim_vec = np.array(sim_vec, dtype=float)
                rec_norm = np.linalg.norm(rec_matrix, axis=1, keepdims=True) + 1e-8
                vec_norm = np.linalg.norm(sim_vec) + 1e-8
                sims = (rec_matrix @ sim_vec) / (rec_norm.flatten() * vec_norm)
                top_idx = np.argsort(sims)[::-1][:5]
                rows = []
                for i in top_idx:
                    row = {"score": float(sims[i])}
                    if rec_meta is not None:
                        for c in rec_meta.columns:
                            row[c] = rec_meta.iloc[i][c]
                    for c in rec_feature_cols:
                        row[c] = float(clean_data.iloc[i][c])
                    rows.append(row)
                rec_table = dash_table.DataTable(
                    data=rows,
                    page_size=5,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#0b2f59",
                        "color": "white",
                        "fontWeight": "bold",
                    },
                )

        return html.Div(
            [
                html.H4(f"Prediction from {filename}", style={"color": "#0b2f59"}),
                dash_table.DataTable(
                    data=[pred_row],
                    page_size=1,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#0b2f59",
                        "color": "white",
                        "fontWeight": "bold",
                    },
                ),
                html.Br(),
                html.H4("Similar songs (by engagement profile)", style={"color": "#0b2f59"}),
                rec_table or html.Div("Not enough data for recommendations."),
            ]
        )
    except Exception as e:
        return f"Error processing {filename}: {e}"


@app.callback(
    [Output("cnn-section", "style"), Output("xgb-section", "style")],
    Input("model-engine", "value"),
)
def toggle_model_sections(engine):
    show = {"display": "block"}
    hide = {"display": "none"}
    if engine == "xgb":
        return hide, show
    return show, hide


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
