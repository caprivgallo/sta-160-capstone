import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table

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
                html.H2("Deep Learning Model Overview", style={"color": "#0b2f59"}),

                html.P(
                    """
                    This section summarizes the design, evaluation, and performance of our 
                    deep learning model. The model uses Spotify acoustic features—including 
                    danceability, energy, tempo, and valence—to predict YouTube engagement 
                    metrics such as views, likes, and comments.

                    After extensive tuning and feature engineering, our models achieved:

                    • Popularity Model → R² ≈ 0.656, MAE ≈ 5.02  
                    • Marketability Model → R² ≈ 0.698, MAE ≈ 2.02  

                    Engagement-driven models produced the strongest predictive power,
                    reflecting modern music consumption dynamics.
                    """,
                    style={"fontSize": "17px", "whiteSpace": "pre-line"},
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
                html.P(
                    """
                    Interactive plots will appear here, including:

                    • Relationships between acoustic features  
                    • Engagement vs. audio-feature comparisons  
                    • Clustering and similarity structures  
                    """,
                    style={"fontSize": "17px", "whiteSpace": "pre-line"},
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

    # Filter by artist
    if selected_artist:
        df = df[df["Artist"] == selected_artist]

    # Filter by album (only if column exists)
    if selected_album and "Album" in df.columns:
        df = df[df["Album"] == selected_album]

    # Text search across string columns
    if search_text and search_text.strip():
        search = search_text.lower()
        string_cols = df.select_dtypes(include="object").columns
        mask = df[string_cols].apply(
            lambda col: col.str.lower().str.contains(search, na=False)
        )
        df = df[mask.any(axis=1)]

    # Only keep the columns we want to display
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
# RUN APP
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)
