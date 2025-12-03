import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table

# =========================================================
# LOAD & CLEAN DATA
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

# Dropdown options
artist_options = [
    {"label": artist, "value": artist}
    for artist in sorted(clean_data["Artist"].dropna().unique())
]

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
		dcc.Tab(label="Introduction", value="intro"),
		dcc.Tab(label="Summary Statistics", value="summary"),
		dcc.Tab(label="Deep Learning Model", value="model"),
		dcc.Tab(label="Visuals", value="visuals"),	
		dcc.Tab(label="Final Report Summary", value="report"),
		dcc.Tab(label="Team & Acknowledgments", value="team),




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
    )    # ------------------------------------------------------
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
    # MODEL TAB
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
    # FINAL REPORT SUMMARY TAB (UPDATED)
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

            html.H3("Abstract", style={"color": "#0b2f59"}),
            html.P(
                """
                This project analyzes a combined Spotify–YouTube dataset to identify musical and engagement 
                features that best predict song popularity. By integrating audio attributes such as tempo, 
                energy, and loudness with listener engagement metrics, we conduct both exploratory analysis 
                and predictive modeling. While several features show meaningful associations with popularity, 
                the results demonstrate that audio characteristics alone provide limited predictive power, 
                highlighting the role of external social and platform-driven factors.
                """
            ),

            html.H3("Motivation", style={"color": "#0b2f59"}),
            html.P(
                """
                With the rapid growth of streaming platforms, understanding what drives song popularity 
                has become increasingly important. Millions of tracks are released each year, yet only 
                a small fraction achieve widespread success. This project seeks to identify which 
                quantifiable song-level characteristics contribute to popularity and how predictive 
                modeling can support decision-making in the music industry.
                """
            ),

            html.H3("Methodology", style={"color": "#0b2f59"}),
            html.P(
                """
                We merged Spotify audio features with YouTube engagement data and conducted extensive data 
                cleaning and preprocessing. Our analysis included exploratory visualization, regression 
                modeling, regularization techniques, similarity analysis, and deep learning models using 
                learned audio embeddings. Model performance was evaluated through cross-validation to 
                assess generalizability.
                """
            ),

            html.H3("Key Findings", style={"color": "#0b2f59"}),
            html.P(
                """
                Tracks with consistent tempo, moderate loudness, higher energy, and specific stylistic 
                characteristics tend to exhibit greater engagement. However, even advanced audio-based 
                models show limited predictive accuracy, indicating that popularity is strongly influenced 
                by non-audio factors such as artist recognition, marketing, and algorithmic promotion.
                """
            ),

            html.Br(),

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
# RUN APP
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)


