import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table

# ============================
# LOAD + CLEAN DATA
# ============================
data = pd.read_csv("merged_final_data.csv")

# tempo cleanup
data["tempo"] = (
    data["tempo"]
    .astype(str)
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .astype(float)
)

# numeric fields
data["energy"] = pd.to_numeric(data["energy"], errors="coerce")

# convert duration_ms → minutes
data["duration_min"] = (data["duration_ms"] / 60000).round(2)

# remove missing
clean_data = data.dropna(subset=["tempo", "energy"])

# dropdown artists
artist_options = [
    {"label": artist, "value": artist}
    for artist in sorted(clean_data["Artist"].dropna().unique())
]

# ============================
# DASH APP
# ============================
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server    # <-- REQUIRED FOR RENDER
app.title = "The Science of Song Success"

# ============================
# LAYOUT
# ============================
app.layout = html.Div(
    style={
        "backgroundColor": "#faf7f7",
        "fontFamily": "Georgia, serif",
        "padding": "25px",
        "color": "#1a1a1a",
    },
    children=[
        # HEADER
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

        # ------------------ TABS ------------------
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

        # FOOTER
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

# ============================
# TAB CALLBACK
# ============================
@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value")
)
def render_tab(selected):

    # ------------------------------------------
    # SUMMARY STATISTICS TAB
    # ------------------------------------------
    if selected == "summary":
        return html.Div(
            [
                html.H2("Summary Statistics", style={"color": "#0b2f59"}),
                html.P(
                    """
                    Below are descriptive summaries of key Spotify and YouTube variables:
                     • tempo — speed of a track in beats per minute (BPM)
                     • energy — measure of intensity (0 to 1)
                     • loudness — volume of track in decibels (dB)
                     • duration_min — length of track in minutes
                     • views, likes, comments — listener engagement on YouTube  
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
                    placeholder="Type song or album name...",
                    style={"width": "100%", "padding": "10px"},
                ),
                html.Br(),
                html.Br(),

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

    # ------------------------------------------
    # MODEL TAB
    # ------------------------------------------
    elif selected == "model":
        return html.Div(
            [
                html.H2("Deep Learning Model Overview", style={"color": "#0b2f59"}),
                html.P(
                    """
                    This section summarizes the architecture and training results of our deep 
                    learning model. It analyzes how well the model predicts popularity metrics 
                    based on audio features and discusses opportunities for improvement.
                    """,
                    style={"fontSize": "17px"},
                ),
            ]
        )

    # ------------------------------------------
    # VISUALS TAB
    # ------------------------------------------
    elif selected == "visuals":
        return html.Div(
            [
                html.H2("Visualizations", style={"color": "#0b2f59"}),
                html.P(
                    "Interactive plots will appear here.",
                    style={"fontSize": "17px"},
                ),
            ]
        )

    # ------------------------------------------
    # FINAL REPORT SUMMARY TAB
    # ------------------------------------------
    elif selected == "report":
        return html.Div(
            [
                html.H2("Final Report Summary", style={"color": "#0b2f59"}),
                html.P(
                    """
                    This section contains a summary of our full written report, including 
                    motivation, methodology, modeling approach, and key findings.
                    """,
                    style={"fontSize": "17px"},
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

    # ------------------------------------------
    # TEAM & REFERENCES TAB
    # ------------------------------------------
    elif selected == "team":
        return html.Div(
            [
                html.H2("Team & Acknowledgments", style={"color": "#0b2f59"}),

                html.Div(
                    style={"display": "flex", "flexWrap": "wrap"},
                    children=[
                        html.Div(
                            [
                                html.H4("Capri Gallo"),
                                html.P("B.S. Statistical Data Science, UC Davis (2026)"),
                            ],
                            style=card(),
                        ),
                        html.Div(
                            [
                                html.H4("Alex Garcia"),
                                html.P("B.S. Statistical Data Science, UC Davis (2026)"),
                            ],
                            style=card(),
                        ),
                        html.Div(
                            [
                                html.H4("Rohan Pillay"),
                                html.P("B.S. Statistical Data Science, UC Davis (2026)"),
                            ],
                            style=card(),
                        ),
                        html.Div(
                            [
                                html.H4("Edward Ron"),
                                html.P("B.S. Statistical Data Science, UC Davis (2026)"),
                            ],
                            style=card(),
                        ),
                        html.Div(
                            [
                                html.H4("Yuxiao Tan"),
                                html.P("B.S. Statistical Data Science, UC Davis (2026)"),
                            ],
                            style=card(),
                        ),
                    ],
                ),

                html.Br(),
                html.H3("Acknowledgments", style={"color": "#0b2f59"}),
                html.P(
                    "Special thanks to Professor Lingfei Cui and the UC Davis Statistics Department for their guidance."
                ),

                html.Br(),
                html.H3("References", style={"color": "#0b2f59"}),
                html.Ul(
                    [
                        html.Li(
                            "Li, Y. et al. (2024). MERT: Acoustic Music Understanding Model with "
                            "Large-Scale Self-Supervised Training. arXiv:2306.00107."
                        ),
                        html.Li(
                            "Grewal, R. (2025). Spotify–YouTube Data. Kaggle. "
                            "https://www.kaggle.com/datasets/rohitgrewal/spotify-youtube-data"
                        ),
                    ]
                ),
            ]
        )

# Helper for team cards
def card():
    return {
        "backgroundColor": "white",
        "padding": "15px",
        "borderRadius": "10px",
        "margin": "10px",
        "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
        "width": "250px",
    }

# ============================
# CALLBACK FOR TABLE FILTERING
# ============================
@app.callback(
    Output("summary-table", "data"),
    [Input("artist-dropdown", "value"), Input("search-input", "value")]
)
def update_table(selected_artist, search_text):
    df = clean_data.copy()

    if selected_artist:
        df = df[df["Artist"] == selected_artist]

    if search_text and search_text.strip():
        search = search_text.lower()
        df = df[df["filename"].str.lower().str.contains(search, na=False)]

    return df.to_dict("records")


# ============================
# RUN APP
# ============================
if __name__ == "__main__":
    app.run(debug=True)
