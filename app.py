import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table

# --- Load and clean data ---
data = pd.read_csv("merged_final_data.csv")
data["tempo"] = (
    data["tempo"]
    .astype(str)
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .astype(float)
)
data["energy"] = pd.to_numeric(data["energy"], errors="coerce")
data["duration_ms"] = data["duration_ms"].apply(lambda x: round(x / 60000, 2))
clean_data = data.dropna(subset=["tempo", "energy"])

artist_options = [{"label": artist, "value": artist} for artist in sorted(clean_data["Artist"].dropna().unique())]

# UC DAVIS COLORS
PRIMARY_COLOR = "#022851"   # UC Davis Navy
ACCENT_COLOR = "#DAAA00"    # UC Davis Gold
BACKGROUND = "#fafafa"

# --- Initialize app ---
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # REQUIRED FOR RENDER DEPLOY
app.title = "The Science of Song Success"

# --- Layout ---
app.layout = html.Div(
    style={
        "backgroundColor": BACKGROUND,
        "fontFamily": "Georgia, serif",
        "padding": "25px",
        "color": PRIMARY_COLOR,
    },
    children=[
        # Header
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "marginBottom": "15px",
            },
            children=[
                html.Img(src="/assets/ucdavis_logo.png", style={"height": "70px"}),
                html.H1(
                    "The Science of Song Success",
                    style={"color": PRIMARY_COLOR, "fontSize": "38px", "fontWeight": "bold", "flex": "1", "textAlign": "center"},
                ),
            ],
        ),

        html.Div(style={"borderBottom": f"3px solid {ACCENT_COLOR}", "width": "260px", "margin": "0 auto", "marginBottom": "20px"}),

        html.P(
            """
            For our capstone project, we built an interactive dashboard to explore how Spotify audio 
            features and YouTube engagement metrics relate to one another — and how these patterns 
            help explain a song’s success. This dashboard includes summary statistics, a deep learning 
            model overview, key visualizations, and a polished final report summary.
            """,
            style={"fontSize": "18px", "textAlign": "center", "lineHeight": "1.7"},
        ),

        html.Br(),

        # Tabs
        dcc.Tabs(
            id="tabs",
            value="summary",
            colors={"border": PRIMARY_COLOR, "primary": PRIMARY_COLOR, "background": BACKGROUND},
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
        html.Hr(),

        # Footer
        html.Div(
            "Developed by Team 13 — STA 160 Capstone | UC Davis, Fall 2025",
            style={"textAlign": "center", "color": PRIMARY_COLOR, "fontSize": "14px", "marginTop": "20px"},
        )
    ],
)

# --- Callback for tabs ---
@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "summary":
        return html.Div(
            [
                html.H2("Summary Statistics", style={"color": PRIMARY_COLOR}),
                html.P(
                    """
                    Below are descriptive summaries of key Spotify and YouTube variables:
                    
                    • tempo — speed of a track in beats per minute (BPM)  
                    • energy — measure of intensity (0 to 1)  
                    • loudness — overall volume of a track (in dB)  
                    • duration_min — track length in minutes  
                    • views, likes, comments — listener engagement on YouTube  
                    """,
                    style={"fontSize": "16px", "whiteSpace": "pre-line"},
                ),
                html.Br(),
                html.Label("Filter by Artist:"),
                dcc.Dropdown(id="artist-dropdown", options=artist_options, placeholder="Select an artist"),
                html.Br(),
                dash_table.DataTable(
                    id="summary-table",
                    columns=[{"name": col, "id": col} for col in ["Artist", "Views", "Likes", "Comments", "tempo", "energy", "duration_ms"]],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": PRIMARY_COLOR, "color": "white", "fontWeight": "bold"},
                    style_data={"backgroundColor": "white", "color": PRIMARY_COLOR, "border": "1px solid #ddd"},
                ),
            ]
        )

    elif tab == "model":
        return html.Div(
            [
                html.H2("Deep Learning Model Overview", style={"color": PRIMARY_COLOR}),
                html.P(
                    """
                    Our deep learning model was trained to predict YouTube popularity metrics using Spotify audio features. 
                    The model incorporates dense layers with dropout regularization, optimizing mean squared error. 
                    Additional experiments involved random forests and regression models for interpretability comparison.
                    """,
                    style={"fontSize": "17px", "lineHeight": "1.7"},
                ),
            ]
        )

    elif tab == "visuals":
        return html.Div(
            [
                html.H2("Visualizations", style={"color": PRIMARY_COLOR}),
                html.P("Interactive plots will appear here in the final version.", style={"fontSize": "17px"}),
            ]
        )

    elif tab == "report":
        return html.Div(
            style={"display": "flex", "gap": "30px"},
            children=[
                # Sidebar
                html.Div(
                    [
                        html.H3("Sections", style={"color": PRIMARY_COLOR}),
                        html.Ul(
                            [
                                html.Li(html.A("1. Motivation", href="#motivation")),
                                html.Li(html.A("2. Data & Processing", href="#data")),
                                html.Li(html.A("3. Modeling", href="#modeling")),
                                html.Li(html.A("4. Key Findings", href="#findings")),
                                html.Li(html.A("5. Conclusions", href="#conclusions")),
                            ],
                            style={"lineHeight": "2"},
                        ),
                    ],
                    style={"width": "22%", "backgroundColor": "#f4f4f4", "padding": "15px", "borderRadius": "8px"},
                ),

                # Report Content
                html.Div(
                    [
                        html.H2("Final Report Summary", style={"color": PRIMARY_COLOR}),
                        html.P("Below is a streamlined summary of our full written report.", style={"fontSize": "16px"}),

                        html.Div(id="motivation", children=[
                            html.H3("1. Motivation", style={"color": PRIMARY_COLOR}),
                            html.P(
                                "We aim to identify measurable audio and engagement patterns that explain why some songs become popular. "
                                "Using Spotify and YouTube together offers a rare multi-platform understanding of musical success.",
                                style={"fontSize": "16px"},
                            ),
                        ]),

                        html.Div(id="data", children=[
                            html.H3("2. Data Collection & Processing", style={"color": PRIMARY_COLOR}),
                            html.P(
                                "Our dataset merges Spotify acoustic features with YouTube engagement statistics for 5,000+ tracks. "
                                "We cleaned inconsistencies, standardized variables, and removed duplicates.",
                                style={"fontSize": "16px"},
                            ),
                        ]),

                        html.Div(id="modeling", children=[
                            html.H3("3. Modeling Approach", style={"color": PRIMARY_COLOR}),
                            html.P(
                                "We evaluated multiple models including deep learning regressors, random forests, and linear models. "
                                "Our primary DNN focused on predicting views and likes with strong performance.",
                                style={"fontSize": "16px"},
                            ),
                        ]),

                        html.Div(id="findings", children=[
                            html.H3("4. Key Findings", style={"color": PRIMARY_COLOR}),
                            html.P(
                                "• Songs with moderate loudness and consistent tempo perform best.\n"
                                "• Energy strongly correlates with likes.\n"
                                "• Danceability trends vary by genre.\n",
                                style={"whiteSpace": "pre-line", "fontSize": "16px"},
                            ),
                        ]),

                        html.Div(id="conclusions", children=[
                            html.H3("5. Conclusions & Future Work", style={"color": PRIMARY_COLOR}),
                            html.P(
                                "Future work includes adding lyric sentiment analysis, artist-level modeling, "
                                "and temporal trends such as release date seasonality.",
                                style={"fontSize": "16px"},
                            ),
                        ]),

                        html.Br(),
                        html.Div(
                            html.A(
                                "📄 Download Full Report (PDF)",
                                href="https://drive.google.com/your-final-report-link",
                                target="_blank",
                                style={
                                    "backgroundColor": PRIMARY_COLOR,
                                    "color": "white",
                                    "padding": "10px 20px",
                                    "borderRadius": "8px",
                                    "textDecoration": "none",
                                    "fontWeight": "bold",
                                },
                            ),
                            style={"textAlign": "center"},
                        ),
                    ],
                    style={"width": "78%", "backgroundColor": "white", "padding": "25px", "borderRadius": "10px",
                           "overflowY": "auto", "maxHeight": "600px",
                           "boxShadow": "0px 2px 6px rgba(0,0,0,0.1)"},
                ),
            ],
        )

    elif tab == "team":
        return html.Div(
            [
                html.H2("Team & Acknowledgments", style={"color": PRIMARY_COLOR}),
                html.Div(
                    [
                        html.Div(
                            [html.H4("Capri Gallo"), html.P("B.S. Statistical Data Science | UC Davis (2026)")],
                            style={"backgroundColor": "white", "padding": "15px", "borderRadius": "10px", "margin": "10px"},
                        ),
                        html.Div(
                            [html.H4("Alex Garcia"), html.P("B.S. Statistical Data Science | UC Davis (2026)")],
                            style={"backgroundColor": "white", "padding": "15px", "borderRadius": "10px", "margin": "10px"},
                        ),
                        html.Div(
                            [html.H4("Rohan Pillay"), html.P("B.S. Statistical Data Science | UC Davis (2026)")],
                            style={"backgroundColor": "white", "padding": "15px", "borderRadius": "10px", "margin": "10px"},
                        ),
                        html.Div(
                            [html.H4("Edward Ron"), html.P("B.S. Statistical Data Science | UC Davis (2026)")],
                            style={"backgroundColor": "white", "padding": "15px", "borderRadius": "10px", "margin": "10px"},
                        ),
                        html.Div(
                            [html.H4("Yuxiao Tan"), html.P("B.S. Statistical Data Science | UC Davis (2026)")],
                            style={"backgroundColor": "white", "padding": "15px", "borderRadius": "10px", "margin": "10px"},
                        ),
                    ],
                    style={"display": "flex", "flexWrap": "wrap"},
                ),

                html.Br(),

                html.H3("References", style={"color": PRIMARY_COLOR}),
                html.Ul(
                    [
                        html.Li("Li, Y. et al. (2024). MERT: Acoustic Music Understanding Model with Large-Scale Self-Supervised Training. arXiv:2306.00107."),
                        html.Li("Grewal, R. (2025). Spotify–YouTube Data. Kaggle. https://www.kaggle.com/datasets/rohitgrewal/spotify-youtube-data"),
                    ]
                ),
            ]
        )


# --- Run app ---
if __name__ == "__main__":
    app.run(debug=True)
