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

# --- Initialize app ---
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "The Science of Song Success"

# --- Layout ---
app.layout = html.Div(
    style={
        "backgroundColor": "#faf7f7",
        "fontFamily": "Georgia, serif",
        "padding": "25px",
        "color": "#222",
    },
    children=[
        # Header section
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
                    style={
                        "color": "#b32020",
                        "fontSize": "36px",
                        "textAlign": "center",
                        "flex": "1",
                        "fontWeight": "bold",
                    },
                ),
            ],
        ),
        html.Div(
            style={
                "borderBottom": "3px solid #b32020",
                "width": "200px",
                "margin": "0 auto",
                "marginBottom": "20px",
            }
        ),
        html.P(
            """
            For our capstone project, we have built an interactive dashboard to visualize our findings. 
            Our dashboard examines how different Spotify audio features and YouTube engagement metrics relate to one another, 
            and how these relationships drive a song's success. Below are several tabs displaying summary statistics, 
            our deep learning model, visual results, and our final report.
            """,
            style={"fontSize": "18px", "textAlign": "center", "lineHeight": "1.7"},
        ),
        html.Br(),

        # Tabs
        dcc.Tabs(
            id="tabs",
            value="summary",
            colors={"border": "#b32020", "primary": "#b32020", "background": "#faf7f7"},
            children=[
                dcc.Tab(label="Summary Statistics", value="summary"),
                dcc.Tab(label="Deep Learning Model", value="model"),
                dcc.Tab(label="Visuals", value="visuals"),
                dcc.Tab(label="Final Report Summary", value="report"),
                dcc.Tab(label="Team & Acknowledgments", value="team"),
            ],
        ),
        html.Div(id="tabs-content"),
    ],
)

# --- Tab callback ---
@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "summary":
        return html.Div(
            [
                html.H2("Summary Statistics", style={"color": "#b32020"}),
                html.P(
                    """
                    This section provides descriptive summaries of our dataset. 
                    You can filter by artist to view individual statistics or browse overall averages below.
                    The following variables describe key audio and engagement metrics:
                    """,
                    style={"fontSize": "16px", "marginBottom": "10px"},
                ),
                html.Ul(
                    [
                        html.Li("🎵 **Tempo (BPM)** – The overall speed or pace of a song, measured in beats per minute."),
                        html.Li("⚡ **Energy** – A measure of intensity and activity, from calm (0.0) to energetic (1.0)."),
                        html.Li("🔊 **Loudness (dB)** – Average decibel level; louder tracks typically sound more powerful."),
                        html.Li("📏 **Duration (min)** – The song length converted from milliseconds."),
                        html.Li("👀 **Views, Likes, Comments** – YouTube engagement metrics reflecting listener interaction."),
                    ],
                    style={"fontSize": "15px", "marginBottom": "25px", "lineHeight": "1.8"},
                ),
                html.Label("Filter by Artist:"),
                dcc.Dropdown(id="artist-dropdown", options=artist_options, placeholder="Select an artist"),
                html.Br(),
                dash_table.DataTable(
                    id="summary-table",
                    columns=[
                        {"name": col, "id": col}
                        for col in ["Artist", "Views", "Likes", "Comments", "tempo", "energy", "duration_ms"]
                    ],
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#b32020",
                        "color": "white",
                        "fontWeight": "bold",
                    },
                    style_data={
                        "backgroundColor": "white",
                        "color": "black",
                        "border": "1px solid #ddd",
                    },
                ),
            ]
        )

    elif tab == "model":
        return html.Div(
            [
                html.H2("Deep Learning Model Overview", style={"color": "#b32020"}),
                html.P(
                    """
                    This section summarizes our deep learning model trained to predict song success metrics 
                    (like views and likes) based on Spotify audio features. 
                    We compared multiple architectures and observed that including features like energy and danceability 
                    improved predictive accuracy significantly.
                    """,
                    style={"fontSize": "17px", "lineHeight": "1.7"},
                ),
                html.Br(),
                html.P(
                    "Future work includes expanding the model with natural language processing on song lyrics and release trends.",
                    style={"fontSize": "16px"},
                ),
            ]
        )

    elif tab == "visuals":
        return html.Div(
            [
                html.H2("Visualizations", style={"color": "#b32020"}),
                html.P(
                    "Interactive charts will be showcased here, illustrating correlations between Spotify features and YouTube engagement metrics.",
                    style={"fontSize": "17px"},
                ),
            ]
        )

    elif tab == "report":
        return html.Div(
            style={"display": "flex", "gap": "30px"},
            children=[
                html.Div(
                    [
                        html.H3("Sections", style={"color": "#b32020"}),
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
                    style={"width": "20%", "backgroundColor": "#f9f9f9", "padding": "15px", "borderRadius": "8px"},
                ),
                html.Div(
                    [
                        html.H2("Final Report Summary", style={"color": "#b32020"}),
                        html.P(
                            "This section provides an overview of our capstone report. Use the navigation menu on the left to explore each part.",
                            style={"fontSize": "16px"},
                        ),
                        html.Div(
                            id="motivation",
                            children=[
                                html.H3("1. Motivation", style={"color": "#b32020"}),
                                html.P(
                                    "Our goal was to understand the measurable patterns behind musical popularity. "
                                    "By combining Spotify and YouTube data, we aimed to uncover data-driven explanations for success.",
                                    style={"fontSize": "16px"},
                                ),
                            ],
                        ),
                        html.Div(
                            id="data",
                            children=[
                                html.H3("2. Data Collection & Processing", style={"color": "#b32020"}),
                                html.P(
                                    "We compiled over 5,000 songs, merging Spotify track metadata and YouTube video analytics. "
                                    "Cleaning steps removed duplicates, standardized durations, and ensured consistent variable formats.",
                                    style={"fontSize": "16px"},
                                ),
                            ],
                        ),
                        html.Div(
                            id="modeling",
                            children=[
                                html.H3("3. Modeling Approach", style={"color": "#b32020"}),
                                html.P(
                                    "Our deep learning model predicts engagement using acoustic features. "
                                    "We also compared regression and random forest models to interpret key predictors.",
                                    style={"fontSize": "16px"},
                                ),
                            ],
                        ),
                        html.Div(
                            id="findings",
                            children=[
                                html.H3("4. Key Findings", style={"color": "#b32020"}),
                                html.P(
                                    "Songs with balanced loudness and strong rhythmic consistency performed best. "
                                    "Energy and danceability were the most influential predictors of engagement.",
                                    style={"fontSize": "16px"},
                                ),
                            ],
                        ),
                        html.Div(
                            id="conclusions",
                            children=[
                                html.H3("5. Conclusions & Future Work", style={"color": "#b32020"}),
                                html.P(
                                    "This project demonstrates how statistical and deep learning techniques can analyze creative success quantitatively. "
                                    "Future directions include sentiment analysis of lyrics and time-based trend modeling.",
                                    style={"fontSize": "16px"},
                                ),
                            ],
                        ),
                        html.Br(),
                        html.Div(
                            [
                                html.A(
                                    "📄 Download Full Report (PDF)",
                                    href="https://drive.google.com/your-final-report-link-here",
                                    target="_blank",
                                    style={
                                        "backgroundColor": "#b32020",
                                        "color": "white",
                                        "padding": "10px 20px",
                                        "borderRadius": "8px",
                                        "textDecoration": "none",
                                        "fontWeight": "bold",
                                    },
                                )
                            ],
                            style={"textAlign": "center"},
                        ),
                    ],
                    style={
                        "width": "80%",
                        "backgroundColor": "white",
                        "padding": "25px",
                        "borderRadius": "10px",
                        "overflowY": "auto",
                        "maxHeight": "600px",
                        "boxShadow": "0px 2px 6px rgba(0,0,0,0.1)",
                    },
                ),
            ],
        )

    elif tab == "team":
        return html.Div(
            [
                html.H2("Team & Acknowledgments", style={"color": "#b32020"}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4("Capri Gallo"),
                                html.P("B.S. Statistical Data Science | University of California, Davis (2026)"),
                            ],
                            style={"backgroundColor": "white", "padding": "15px", "borderRadius": "10px", "margin": "10px"},
                        ),
                        html.Div(
                            [
                                html.H4("Alex Garcia"),
                                html.P("B.S. Statistical Data Science | University of California, Davis (2026)"),
                            ],
                            style={"backgroundColor": "white", "padding": "15px", "borderRadius": "10px", "margin": "10px"},
                        ),
                        html.Div(
                            [
                                html.H4("Rohan Pillay"),
                                html.P("B.S. Statistical Data Science | University of California, Davis (2026)"),
                            ],
                            style={"backgroundColor": "white", "padding": "15px", "borderRadius": "10px", "margin": "10px"},
                        ),
                        html.Div(
                            [
                                html.H4("Edward Ron"),
                                html.P("B.S. Statistical Data Science | University of California, Davis (2026)"),
                            ],
                            style={"backgroundColor": "white", "padding": "15px", "borderRadius": "10px", "margin": "10px"},
                        ),
                        html.Div(
                            [
                                html.H4("Yuxiao Tan"),
                                html.P("B.S. Statistical Data Science | University of California, Davis (2026)"),
                            ],
                            style={"backgroundColor": "white", "padding": "15px", "borderRadius": "10px", "margin": "10px"},
                        ),
                    ],
                    style={"display": "flex", "flexWrap": "wrap"},
                ),
                html.Br(),
                html.H3("Acknowledgments", style={"color": "#b32020"}),
                html.P("Special thanks to Professor Lingfei Cui and the UC Davis Statistics Department for their guidance."),
                html.Br(),
                html.H3("References", style={"color": "#b32020"}),
                html.Ul(
                    [
                        html.Li(
                            "Li, Y. et al. (2024). MERT: Acoustic Music Understanding Model with Large-Scale Self-Supervised Training. arXiv:2306.00107."
                        ),
                        html.Li(
                            "Grewal, R. (2025). Spotify–YouTube Data. Kaggle. https://www.kaggle.com/datasets/rohitgrewal/spotify-youtube-data"
                        ),
                    ],
                    style={"fontSize": "15px", "lineHeight": "1.8"},
                ),
            ]
        )

# --- Run app ---
if __name__ == "__main__":
    app.run(debug=True)
