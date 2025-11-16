import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table

# --- Load and clean data ---
data = pd.read_csv("merged_final_data.csv")

# Clean tempo column (remove brackets, convert to float)
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

# --- Create dropdown options ---
artist_options = [{"label": artist, "value": artist} for artist in sorted(clean_data["Artist"].dropna().unique())]

# --- Initialize app ---
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Music Analytics Dashboard"

# --- App layout ---
app.layout = html.Div(
    style={
        "backgroundColor": "#faf7f7",
        "fontFamily": "Georgia, serif",
        "padding": "25px",
        "color": "#222",
    },
    children=[
        # Header
        html.Div(
            style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"},
            children=[
                html.H1("🎶 Music Analytics & Insights", style={"color": "#b32020", "fontSize": "36px"}),
                html.Img(src="/assets/ucdavis_logo.png", style={"height": "70px", "marginLeft": "15px"}),
            ],
        ),
        html.Hr(style={"borderTop": "3px solid #b32020"}),

        html.P(
            """
            For our capstone project, we have built an interactive dashboard to visualize our findings.
            Our dashboard examines how different Spotify audio features and YouTube engagement metrics relate to one another,
            and how this relation drives a song's success. Below are several tabs to display simple summary statistics,
            the deep learning model we trained, and all relevant visuals for this project.
            """,
            style={"fontSize": "18px", "lineHeight": "1.6"},
        ),

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

# --- Callbacks for tabs ---
@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "summary":
        return html.Div(
            [
                html.H2("Summary Statistics", style={"color": "#b32020"}),
                html.Label("Select an Artist:"),
                dcc.Dropdown(id="artist-dropdown", options=artist_options, placeholder="Select an artist"),
                html.Br(),
                dash_table.DataTable(
                    id="summary-table",
                    columns=[{"name": col, "id": col} for col in ["Artist", "Views", "Likes", "Comments", "tempo", "energy", "duration_ms"]],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#b32020", "color": "white", "fontWeight": "bold"},
                    style_data={"backgroundColor": "white", "color": "black", "border": "1px solid #ddd"},
                ),
            ]
        )

    elif tab == "model":
        return html.Div(
            [
                html.H2("Deep Learning Model Overview", style={"color": "#b32020"}),
                html.P(
                    "This section presents the architecture, training results, and evaluation metrics of our deep learning model.",
                    style={"fontSize": "17px"},
                ),
            ]
        )

    elif tab == "visuals":
        return html.Div(
            [
                html.H2("Visualizations", style={"color": "#b32020"}),
                html.P("Interactive plots and feature comparisons will be displayed here.", style={"fontSize": "17px"}),
            ]
        )

    elif tab == "report":
        return html.Div(
            [
                html.H2("Final Report Summary", style={"color": "#b32020"}),
                html.P(
                    "Below is a condensed overview of our capstone project's final report. "
                    "Each section summarizes the main ideas, findings, and lessons learned. "
                    "Once the written report is finalized, this section will include the full executive summary.",
                    style={"fontSize": "17px", "lineHeight": "1.7"},
                ),
                html.H3("1. Motivation", style={"color": "#b32020"}),
                html.P(
                    "Our project aims to understand the factors that drive a song’s success by analyzing "
                    "Spotify audio features and YouTube engagement metrics.",
                    style={"fontSize": "16px", "lineHeight": "1.7"},
                ),
                html.H3("2. Data Collection & Processing", style={"color": "#b32020"}),
                html.P(
                    "The dataset combines Spotify track-level audio data with YouTube video statistics from Rohit Grewal’s Kaggle dataset. "
                    "After cleaning and merging, we analyzed over 5,000 songs.",
                    style={"fontSize": "16px", "lineHeight": "1.7"},
                ),
                html.H3("3. Modeling Approach", style={"color": "#b32020"}),
                html.P(
                    "We developed multiple models, including regression and a deep learning neural network, "
                    "to predict song popularity based on key features.",
                    style={"fontSize": "16px", "lineHeight": "1.7"},
                ),
                html.H3("4. Key Findings", style={"color": "#b32020"}),
                html.P(
                    "Our findings indicate that energy, tempo, and loudness correlate strongly with listener engagement. "
                    "Songs with balanced acoustic features—moderate loudness, high energy, and mid-range tempo—perform better on YouTube.",
                    style={"fontSize": "16px", "lineHeight": "1.7"},
                ),
                html.H3("5. Conclusions & Future Work", style={"color": "#b32020"}),
                html.P(
                    "This project demonstrates how quantitative analysis can reveal patterns in music success. "
                    "Future work may include genre-based modeling, sentiment analysis of lyrics, and integration of streaming trends over time.",
                    style={"fontSize": "16px", "lineHeight": "1.7"},
                ),
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
                    style={"textAlign": "center", "marginTop": "25px"},
                ),
            ],
            style={
                "backgroundColor": "white",
                "padding": "25px",
                "borderRadius": "10px",
                "boxShadow": "0px 2px 6px rgba(0,0,0,0.1)",
            },
        )

    elif tab == "team":
        return html.Div(
            [
                html.H2("Team & Acknowledgments", style={"color": "#b32020"}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4("Capri Gallo", style={"marginBottom": "0"}),
                                html.P("Statistical Data Science, UC Davis"),
                            ],
                            style={"padding": "10px", "backgroundColor": "white", "borderRadius": "10px", "margin": "5px"},
                        ),
                        html.Div(
                            [
                                html.H4("Alex Garcia", style={"marginBottom": "0"}),
                                html.P("Statistical Data Science, UC Davis"),
                            ],
                            style={"padding": "10px", "backgroundColor": "white", "borderRadius": "10px", "margin": "5px"},
                        ),
                        html.Div(
                            [
                                html.H4("Rohan Pillay", style={"marginBottom": "0"}),
                                html.P("Statistical Data Science, UC Davis"),
                            ],
                            style={"padding": "10px", "backgroundColor": "white", "borderRadius": "10px", "margin": "5px"},
                        ),
                        html.Div(
                            [
                                html.H4("Edward Ron", style={"marginBottom": "0"}),
                                html.P("Statistical Data Science, UC Davis"),
                            ],
                            style={"padding": "10px", "backgroundColor": "white", "borderRadius": "10px", "margin": "5px"},
                        ),
                        html.Div(
                            [
                                html.H4("Yuxiao Tan", style={"marginBottom": "0"}),
                                html.P("Statistical Data Science, UC Davis"),
                            ],
                            style={"padding": "10px", "backgroundColor": "white", "borderRadius": "10px", "margin": "5px"},
                        ),
                    ],
                    style={"display": "flex", "flexWrap": "wrap"},
                ),
                html.Br(),
                html.H3("Acknowledgments", style={"color": "#b32020"}),
                html.P("Special thanks to Professor Lingfei Cui and the UC Davis Statistics Department for their guidance."),
                html.H3("References", style={"color": "#b32020"}),
                html.Ul(
                    [
                        html.Li("Li, Y. et al. (2024). MERT: Acoustic Music Understanding Model with Large-Scale Self-Supervised Training. *arXiv:2306.00107*."),
                        html.Li("Grewal, R. (2025). Spotify–YouTube Data. *Kaggle*. https://www.kaggle.com/datasets/rohitgrewal/spotify-youtube-data"),
                    ]
                ),
            ]
        )

# --- Run app ---
if __name__ == "__main__":
    app.run(debug=True)
