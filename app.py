import os
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# === Load and clean dataset ===
data = pd.read_csv("merged_final_data.csv")

# Clean numeric columns
data["tempo"] = (
    data["tempo"]
    .astype(str)
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .astype(float)
)
data["energy"] = pd.to_numeric(data["energy"], errors="coerce")
data["loudness"] = pd.to_numeric(data["loudness"], errors="coerce")
data["duration_ms"] = pd.to_numeric(data["duration_ms"], errors="coerce")

# Convert duration to MM:SS
data["duration_min"] = data["duration_ms"].apply(
    lambda x: f"{int(x // 60000)}:{int((x % 60000) / 1000):02d}" if pd.notnull(x) else None
)
data = data.dropna(subset=["tempo", "energy"])

# Optional model files
model_summary_text = None
model_metrics = None
training_data = None

if os.path.exists("model_summary.txt"):
    with open("model_summary.txt", "r") as f:
        model_summary_text = f.read()

if os.path.exists("model_results.csv"):
    model_metrics = pd.read_csv("model_results.csv")

if os.path.exists("training_history.csv"):
    training_data = pd.read_csv("training_history.csv")
else:
    training_data = pd.DataFrame({
        "epoch": range(1, 21),
        "loss": [1.0/(i+0.5) + 0.05 for i in range(20)],
        "val_loss": [1.0/(i+0.7) + 0.07 for i in range(20)]
    })

# === App setup ===
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
app.title = "The Science of Song Success"

# === Layout ===
app.layout = html.Div(
    style={"backgroundColor": "#fff5f5", "fontFamily": "Raleway, sans-serif", "padding": "25px"},
    children=[
        html.Div(
            [html.Img(
    src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/UC_Davis_wordmark.svg/512px-UC_Davis_wordmark.svg.png",
    style={"height": "60px", "display": "block", "margin": "0 auto 10px auto"},
),
                ),
                html.H1(
                    "The Science of Song Success",
                    style={
                        "textAlign": "center",
                        "color": "#8B0000",
                        "fontWeight": "800",
                        "fontSize": "36px",
                        "marginBottom": "5px",
                    },
                ),
                html.P(
                    "For our capstone project, we have built an interactive dashboard to visualize our findings. "
                    "Our dashboard examines how different Spotify audio features and YouTube engagement metrics "
                    "relate to one another, and how this relation drives a song's success. Below are several tabs "
                    "to display simple summary statistics, the deep learning model we trained, and all relevant visuals for this project.",
                    style={
                        "textAlign": "center",
                        "fontSize": "17px",
                        "maxWidth": "950px",
                        "margin": "auto",
                        "lineHeight": "1.6",
                    },
                ),
            ]
        ),
        html.Br(),

        dcc.Tabs(
            id="tabs",
            value="summary",
            children=[
                dcc.Tab(label="Summary Statistics", value="summary", style={"fontWeight": "600"}),
                dcc.Tab(label="Deep Learning Model", value="model", style={"fontWeight": "600"}),
                dcc.Tab(label="Visuals", value="visuals", style={"fontWeight": "600"}),
                dcc.Tab(label="Presentation", value="presentation", style={"fontWeight": "600"}),
                dcc.Tab(label="Team & Acknowledgments", value="team", style={"fontWeight": "600"}),
            ],
        ),
        html.Div(id="tab-content", style={"padding": "25px"}),

        html.Footer(
            "Developed by Team 13 — The Science of Song Success | UC Davis Fall 2025",
            style={"textAlign": "center", "color": "#8B0000", "marginTop": "40px", "fontSize": "13px"},
        ),
    ],
)

# === Tabs rendering ===
@app.callback(Output("tab-content", "children"), [Input("tabs", "value")])
def render_tab(tab):
    if tab == "summary":
        return html.Div(
            [
                html.H3("Summary Statistics", style={"color": "#8B0000", "fontWeight": "bold"}),
                html.Label("Filter by Artist:"),
                dcc.Dropdown(
                    id="artist-dropdown",
                    options=[{"label": a, "value": a} for a in sorted(data["Artist"].unique())],
                    value=None,
                    placeholder="Select an artist...",
                    style={"width": "50%", "marginBottom": "25px"},
                ),
                html.Label("Search Songs or Albums:"),
                dcc.Input(
                    id="search-bar",
                    type="text",
                    placeholder="Search...",
                    debounce=True,
                    style={
                        "width": "50%",
                        "marginBottom": "25px",
                        "padding": "8px",
                        "border": "1px solid #8B0000",
                        "borderRadius": "5px",
                    },
                ),
                html.Div(id="summary-output"),
            ]
        )

    elif tab == "model":
        content = [
            html.H3("Deep Learning Model Overview", style={"color": "#8B0000", "fontWeight": "bold"}),
            html.P(
                "Our deep learning model was designed to predict song popularity using both Spotify audio features "
                "and YouTube engagement metrics. It captures nonlinear relationships between sound characteristics "
                "and listener behavior.",
                style={"fontSize": "16px", "maxWidth": "900px"},
            ),
            html.Br(),
        ]

        if model_summary_text:
            content.append(
                html.Div(
                    [
                        html.H5("Model Architecture", style={"color": "#8B0000"}),
                        html.Pre(model_summary_text, style={"backgroundColor": "#fff", "padding": "10px", "borderRadius": "8px"}),
                    ]
                )
            )
        else:
            content.append(html.P("Model summary not yet uploaded (expected: model_summary.txt)."))

        if model_metrics is not None:
            content.append(
                html.Div(
                    [
                        html.H5("Performance Metrics", style={"color": "#8B0000"}),
                        dash_table.DataTable(
                            model_metrics.to_dict("records"),
                            [{"name": i, "id": i} for i in model_metrics.columns],
                            style_table={"overflowX": "auto"},
                            style_header={"backgroundColor": "#8B0000", "color": "white", "fontWeight": "bold"},
                            style_data={"backgroundColor": "#fffaf9", "border": "1px solid #eee"},
                        ),
                    ]
                )
            )

        fig = px.line(
            training_data,
            x="epoch",
            y=["loss", "val_loss"],
            labels={"value": "Loss", "epoch": "Epoch", "variable": "Type"},
            title="Training and Validation Loss Curve",
        )
        fig.update_layout(
            plot_bgcolor="#fffaf9",
            paper_bgcolor="#fffaf9",
            title_font_color="#8B0000",
            font={"family": "Raleway"},
        )

        content.append(
            html.Div(
                [
                    html.H5("Training Visualizations", style={"color": "#8B0000", "marginTop": "25px"}),
                    dcc.Graph(figure=fig),
                    html.P(
                        "This plot will automatically update if training_history.csv is uploaded.",
                        style={"fontSize": "13px", "fontStyle": "italic"},
                    ),
                ]
            )
        )
        return html.Div(content)

    elif tab == "visuals":
        return html.Div(
            [
                html.H3("Visuals and Data Exploration", style={"color": "#8B0000", "fontWeight": "bold"}),
                html.P(
                    "This section will include interactive charts and correlation visualizations showing relationships between song features and popularity metrics.",
                    style={"fontSize": "16px", "maxWidth": "900px"},
                ),
            ]
        )

    elif tab == "presentation":
        return html.Div(
            [
                html.H3("Capstone Video Presentation", style={"color": "#8B0000", "fontWeight": "bold"}),
                html.P(
                    "Our final video presentation will appear here soon. "
                    "This section will include our recorded explanation of the project, methodology, and results.",
                    style={"fontSize": "16px", "maxWidth": "900px"},
                ),
                html.Div(
                    html.P(
                        "Stay tuned for the video upload closer to our final submission!",
                        style={"fontStyle": "italic", "color": "#a33"},
                    ),
                    style={"marginTop": "30px", "textAlign": "center"},
                ),
            ]
        )

    elif tab == "team":
        return html.Div(
            [
                html.H3("Team & Acknowledgments", style={"color": "#8B0000", "fontWeight": "bold", "textAlign": "center"}),
                html.Br(),
                html.H5("Team Members", style={"color": "#8B0000", "textAlign": "center", "marginBottom": "25px"}),

                dbc.Row(
                    [
                        *[
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H5(name, className="card-title", style={"color": "#8B0000"}),
                                            html.P(
                                                "B.S. Statistical Data Science | University of California, Davis (2026)",
                                                style={"fontSize": "15px"},
                                            ),
                                        ]
                                    ),
                                    style={
                                        "backgroundColor": "#fff",
                                        "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
                                        "borderRadius": "10px",
                                        "textAlign": "center",
                                    },
                                ),
                                width=3,
                            )
                            for name in ["Capri Gallo", "Alex Garcia", "Rohan Pillay", "Edward Ron", "Yuxiao Tan"]
                        ]
                    ],
                    className="g-4",
                    justify="center",
                ),

                html.Br(),
                html.H5("Acknowledgments", style={"color": "#8B0000", "textAlign": "center"}),
                html.P(
                    "Special thanks to Professor Lingfei Cui and the UC Davis Statistics Department for their guidance and support throughout this project.",
                    style={"textAlign": "center", "maxWidth": "800px", "margin": "auto"},
                ),
            ]
        )

    return html.P("Select a tab to view content.")


# === Summary tab callback ===
@app.callback(
    Output("summary-output", "children"),
    [Input("artist-dropdown", "value"), Input("search-bar", "value")],
)
def update_summary(artist, search_query):
    df = data.copy()
    if artist:
        df = df[df["Artist"] == artist]
    if search_query:
        search_query = search_query.lower()
        df = df[
            df["filename"].str.lower().str.contains(search_query, na=False)
            | df["Album"].str.lower().str.contains(search_query, na=False)
            | df["Artist"].str.lower().str.contains(search_query, na=False)
        ]

    stats = [
        {"label": "Songs", "value": len(df)},
        {"label": "Avg Tempo (BPM)", "value": f"{df['tempo'].mean():.1f}"},
        {"label": "Avg Energy", "value": f"{df['energy'].mean():.2f}"},
        {"label": "Avg Loudness (dB)", "value": f"{df['loudness'].mean():.1f}"},
    ]

    cards = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5(stat["label"], style={"color": "#8B0000"}),
                            html.H3(stat["value"]),
                        ]
                    ),
                    style={"backgroundColor": "#fff", "boxShadow": "0 2px 6px rgba(0,0,0,0.1)", "borderRadius": "10px"},
                ),
                width=3,
            )
            for stat in stats
        ],
        className="g-3",
    )

    table = dash_table.DataTable(
        df[["filename", "duration_min", "tempo", "loudness", "energy", "Views", "Likes"]].to_dict("records"),
        [{"name": c, "id": c} for c in ["filename", "duration_min", "tempo", "loudness", "energy", "Views", "Likes"]],
        style_table={"overflowX": "auto", "marginTop": "25px"},
        style_header={"backgroundColor": "#8B0000", "color": "white", "fontWeight": "bold", "textAlign": "center"},
        style_data={"backgroundColor": "#fffaf9", "border": "1px solid #eee"},
        sort_action="native",
        filter_action="native",
        page_size=10,
    )

    return html.Div([cards, html.Br(), table])


if __name__ == "__main__":
    app.run_server(debug=True)
