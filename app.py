import os
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# === Load and clean dataset ===
data = pd.read_csv("merged_final_data.csv")

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

data["duration_min"] = data["duration_ms"].apply(
    lambda x: f"{int(x // 60000)}:{int((x % 60000) / 1000):02d}" if pd.notnull(x) else None
)
data = data.dropna(subset=["tempo", "energy"])

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
app.title = "The Science of Song Success"

# === Layout ===
app.layout = html.Div(
    style={"backgroundColor": "#fff5f5", "fontFamily": "Raleway, sans-serif", "padding": "25px"},
    children=[
        html.Div(
            [
                # 🔹 UC Davis seal aligned to left
                html.Div(
                    html.Img(
                        src="/assets/ucdavis_logo.png",
                        style={
                            "height": "80px",
                            "float": "left",
                            "marginRight": "25px",
                            "marginLeft": "15px",
                        },
                    ),
                    style={"textAlign": "left"},
                ),

                # 🔹 Main title & description centered beside logo
                html.Div(
                    [
                        html.H1(
                            "The Science of Song Success",
                            style={
                                "textAlign": "center",
                                "color": "#8B0000",
                                "fontWeight": "800",
                                "fontSize": "36px",
                                "marginBottom": "8px",
                            },
                        ),
                        html.Div(
                            style={
                                "width": "160px",
                                "height": "3px",
                                "backgroundColor": "#C8A200",
                                "margin": "8px auto 15px auto",
                                "borderRadius": "2px",
                            }
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
                html.Div(style={"clear": "both"}),  # clears float so layout doesn’t break
            ]
        ),
        html.Br(),
        # (your tabs stay here)
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

# === Tabs rendering (shortened to key section) ===
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

# === Summary callback (same as before) ===
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
                        [html.H5(stat["label"], style={"color": "#8B0000"}), html.H3(stat["value"])]
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
