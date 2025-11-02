import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# === Load Data ===
data = pd.read_csv("merged_final_data.csv")

# --- Clean numeric columns ---
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

# Drop missing values
data = data.dropna(subset=["tempo", "energy"])

# === App Setup ===
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
app.title = "STA 160 Capstone Dashboard"

# === Layout ===
app.layout = html.Div(
    style={"backgroundColor": "#fff5f5", "fontFamily": "Open Sans, sans-serif", "padding": "20px"},
    children=[
        html.H1(
            "STA 160 Capstone Dashboard",
            style={"textAlign": "center", "color": "#8B0000", "fontWeight": "bold"},
        ),
        html.P(
            "For our STA 160 thesis capstone, we have built an interactive dashboard to visualize our findings. "
            "It examines how Spotify audio features and YouTube engagement metrics (such as views, likes, and comments) "
            "relate to one another to uncover what drives a song’s success. Use the tabs below to explore our interactive analyses and results.",
            style={"textAlign": "center", "fontSize": "16px", "maxWidth": "1000px", "margin": "auto"},
        ),
        html.Br(),

        dcc.Tabs(
            id="tabs",
            value="summary",
            children=[
                dcc.Tab(label="Summary Statistics", value="summary", style={"fontWeight": "600"}),
                dcc.Tab(label="Scatterplots", value="scatter", style={"fontWeight": "600"}),
                dcc.Tab(label="Distributions", value="dist", style={"fontWeight": "600"}),
                dcc.Tab(label="Correlation Heatmap", value="corr", style={"fontWeight": "600"}),
            ],
        ),
        html.Div(id="tab-content", style={"padding": "20px"}),

        html.Footer(
            "Developed by Team 13 – STA 160 Capstone, UC Davis (Fall 2025)",
            style={"textAlign": "center", "color": "#8B0000", "marginTop": "40px", "fontSize": "13px"},
        ),
    ],
)

# === Tab rendering ===
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
                html.Div(id="summary-output"),
            ]
        )
    else:
        return html.P("Additional visualizations will appear here.")


@app.callback(Output("summary-output", "children"), [Input("artist-dropdown", "value")])
def update_summary(artist):
    df = data if artist is None else data[data["Artist"] == artist]

    # Key stats for cards
    stats = [
        {"label": "Songs", "value": len(df)},
        {"label": "Avg Tempo (BPM)", "value": f"{df['tempo'].mean():.1f}"},
        {"label": "Avg Energy", "value": f"{df['energy'].mean():.2f}"},
        {"label": "Avg Loudness (dB)", "value": f"{df['loudness'].mean():.1f}"},
    ]

    # Cards layout
    cards = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5(stat["label"], className="card-title", style={"color": "#8B0000"}),
                            html.H3(stat["value"], className="card-text"),
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

    # Data Table
    table = dash_table.DataTable(
        df[["filename", "duration_min", "tempo", "loudness", "energy", "Views", "Likes"]].head(10).to_dict("records"),
        [{"name": col, "id": col} for col in ["filename", "duration_min", "tempo", "loudness", "energy", "Views", "Likes"]],
        style_table={"overflowX": "auto", "marginTop": "25px"},
        style_header={
            "backgroundColor": "#8B0000",
            "color": "white",
            "fontWeight": "bold",
            "textAlign": "center",
        },
        style_data={"backgroundColor": "#fffaf9", "border": "1px solid #eee"},
        sort_action="native",
        page_size=10,
    )

    return html.Div([cards, html.Br(), table])


if __name__ == "__main__":
    app.run_server(debug=True)
