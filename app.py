import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# === Load and clean data ===
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

# Drop missing values
data = data.dropna(subset=["tempo", "energy"])

# === App setup ===
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
app.title = "The Science of Song Success"

# === Layout ===
app.layout = html.Div(
    style={
        "backgroundColor": "#fff5f5",
        "fontFamily": "Open Sans, sans-serif",
        "padding": "20px",
    },
    children=[
        html.H1(
            "The Science of Song Success",
            style={
                "textAlign": "center",
                "color": "#8B0000",
                "fontWeight": "bold",
                "marginBottom": "10px",
            },
        ),
        html.P(
            "For our capstone project, we have built an interactive dashboard to visualize our findings. "
            "Our dashboard examines how different Spotify audio features and YouTube engagement metrics "
            "relate to one another, and how this relation drives a song’s success. Below are several tabs to display "
            "simple summary statistics, the deep learning model we trained, and all relevant visuals for this project.",
            style={
                "textAlign": "center",
                "fontSize": "16px",
                "maxWidth": "950px",
                "margin": "auto",
                "lineHeight": "1.6",
            },
        ),
        html.Br(),

        dcc.Tabs(
            id="tabs",
            value="summary",
            children=[
                dcc.Tab(label="Summary Statistics", value="summary", style={"fontWeight": "600"}),
                dcc.Tab(label="Deep Learning Model", value="model", style={"fontWeight": "600"}),
                dcc.Tab(label="Visuals", value="visuals", style={"fontWeight": "600"}),
            ],
        ),
        html.Div(id="tab-content", style={"padding": "20px"}),

        html.Footer(
            "Developed by Team 13 – The Science of Song Success | UC Davis, Fall 2025",
            style={
                "textAlign": "center",
                "color": "#8B0000",
                "marginTop": "40px",
                "fontSize": "13px",
            },
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
        return html.Div(
            [
                html.H3("Deep Learning Model Overview", style={"color": "#8B0000", "fontWeight": "bold"}),
                html.P(
                    "Our model uses audio and engagement features to predict song popularity. "
                    "We trained it using regression and deep learning architectures to identify the strongest predictors "
                    "of success across thousands of tracks.",
                    style={"fontSize": "16px", "maxWidth": "900px"},
                ),
            ]
        )

    elif tab == "visuals":
        return html.Div(
            [
                html.H3("Visuals and Data Exploration", style={"color": "#8B0000", "fontWeight": "bold"}),
                html.P(
                    "This section will include interactive charts, scatterplots, and correlation heatmaps "
                    "showing the relationships between song features and performance metrics.",
                    style={"fontSize": "16px", "maxWidth": "900px"},
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

    # Apply filters
    if artist:
        df = df[df["Artist"] == artist]
    if search_query:
        search_query = search_query.lower()
        df = df[
            df["filename"].str.lower().str.contains(search_query, na=False)
            | df["Album"].str.lower().str.contains(search_query, na=False)
            | df["Artist"].str.lower().str.contains(search_query, na=False)
        ]

    # Summary cards
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
                            html.H5(stat["label"], className="card-title", style={"color": "#8B0000"}),
                            html.H3(stat["value"], className="card-text"),
                        ]
                    ),
                    style={
                        "backgroundColor": "#fff",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
                        "borderRadius": "10px",
                    },
                ),
                width=3,
            )
            for stat in stats
        ],
        className="g-3",
    )

    # Interactive data table
    table = dash_table.DataTable(
        df[["filename", "duration_min", "tempo", "loudness", "energy", "Views", "Likes"]].to_dict("records"),
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
        filter_action="native",
        page_size=10,
    )

    return html.Div([cards, html.Br(), table])


if __name__ == "__main__":
    app.run_server(debug=True)
