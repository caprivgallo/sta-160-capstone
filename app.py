import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# === Load Data ===
data = pd.read_csv("merged_final_data.csv")

# Clean tempo values
data["tempo"] = (
    data["tempo"]
    .astype(str)
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .astype(float)
)
data["energy"] = pd.to_numeric(data["energy"], errors="coerce")

# Drop missing numeric rows
data = data.dropna(subset=["tempo", "energy"])

# === App setup with custom theme ===
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
            "relate to one another to uncover what drives a song’s success. Explore the tabs below to view our analyses and interactive summaries.",
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
    ],
)


# === Callbacks ===
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
                    style={"width": "50%", "marginBottom": "20px"},
                ),
                html.Div(id="summary-output"),
            ]
        )
    else:
        return html.P("Additional visualizations will appear here.")


@app.callback(Output("summary-output", "children"), [Input("artist-dropdown", "value")])
def update_summary(artist):
    df = data if artist is None else data[data["Artist"] == artist]

    summary = {
        "Average Tempo": round(df["tempo"].mean(), 2),
        "Average Energy": round(df["energy"].mean(), 2),
        "Average Loudness": round(df["loudness"].mean(), 2),
        "Average Views": int(df["Views"].mean()),
        "Average Likes": int(df["Likes"].mean()),
    }

    return html.Div(
        [
            html.H4(f"Summary for {'All Artists' if artist is None else artist}", style={"color": "#8B0000"}),
            html.Ul([html.Li(f"{k}: {v:,}") for k, v in summary.items()]),
            html.Br(),
            dash_table.DataTable(
                df.head(10).to_dict("records"),
                [{"name": i, "id": i} for i in df.columns],
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#8B0000",
                    "color": "white",
                    "fontWeight": "bold",
                },
                style_data={"backgroundColor": "#fffaf9", "border": "1px solid #eee"},
                page_size=10,
            ),
        ]
    )


if __name__ == "__main__":
    app.run_server(debug=True)
