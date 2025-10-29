import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from dash import Dash, dcc, html, Input, Output, dash_table

# --- Load dataset ---
data = pd.read_csv("merged_final_data.csv")

# --- Clean dataset ---
drop_cols = ["filename", "Uri", "Url_spotify", "Url_youtube", "video_id", "Track_URL_Spotify"]
for col in drop_cols:
    if col in data.columns:
        data = data.drop(columns=[col])

# Clean Track/Title
if "Track" in data.columns:
    data["Track"] = data["Track"].astype(str).str.replace(".opus", "", regex=False)
if "Title" in data.columns:
    data["Title"] = data["Title"].astype(str).str.replace(".opus", "", regex=False)

# Make column names pretty
data = data.rename(columns=lambda x: x.replace("_", " ").title())

# Clean tempo column
if "Tempo" in data.columns:
    data["Tempo"] = (
        data["Tempo"].astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .astype(float)
    )

# Numeric columns
numeric_cols = data.select_dtypes(include="number").columns

# --- Initialize app ---
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# --- Layout ---
app.layout = html.Div(
    style={
        "fontFamily": '"Helvetica Neue", Arial, sans-serif',
        "margin": "0",
        "padding": "0",
        "backgroundColor": "#fafafa"
    },
    children=[
        # Header
        html.Div(
            "STA 160 Capstone Dashboard",
            style={
                "textAlign": "center",
                "backgroundColor": "#2C3E50",
                "color": "white",
                "padding": "20px",
                "fontSize": "30px",
                "fontWeight": "600",
                "letterSpacing": "1px"
            }
        ),

        # Intro
        html.P(
            "For our STA 160 thesis capstone, we have built an interactive dashboard to display our findings. "
            "It explores how audio features from Spotify and popularity metrics from YouTube "
            "interact to reveal what makes certain songs successful. "
            "Use the tabs below to explore different views of the data.",
            style={
                "textAlign": "center",
                "fontSize": "18px",
                "maxWidth": "900px",
                "margin": "25px auto",
                "color": "#2C3E50",
                "lineHeight": "1.6"
            }
        ),

        # Tabs
        dcc.Tabs(
            id="tabs",
            value="overview",
            children=[
                dcc.Tab(label="Overview", value="overview",
                        style={"padding": "12px"},
                        selected_style={"backgroundColor": "#1ABC9C", "color": "white", "fontWeight": "bold"}),

                dcc.Tab(label="Scatterplots", value="scatter",
                        style={"padding": "12px"},
                        selected_style={"backgroundColor": "#1ABC9C", "color": "white", "fontWeight": "bold"}),

                dcc.Tab(label="Distributions", value="dist",
                        style={"padding": "12px"},
                        selected_style={"backgroundColor": "#1ABC9C", "color": "white", "fontWeight": "bold"}),

                dcc.Tab(label="Correlation Heatmap", value="corr",
                        style={"padding": "12px"},
                        selected_style={"backgroundColor": "#1ABC9C", "color": "white", "fontWeight": "bold"}),
            ],
            style={"fontWeight": "500", "fontSize": "16px"}
        ),

        html.Div(id="tabs-content", style={"margin": "20px"})
    ]
)

# --- Tabs callback ---
@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "overview":
        table_cols = ["Track", "Artist", "Title", "Album", "Views", "Likes", "Comments", "Tempo", "Energy"]
        available_cols = [c for c in table_cols if c in data.columns]

        return html.Div([
            html.H2("Project Overview"),
            html.P(
                "Our dataset combines Spotify audio features (such as tempo, energy, loudness, danceability) "
                "with YouTube engagement metrics (views, likes, comments). "
                "The goal is to better understand patterns across platforms "
                "and identify which features might drive popularity."
            ),

            html.Div([
                html.Div([
                    html.Label("Filter by Artist:"),
                    dcc.Dropdown(
                        id="artist-filter",
                        options=[{"label": a, "value": a} for a in sorted(data["Artist"].dropna().unique())],
                        placeholder="Select an Artist",
                        multi=False,
                        style={"width": "300px"}
                    )
                ], style={"display": "inline-block", "marginRight": "20px"}),

                html.Div([
                    html.Label("Filter by Album:"),
                    dcc.Dropdown(
                        id="album-filter",
                        options=[{"label": a, "value": a} for a in sorted(data["Album"].dropna().unique())],
                        placeholder="Select an Album",
                        multi=False,
                        style={"width": "300px"}
                    )
                ], style={"display": "inline-block"})
            ], style={"marginBottom": "20px"}),

            dash_table.DataTable(
                id="main-table",
                data=data[available_cols].to_dict("records"),
                columns=[{"name": i, "id": i} for i in available_cols],
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_current=0,
                page_size=15,
                style_table={"overflowX": "auto", "border": "1px solid #ddd"},
                style_cell={
                    "padding": "10px",
                    "textAlign": "left",
                    "fontFamily": '"Helvetica Neue", Arial, sans-serif',
                    "backgroundColor": "white",
                    "fontSize": "14px"
                },
                style_header={
                    "backgroundColor": "#1ABC9C",
                    "color": "white",
                    "fontWeight": "bold",
                    "fontSize": "15px"
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"},
                    {"if": {"state": "active"}, "backgroundColor": "#eafaf7", "border": "1px solid #1ABC9C"}
                ]
            )
        ])

    elif tab == "scatter":
        return html.Div([
            html.H2("Interactive Scatterplots"),
            html.P("Choose variables to explore relationships between features and engagement."),
            html.Label("Select X-axis:"),
            dcc.Dropdown(id="x-col", options=[{"label": c, "value": c} for c in numeric_cols], value="Tempo"),
            html.Label("Select Y-axis:"),
            dcc.Dropdown(id="y-col", options=[{"label": c, "value": c} for c in numeric_cols], value="Views"),
            dcc.Graph(id="scatter-plot")
        ])

    elif tab == "dist":
        return html.Div([
            html.H2("Distributions"),
            html.P("Explore the distribution of different numeric features."),
            html.Label("Select variable:"),
            dcc.Dropdown(id="dist-col", options=[{"label": c, "value": c} for c in numeric_cols], value="Likes"),
            dcc.Graph(id="hist-plot"),
            dcc.Graph(id="box-plot")
        ])

    elif tab == "corr":
        corr = data[numeric_cols].corr()
        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            annotation_text=corr.round(2).values,
            showscale=True,
            colorscale="Teal"
        )
        return html.Div([
            html.H2("Correlation Heatmap"),
            html.P("This heatmap shows correlations between different features in the dataset."),
            dcc.Graph(figure=fig)
        ])

# --- Callbacks for scatter and dist ---
@app.callback(
    Output("scatter-plot", "figure"),
    Input("x-col", "value"),
    Input("y-col", "value"),
    prevent_initial_call=True
)
def update_scatter(x_col, y_col):
    return px.scatter(
        data, x=x_col, y=y_col,
        opacity=0.6, template="plotly_white",
        title=f"{x_col} vs {y_col}",
        labels={x_col: x_col, y_col: y_col},
        color_discrete_sequence=["#1ABC9C"]
    )

@app.callback(
    Output("hist-plot", "figure"),
    Output("box-plot", "figure"),
    Input("dist-col", "value"),
    prevent_initial_call=True
)
def update_dist(col):
    hist = px.histogram(data, x=col, nbins=30, template="plotly_white",
                        title=f"Histogram of {col}", color_discrete_sequence=["#1ABC9C"])
    box = px.box(data, y=col, template="plotly_white",
                 title=f"Boxplot of {col}", color_discrete_sequence=["#1ABC9C"])
    return hist, box

# --- Callback for filtering table ---
@app.callback(
    Output("main-table", "data"),
    Input("artist-filter", "value"),
    Input("album-filter", "value")
)
def update_table(selected_artist, selected_album):
    filtered = data.copy()
    if selected_artist:
        filtered = filtered[filtered["Artist"] == selected_artist]
    if selected_album:
        filtered = filtered[filtered["Album"] == selected_album]

    table_cols = ["Track", "Artist", "Title", "Album", "Views", "Likes", "Comments", "Tempo", "Energy"]
    available_cols = [c for c in table_cols if c in filtered.columns]
    return filtered[available_cols].to_dict("records")

# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
