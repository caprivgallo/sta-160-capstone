import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from dash import Dash, dcc, html, Input, Output, dash_table

# --- Load and clean data ---
data = pd.read_csv("merged_final_data.csv")

# Clean tempo (remove brackets if they exist)
data["tempo"] = (
    data["tempo"].astype(str)
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .astype(float)
)

# Ensure numeric columns
numeric_cols = data.select_dtypes(include="number").columns

# --- Initialize app ---
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# --- Layout with Tabs ---
app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "margin": "20px"},
    children=[
        html.H1("🎶 STA 160 Capstone Dashboard", style={"textAlign": "center"}),

        html.P(
            "For our STA 160 thesis capstone, we have built an interactive dashboard to display our findings. "
            "It explores how audio features from Spotify and popularity metrics from YouTube "
            "interact to reveal what makes certain songs successful. "
            "Use the tabs below to explore different views of the data.",
            style={"textAlign": "center", "fontSize": "18px"}
        ),

        dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="Overview", value="overview"),
            dcc.Tab(label="Scatterplots", value="scatter"),
            dcc.Tab(label="Distributions", value="dist"),
            dcc.Tab(label="Correlation Heatmap", value="corr"),
        ]),

        html.Div(id="tabs-content", style={"marginTop": "20px"})
    ]
)

# --- Callbacks for tabs ---
@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "overview":
        return html.Div([
            html.H2("Project Overview"),
            html.P(
                "Our dataset combines Spotify audio features "
                "(such as tempo, energy, loudness, danceability) "
                "with YouTube engagement metrics (views, likes, comments). "
                "The goal is to better understand patterns across platforms "
                "and identify which features might drive popularity."
            ),
            html.P("Below is a preview of the dataset (first 5 rows):", style={"marginTop": "15px"}),

            dash_table.DataTable(
                data=data.head(5).to_dict("records"),
                columns=[{"name": i, "id": i} for i in data.columns],
                style_table={"overflowX": "auto"},
                style_cell={"padding": "5px", "textAlign": "left", "fontFamily": "Arial"},
                style_header={"backgroundColor": "#f4f4f4", "fontWeight": "bold"}
            )
        ])

    elif tab == "scatter":
        return html.Div([
            html.H2("Interactive Scatterplots"),
            html.P("Choose variables to see relationships between audio features and engagement metrics."),
            html.Label("Select X-axis:"),
            dcc.Dropdown(id="x-col", options=[{"label": c, "value": c} for c in numeric_cols], value="tempo"),
            html.Label("Select Y-axis:"),
            dcc.Dropdown(id="y-col", options=[{"label": c, "value": c} for c in numeric_cols], value="energy"),
            dcc.Graph(id="scatter-plot")
        ])

    elif tab == "dist":
        return html.Div([
            html.H2("Distributions"),
            html.P("Explore the distribution of different numeric features."),
            html.Label("Select variable:"),
            dcc.Dropdown(id="dist-col", options=[{"label": c, "value": c} for c in numeric_cols], value="views"),
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
            colorscale="Blues"
        )
        return html.Div([
            html.H2("Correlation Heatmap"),
            html.P("This heatmap shows correlations between different features in the dataset."),
            dcc.Graph(figure=fig)
        ])

# --- Scatterplot callback ---
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
        labels={x_col: x_col, y_col: y_col}
    )

# --- Distribution callback ---
@app.callback(
    Output("hist-plot", "figure"),
    Output("box-plot", "figure"),
    Input("dist-col", "value"),
    prevent_initial_call=True
)
def update_dist(col):
    hist = px.histogram(data, x=col, nbins=30, template="plotly_white", title=f"Histogram of {col}")
    box = px.box(data, y=col, template="plotly_white", title=f"Boxplot of {col}")
    return hist, box

# --- Run server ---
if __name__ == "__main__":
    app.run(debug=True)
