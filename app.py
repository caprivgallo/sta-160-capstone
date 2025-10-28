import pandas as pd
from dash import Dash, dcc, html, Input, Output

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

# Ensure energy is numeric
data["energy"] = pd.to_numeric(data["energy"], errors="coerce")

# Drop rows with missing numeric values
clean_data = data.dropna()

# --- Initialize app ---
app = Dash(__name__)
server = app.server  # needed for deployment later

# Get list of numeric columns
numeric_cols = clean_data.select_dtypes(include="number").columns

# --- Layout ---
app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "padding": "20px"},
    children=[
        html.H1("🎶 STA 160 Capstone Dashboard", style={"textAlign": "center"}),

        html.Div(
            [
                html.Label("Select X-axis:"),
                dcc.Dropdown(
                    id="x-axis",
                    options=[{"label": col, "value": col} for col in numeric_cols],
                    value="tempo",  # default x-axis
                ),
                html.Label("Select Y-axis:"),
                dcc.Dropdown(
                    id="y-axis",
                    options=[{"label": col, "value": col} for col in numeric_cols],
                    value="energy",  # default y-axis
                ),
            ],
            style={"width": "40%", "margin": "auto"},
        ),

        dcc.Graph(id="scatter-plot", style={"height": "80vh"}),
    ],
)

# --- Callback for interactive updates ---
@app.callback(
    Output("scatter-plot", "figure"),
    Input("x-axis", "value"),
    Input("y-axis", "value"),
)
def update_graph(x_col, y_col):
    fig = {
        "data": [
            {
                "x": clean_data[x_col],
                "y": clean_data[y_col],
                "mode": "markers",
                "type": "scatter",
                "marker": {"opacity": 0.6, "size": 8, "color": "royalblue"},
            }
        ],
        "layout": {
            "title": f"{x_col} vs {y_col}",
            "xaxis": {"title": x_col},
            "yaxis": {"title": y_col},
        },
    }
    return fig

# --- Run server ---
if __name__ == "__main__":
    app.run(debug=True)
