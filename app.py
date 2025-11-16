import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table

# -------------------------------
#  Load & clean data
# -------------------------------

data = pd.read_csv("merged_final_data.csv")

# Clean tempo stored like "[120]"
data["tempo"] = (
    data["tempo"]
    .astype(str)
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .astype(float)
)

data["energy"] = pd.to_numeric(data["energy"], errors="coerce")

# Convert duration to minutes
data["duration_ms"] = data["duration_ms"].apply(lambda x: round(x / 60000, 2))
data.rename(columns={"duration_ms": "duration_min"}, inplace=True)

clean_data = data.dropna(subset=["tempo", "energy"])

artist_options = [
    {"label": artist, "value": artist}
    for artist in sorted(clean_data["Artist"].dropna().unique())
]

# UC Davis Colors
UC_NAVY = "#022851"
UC_GOLD = "#DAA900"
BG_LIGHT = "#f9f7f7"

# -------------------------------
#  Initialize App
# -------------------------------

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "The Science of Song Success"

# -------------------------------
#  Layout
# -------------------------------

app.layout = html.Div(
    style={"backgroundColor": BG_LIGHT, "fontFamily": "Georgia, serif", "padding": "25px"},
    children=[
        # Header Row
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
            },
            children=[
                html.Img(src="/assets/ucdavis_logo.png", style={"height": "70px"}),
                html.H1(
                    "The Science of Song Success",
                    style={
                        "color": UC_NAVY,
                        "fontSize": "38px",
                        "textAlign": "center",
                        "flex": "1",
                        "fontWeight": "bold",
                    },
                ),
            ],
        ),

        html.Div(
            style={
                "borderBottom": f"4px solid {UC_GOLD}",
                "width": "220px",
                "margin": "0 auto",
                "marginBottom": "25px",
            }
        ),

        html.P(
            """
            For our capstone project, we built an interactive dashboard to explore how Spotify audio 
            features and YouTube engagement metrics relate to one another — and how these patterns 
            help explain a song’s success. This dashboard includes summary statistics, a deep learning 
            model overview, key visualizations, and a polished final report summary.
            """,
            style={"fontSize": "17px", "textAlign": "center", "lineHeight": "1.7", "maxWidth": "900px", "margin": "0 auto"},
        ),

        html.Br(),

        # Tabs Navigation
        dcc.Tabs(
            id="tabs",
            value="summary",
            colors={"border": UC_NAVY, "primary": UC_NAVY, "background": BG_LIGHT},
            children=[
                dcc.Tab(label="Summary Statistics", value="summary"),
                dcc.Tab(label="Deep Learning Model", value="model"),
                dcc.Tab(label="Visuals", value="visuals"),
                dcc.Tab(label="Final Report Summary", value="report"),
                dcc.Tab(label="Team & Acknowledgments", value="team"),
            ],
        ),

        html.Div(id="tabs-content"),

        html.Br(),

        # Footer
        html.Div(
            "Developed by Team 13 — STA 160 Capstone | UC Davis, Fall 2025",
            style={
                "textAlign": "center",
                "marginTop": "40px",
                "color": UC_NAVY,
                "fontWeight": "bold",
            },
        ),
    ],
)

# -------------------------------
#  Callback for tabs
# -------------------------------

@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_tab(tab):

    # ---------------------------
    # SUMMARY STATISTICS TAB
    # ---------------------------
    if tab == "summary":
        return html.Div(
            [
                html.H2("Summary Statistics", style={"color": UC_NAVY}),
                html.P(
                    """
                    Below are descriptive summaries of key Spotify and YouTube variables:
                    • tempo — speed of a track in BPM  
                    • energy — intensity (0 to 1)  
                    • loudness — overall volume (in dB)  
                    • duration_min — track length in minutes  
                    • views/likes/comments — listener engagement on YouTube  
                    """,
                    style={"fontSize": "15px", "whiteSpace": "pre-line", "marginBottom": "20px"},
                ),

                html.Label("Filter by Artist:"),
                dcc.Dropdown(id="artist-dropdown", options=artist_options, placeholder="Select an artist"),

                html.Br(),

                dash_table.DataTable(
                    id="summary-table",
                    columns=[
                        {"name": col, "id": col}
                        for col in ["Artist", "Views", "Likes", "Comments", "tempo", "energy", "duration_min"]
                    ],
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": UC_NAVY,
                        "color": "white",
                        "fontWeight": "bold",
                    },
                    style_data={
                        "backgroundColor": "white",
                        "color": "black",
                    },
                ),
            ]
        )

    # ---------------------------
    # MODEL TAB
    # ---------------------------
    elif tab == "model":
        return html.Div(
            [
                html.H2("Deep Learning Model Overview", style={"color": UC_NAVY}),
                html.P(
                    """
                    This section summarizes the deep learning model trained to predict 
                    engagement metrics using Spotify acoustic features. Additional 
                    exploration included regression and tree-based models 
                    to compare predictive performance.
                    """,
                    style={"fontSize": "16px"},
                )
            ]
        )

    # ---------------------------
    # VISUALS TAB
    # ---------------------------
    elif tab == "visuals":
        return html.Div(
            [
                html.H2("Visualizations", style={"color": UC_NAVY}),
                html.P("Plots and graphics will appear here.", style={"fontSize": "16px"}),
            ]
        )

    # ---------------------------
    # FINAL REPORT SUMMARY TAB
    # ---------------------------
    elif tab == "report":
        return html.Div(
            style={
                "backgroundColor": "white",
                "padding": "25px",
                "borderRadius": "10px",
                "boxShadow": "0px 2px 6px rgba(0,0,0,0.1)",
            },
            children=[
                html.H2("Final Report Summary", style={"color": UC_NAVY}),

                html.P(
                    """
                    This section provides a structured summary of our full capstone report, including 
                    motivation, data preparation, modeling choices, findings, and future work.
                    """,
                    style={"fontSize": "16px"},
                ),

                html.Br(),

                html.H3("Download Full Report", style={"color": UC_NAVY}),
                html.Div(
                    [
                        html.A(
                            "📄 Click here to open the Full Final Report",
                            href="https://docs.google.com/document/d/1wYoX1HOKK6wEcU_JRXqQMJ0NLWfyZAwa5gW9crADFms/edit?usp=sharing",
                            target="_blank",
                            style={
                                "backgroundColor": UC_NAVY,
                                "color": "white",
                                "padding": "10px 20px",
                                "borderRadius": "8px",
                                "textDecoration": "none",
                                "fontWeight": "bold",
                                "fontSize": "16px",
                            },
                        )
                    ],
                    style={"textAlign": "center", "marginTop": "20px"},
                ),
            ],
        )

    # ---------------------------
    # TEAM TAB
    # ---------------------------
    elif tab == "team":
        return html.Div(
            [
                html.H2("Team & Acknowledgments", style={"color": UC_NAVY}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4(name, style={"color": UC_NAVY}),
                                html.P("B.S. Statistical Data Science | UC Davis (2026)"),
                            ],
                            style={
                                "backgroundColor": "white",
                                "padding": "15px",
                                "borderRadius": "10px",
                                "margin": "10px",
                                "width": "250px",
                            },
                        )
                        for name in ["Capri Gallo", "Alex Garcia", "Rohan Pillay", "Edward Ron", "Yuxiao Tan"]
                    ],
                    style={"display": "flex", "flexWrap": "wrap"},
                ),

                html.H3("Acknowledgments", style={"color": UC_NAVY}),
                html.P("Special thanks to Professor Lingfei Cui and the UC Davis Statistics Department."),

                html.H3("References", style={"color": UC_NAVY}),
                html.Ul(
                    [
                        html.Li("Li, Y. et al. (2024). MERT: Acoustic Music Understanding Model. arXiv:2306.00107."),
                        html.Li("Grewal, R. (2025). Spotify–YouTube Data. Kaggle."),
                    ]
                ),
            ]
        )

# -------------------------------
#  Run App
# -------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)

