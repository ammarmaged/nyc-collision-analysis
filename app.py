
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# -----------------------------
# Load Parquet file
# -----------------------------
data_path = Path(__file__).parent / "integrated_dataset_cleaned.parquet"
df = pd.read_parquet(data_path)
# Ensure CRASH_DATETIME is datetime type and extract year
df['CRASH_DATETIME'] = pd.to_datetime(df['CRASH_DATETIME'], errors='coerce')
df['CRASH_YEAR'] = df['CRASH_DATETIME'].dt.year

# -----------------------------
# Initialize Dash app
# -----------------------------
app = Dash(__name__)
app.title = "NYC Motor Vehicle Collisions Dashboard"

# -----------------------------
# Layout
# -----------------------------
app.layout = html.Div([
    html.H1("NYC Motor Vehicle Collisions Dashboard", style={'textAlign': 'center'}),
    
    # Filters
    html.Div([
        dcc.Dropdown(
            id="borough-filter",
            options=[{"label": b, "value": b} for b in sorted(df["BOROUGH"].dropna().unique())],
            placeholder="Select Borough",
            clearable=True
        ),
        dcc.Dropdown(
            id="year-filter",
            options=[{"label": str(y), "value": y} for y in sorted(df["CRASH_YEAR"].dropna().unique())],
            placeholder="Select Year",
            clearable=True
        ),
    ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"}),

    # Search Box
    dcc.Input(
        id="search-box",
        type="text",
        placeholder="Search (e.g., 'Brooklyn pedestrian')",
        style={"width": "400px", "marginBottom": "20px"}
    ),
    
    # Generate Report Button
    html.Button("Generate Report", id="generate-btn", n_clicks=0, style={"marginBottom": "20px"}),

    # Graphs
    dcc.Graph(id="graph1"),
    dcc.Graph(id="graph2")
])

# -----------------------------
# Callback
# -----------------------------
@app.callback(
    [Output("graph1", "figure"),
     Output("graph2", "figure")],
    [Input("generate-btn", "n_clicks"),
     Input("borough-filter", "value"),
     Input("year-filter", "value"),
     Input("search-box", "value")]
)
def update_report(n, borough, year, search):
    filtered = df.copy()

    # Apply filters
    if borough:
        filtered = filtered[filtered["BOROUGH"] == borough]

    if year:
        filtered = filtered[filtered["CRASH_YEAR"] == year]

    # Search mode
    if search:
        search = search.lower()
        filtered = filtered[
            filtered["BOROUGH"].str.lower().str.contains(search, na=False) |
            filtered["CONTRIBUTING_FACTOR_VEHICLE_1"].str.lower().str.contains(search, na=False)
        ]

    # Graph 1: Crashes per Borough
    fig1 = px.histogram(
        filtered, x="BOROUGH", title="Crashes per Borough",
        labels={"BOROUGH": "Borough", "count": "Number of Crashes"}
    )

    # Graph 2: Crashes per Year
    fig2 = px.histogram(
        filtered, x="CRASH_YEAR", title="Crashes per Year",
        labels={"CRASH_YEAR": "Year", "count": "Number of Crashes"}
    )

    return fig1, fig2

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)



# Preview
print(df.head())

app = Dash(__name__)

app.layout = html.Div([
    html.H1("NYC Motor Vehicle Collisions Dashboard"),

    # Filters
    html.Div([
        dcc.Dropdown(
            id="borough-filter",
            options=[{"label": b, "value": b} for b in df["BOROUGH"].dropna().unique()],
            placeholder="Select Borough",
        ),
        dcc.Dropdown(
            id="year-filter",
            options=[{"label": str(y), "value": y} for y in df["CRASH_DATETIME"].unique()],
            placeholder="Select Year",
        ),
    ], style={"display": "flex", "gap": "20px"}),

    # Search Box
    dcc.Input(
        id="search-box",
        type="text",
        placeholder="Search (e.g., 'Brooklyn 2022 pedestrian')",
        style={"width": "400px", "marginTop": "20px"}
    ),
    
    # Generate Report Button
    html.Button("Generate Report", id="generate-btn", n_clicks=0),

    # Graphs
    dcc.Graph(id="graph1"),
    dcc.Graph(id="graph2")
])

@app.callback(
    [Output("graph1", "figure"),
     Output("graph2", "figure")],
    [Input("generate-btn", "n_clicks"),
     Input("borough-filter", "value"),
     Input("year-filter", "value"),
     Input("search-box", "value")]
)
def update_report(n, borough, year, search):
    filtered = df.copy()

    # Apply filters
    if borough:
        filtered = filtered[filtered["BOROUGH"] == borough]

    if year:
        filtered = filtered[filtered["CRASH_YEAR"] == year]

    # Search mode
    if search:
        search = search.lower()
        filtered = filtered[
            filtered["BOROUGH"].str.lower().str.contains(search, na=False)
            | filtered["CONTRIBUTING_FACTOR_VEHICLE_1"].str.lower().str.contains(search, na=False)
        ]

    fig1 = px.histogram(filtered, x="BOROUGH", title="Crashes per Borough")
    fig2 = px.histogram(filtered, x="CRASH_YEAR", title="Crashes per Year")

    return fig1, fig2


if __name__ == "__main__":
    app.run(debug=True, open_browser=True)



