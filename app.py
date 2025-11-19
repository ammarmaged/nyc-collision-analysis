import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc 
from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update 
from pathlib import Path

from components import create_filters_grid, get_filter_config_with_year

# Data loading
DATA_PATH = Path(__file__).parent / "integrated_dataset_cleaned.parquet"

def load_data(path: Path):
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

df = load_data(DATA_PATH)

# Get filter config and prepare data with year column
filter_config, df_with_year = get_filter_config_with_year(df)

app = Dash(__name__, external_stylesheets=["assets/styles.css", dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "NYC Motor Vehicle Collisions Analysis"

# --- STYLES ---
TOGGLE_BUTTON_STYLE = {
    "position": "fixed",
    "top": "20px",
    "left": "20px",
    "zIndex": 1001,
}

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "280px",
    "zIndex": 1000,
    "backgroundColor": "#2c3e50", 
    "padding": "20px",
    "overflowY": "auto"
}

SIDEBAR_HIDDEN = {
    **SIDEBAR_STYLE,
    "transform": "translateX(-100%)",
    "transition": "transform 0.3s ease-in-out"
}

CONTENT_STYLE = {
    "marginLeft": "0px",
    "minHeight": "100vh",
    "transition": "margin-left 0.3s ease-in-out",
    "padding": "20px"
}

CONTENT_WITH_SIDEBAR = {
    **CONTENT_STYLE,
    "marginLeft": "280px"
}

# --- LAYOUT ---
app.layout = html.Div([
    
    # Toggle button
    html.Button(
        "☰ Filters",
        id="toggle-button",
        style=TOGGLE_BUTTON_STYLE,
        n_clicks=0
    ),
    
    # Sidebar
    html.Div([
        html.Div("NYC Collisions", className="sidebar-title", style={"color": "white", "fontSize": "24px", "marginBottom": "20px"}),
        
        # Generate Report Button
        dbc.Button(
            "Generate Report",
            id="generate-report-btn",
            color="success", 
            className="mb-3 w-100", 
            n_clicks=0
        ),
        html.Hr(style={"borderTop": "1px solid white"}),

        html.H4("Filters", className="filters-title", style={"color": "white"}),
        
        # Search input
        html.Div([
            dcc.Input(
                id="search-input",
                type="text",
                placeholder="Search (e.g. 2011, Manhattan)...",
                debounce=True,
                style={"width": "100%", "padding": "8px", "marginBottom": "12px", "borderRadius": "4px"}
            )
        ], style={"marginBottom": "12px"}),

        # Filters Grid
        create_filters_grid(df_with_year, filter_config, sidebar=True)

    ], id="sidebar", style=SIDEBAR_STYLE),
    
    # Main content
    html.Div([
        html.H1("NYC Motor Vehicle Collisions Analysis"),
        html.P("Select filters or use the search bar, then click 'Generate Report' to view insights."),
        
        # Report Display Area (A standard Div, initially hidden)
        html.Div(
            id="report-display-area",
            style={
                'display': 'none', 
                'padding': '20px', 
                'marginTop': '20px', 
                'border': '1px solid #ccc', 
                'borderRadius': '5px', 
                'backgroundColor': '#fff'
            }
        ),
        
        html.Div(id="main-dashboard-charts") 
    ], id="content", style=CONTENT_WITH_SIDEBAR)

], style={"display": "flex", "flexDirection": "column"})


# --- CALLBACKS ---

# 1. Toggle Sidebar (No change)
@app.callback(
    [Output("sidebar", "style"),
     Output("content", "style"),
     Output("toggle-button", "style")],
    [Input("toggle-button", "n_clicks")],
    prevent_initial_call=False
)
def toggle_sidebar(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        sidebar_hidden = {**SIDEBAR_HIDDEN}
        content = {**CONTENT_STYLE, "marginLeft": "0px"}
        button = {**TOGGLE_BUTTON_STYLE, "zIndex": 1001}
        return sidebar_hidden, content, button
    else:
        button = {**TOGGLE_BUTTON_STYLE, "zIndex": 1001}
        return SIDEBAR_STYLE, CONTENT_WITH_SIDEBAR, button

# 2. Search -> Auto-apply filters (No change)
@app.callback(
    [Output(f"{cfg.get('id') or cfg['column'].lower()}-filter", "value", allow_duplicate=True) for cfg in filter_config],
    [Input("search-input", "value")],
    prevent_initial_call="initial_duplicate" 
)
def apply_search_to_filters(search_value):
    if not search_value:
        return [None] * len(filter_config)

    import re
    
    def _format_label(v):
        if v is None: return ""
        try:
            import pandas as _pd
            if isinstance(v, (_pd.Timestamp,)): return v.strftime("%Y-%m-%d %H:%M:%S")
        except Exception: pass
        s = str(v).strip()
        while s.startswith("_"): s = s[1:]
        s = s.replace("_", " ")
        try:
            if "." in s:
                f = float(s)
                if f.is_integer(): return str(int(f))
        except Exception: pass
        return s

    tokens = [t.lower() for t in re.findall(r"[\w]+", str(search_value))]

    def _find_matches_for_column(column_name):
        if df_with_year is None or column_name not in df_with_year.columns:
            return None
        unique_vals = df_with_year[column_name].dropna().unique()
        candidates = []
        for v in unique_vals:
            label = _format_label(v).lower()
            words = label.split()
            candidates.append((label, words, v))
        matched = []
        for token in tokens:
            for label, words, v in candidates:
                if label == token and v not in matched: matched.append(v)
        for token in tokens:
            for label, words, v in candidates:
                if token in words and v not in matched: matched.append(v)
        return matched if matched else None

    outputs = []
    for cfg in filter_config:
        outputs.append(_find_matches_for_column(cfg['column']))
    return outputs


# 3. Expand "All" Selections (No change)
@app.callback(
    [Output(f"{cfg.get('id') or cfg['column'].lower()}-filter", "value", allow_duplicate=True) for cfg in filter_config],
    [Input(f"{cfg.get('id') or cfg['column'].lower()}-filter", "value") for cfg in filter_config],
    prevent_initial_call=True
)
def expand_all_selections(*values):
    outputs = []
    for i, val in enumerate(values):
        col_name = filter_config[i]['column']
        if val is not None:
            vals = val if isinstance(val, list) else [val]
            if "__ALL__" in vals:
                if df_with_year is not None and col_name in df_with_year.columns:
                    all_vals = df_with_year[col_name].dropna().unique().tolist()
                    outputs.append(all_vals)
                else:
                    outputs.append(vals)
            else:
                outputs.append(vals)
        else:
            outputs.append(None)
    return outputs


# 4. Generate Report Callback 
@app.callback(
    [Output("report-display-area", "children", allow_duplicate=True),
     Output("report-display-area", "style", allow_duplicate=True)],
    [Input("generate-report-btn", "n_clicks")],
    [State(f"{cfg.get('id') or cfg['column'].lower()}-filter", "value") for cfg in filter_config],
    prevent_initial_call=True
)
def generate_report_in_page(n_open, *filter_values):
    if n_open is None or n_open == 0:
        return no_update, no_update

    # 1. Filter Data
    filtered_df = df_with_year.copy()
    
    for i, val in enumerate(filter_values):
        col_name = filter_config[i]['column']
        if val: 
            valid_vals = val if isinstance(val, list) else [val]
            # Filter out nulls before applying isin() filter
            filtered_df = filtered_df[filtered_df[col_name].isin(filtered_df[col_name].dropna().unique())]
            filtered_df = filtered_df[filtered_df[col_name].isin(valid_vals)]

    # 2. Calculate Stats
    total_crashes = len(filtered_df)
    total_injured = filtered_df['NUMBER OF PERSONS INJURED'].sum() if 'NUMBER OF PERSONS INJURED' in filtered_df else 0
    total_killed = filtered_df['NUMBER OF PERSONS KILLED'].sum() if 'NUMBER OF PERSONS KILLED' in filtered_df else 0

    # 3. Generate Visualizations
    charts = []
    
    # NEW Chart A: Scatter Map (using LATITUDE and LONGITUDE)
    map_data = filtered_df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    
    # PERFORMANCE FIX: Sample the data if the filtered set is too large
    MAX_MAP_POINTS = 10000 
    if len(map_data) > MAX_MAP_POINTS:
        map_data = map_data.sample(MAX_MAP_POINTS, random_state=42)
    
    if not map_data.empty:
        # Changed to px.scatter_mapbox for dot visualization
        fig_map = px.scatter_mapbox(
            map_data, 
            lat='LATITUDE', 
            lon='LONGITUDE', 
            # Marker configuration is now via update_traces
            mapbox_style="open-street-map", 
            zoom=9,
            center=dict(lat=40.730610, lon=-73.935242), # Center of NYC (approx)
            title="Collision Location Map (Sampled)" if len(filtered_df) > MAX_MAP_POINTS else "Collision Location Map"
        )
        fig_map.update_traces(
            marker=dict(size=5, opacity=0.7, color='red'), # Sets size, opacity, and color for the 'dots'
            selector=dict(mode='markers')
        )
        fig_map.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
        )
        charts.append(dcc.Graph(figure=fig_map))
    else:
        charts.append(html.Div(dbc.Alert("No valid LATITUDE/LONGITUDE data available for the map.", color="warning"), style={'height': '400px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}))


    # Chart B: Line Chart (Crashes by Year)
    if 'YEAR' in filtered_df.columns:
        year_counts = filtered_df['YEAR'].value_counts().sort_index().reset_index()
        year_counts.columns = ['Year', 'Crashes']
        fig_year = px.line(year_counts, x='Year', y='Crashes', title="Crashes Trend by Year", markers=True)
        charts.append(dcc.Graph(figure=fig_year))

    # Chart C: Bar Chart (Top Contributing Factors)
    factor_column = 'CONTRIBUTING FACTOR VEHICLE 1'
    if factor_column in filtered_df.columns:
        # Filter out 'UNSPECIFIED' and 'UNKNOWN' factors (case-insensitive)
        non_unspecified_factors = filtered_df[
            ~filtered_df[factor_column].str.upper().str.contains("UNSPECIFIED|UNKNOWN", na=False)
        ]
        
        factor_counts = non_unspecified_factors[factor_column].value_counts().head(5).reset_index()
        factor_counts.columns = ['Factor', 'Count']
        
        fig_factor = px.bar(
            factor_counts, 
            x='Factor', 
            y='Count', 
            # Title is now clean as requested
            title='Top 5 Contributing Factors',
            color='Factor'
        )
        charts.append(dcc.Graph(figure=fig_factor))

    # Chart D: Pie Chart (Injury Type Distribution)
    injury_column = 'PERSON_INJURY'
    if injury_column in filtered_df.columns:
        # Filter out 'NO APPARENT INJURY' to make the breakdown meaningful
        injury_counts_filtered = filtered_df[
            filtered_df[injury_column] != 'NO APPARENT INJURY'
        ][injury_column].value_counts().reset_index()
        
        injury_counts_filtered.columns = ['Injury Type', 'Count']
        
        if not injury_counts_filtered.empty:
            fig_pie = px.pie(
                injury_counts_filtered, 
                names='Injury Type', 
                values='Count', 
                title='Distribution of Person Injury Types (Excluding No Injury)'
            )
            charts.append(dcc.Graph(figure=fig_pie))


    # 4. Construct Report Layout
    report_content = html.Div([
        
        # Hide Report Button 
        dbc.Row(
            dbc.Col(
                dbc.Button("Hide Report", id="hide-report-div-btn", color="secondary", className="float-end"), 
                width={"size": 3, "offset": 9}
            ), className="mb-3"
        ),
        
        html.H3("Collision Analysis Report", className="mb-4"),
        html.H4("Summary Statistics", className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Total Crashes"),
                dbc.CardBody(html.H2(f"{total_crashes:,}"))
            ], color="primary", inverse=True), md=4),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Persons Injured"),
                dbc.CardBody(html.H2(f"{int(total_injured):,}"))
            ], color="warning", inverse=True), md=4),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Persons Killed"),
                dbc.CardBody(html.H2(f"{int(total_killed):,}"))
            ], color="danger", inverse=True), md=4),
        ], className="mb-4"),
        
        html.Hr(),
        html.H4("Visualizations", className="mb-3"),
        
        # Layout: 2x2 grid for the four charts (Map, Line, Bar, Pie)
        dbc.Row([
            dbc.Col(charts[0], md=6),
            dbc.Col(charts[1], md=6),
        ], className="mb-4") if len(charts) >= 2 else None,
        
        dbc.Row([
            dbc.Col(charts[2], md=6),
            dbc.Col(charts[3], md=6),
        ]) if len(charts) >= 4 else html.Div(charts[2]) if len(charts) == 3 else None,

    ])

    # Return the content and the style to make the Div visible
    return report_content, {
        'display': 'block', 
        'padding': '20px', 
        'marginTop': '20px', 
        'border': '1px solid #ccc', 
        'borderRadius': '5px', 
        'backgroundColor': '#fff'
    }


# 5. Callback to hide the report when the internal button is clicked (No change)
@app.callback(
    [Output("report-display-area", "children", allow_duplicate=True),
     Output("report-display-area", "style", allow_duplicate=True)],
    [Input("hide-report-div-btn", "n_clicks")],
    prevent_initial_call=True
)
def hide_report_in_page(n_close):
    if n_close and n_close > 0:
        # Return None for children and the style to hide the Div
        return None, {'display': 'none'}
    return no_update, no_update


if __name__ == "__main__":
    app.run(debug=True)