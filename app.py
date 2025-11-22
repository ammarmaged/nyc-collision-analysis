import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, no_update
from pathlib import Path
import pickle

# Import simple components
from components import create_filters_grid, FILTER_COLUMNS

# =========================
# Data loading (Fast Path)
# =========================

# Uses your EXISTING file name
DATA_PATH = Path(__file__).parent / "integrated_dataset_cleaned_final.parquet"
STATIC_PATH = Path(__file__).parent / "static_results.pkl"

def load_data():
    try:
        # Load the main dataset
        df = pd.read_parquet(DATA_PATH)
        
        # Ensure time columns exist for the dashboard features (Fast if data is already categorical)
        if "CRASH_DATETIME" in df.columns:
             if not pd.api.types.is_datetime64_any_dtype(df["CRASH_DATETIME"]):
                 df["CRASH_DATETIME"] = pd.to_datetime(df["CRASH_DATETIME"], errors="coerce")
             
             # Check if we need to create these, or if they are already in your cleaned file
             if "YEAR" not in df.columns: df["YEAR"] = df["CRASH_DATETIME"].dt.year
             if "MONTH" not in df.columns: df["MONTH"] = df["CRASH_DATETIME"].dt.month_name().astype("category")
             if "HOUR" not in df.columns: df["HOUR"] = df["CRASH_DATETIME"].dt.hour
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def load_static_results():
    try:
        with open(STATIC_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        print("Static results file not found. RQs will be empty.")
        return {}

# Load Data
df_with_year = load_data()
static_data = load_static_results()

# Build Filter Config
filter_config = FILTER_COLUMNS.copy()
if "YEAR" in df_with_year.columns:
    filter_config.insert(0, {"column": "YEAR", "label": "Year"})

# =========================
# Styling Helper
# =========================

def style_figure(fig, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f5f5f7"),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    if title is not None:
        fig.update_layout(title=dict(text=title, font=dict(size=16)))
    return fig

# =========================
# REBUILD STATIC FIGURES (From Pre-calc Data)
# =========================

def build_static_figures(data_dict):
    figs = {}
    
    # RQ1
    if "rq1" in data_dict:
        f = px.bar(data_dict["rq1"], x="BOROUGH", y="CrashCount", text="STREET NAME", color="BOROUGH", title="RQ1 – Street with Most Crashes per Borough (All Time)")
        f.update_traces(textposition="auto")
        figs["rq1"] = style_figure(f, "RQ1 – Street with Most Crashes per Borough (All Time)")
    else: figs["rq1"] = None

    # RQ2
    if "rq2" in data_dict:
        f = px.pie(data_dict["rq2"], names="DAY_OF_WEEK", values="CrashCount", title="RQ2 – Crash Distribution by Day of Week")
        figs["rq2"] = style_figure(f, "RQ2 – Crash Distribution by Day of Week")
    else: figs["rq2"] = None

    # RQ3
    if "rq3" in data_dict:
        df3 = data_dict["rq3"].copy()
        df3["BODILY_INJURY"] = df3["BODILY_INJURY"].astype(str)
        f = px.density_heatmap(df3, x="BODILY_INJURY", y="Status", z="Count", text_auto=True, title="RQ3 – Heatmap: Bodily Injuries Resulting in Death", color_continuous_scale="Reds")
        figs["rq3"] = style_figure(f, "RQ3 – Heatmap: Bodily Injuries Resulting in Death")
    else: figs["rq3"] = None

    # RQ4
    if "rq4" in data_dict:
        df4 = data_dict["rq4"].copy()
        df4["POSITION_IN_VEHICLE"] = df4["POSITION_IN_VEHICLE"].astype(str)
        f = px.density_heatmap(df4, x="POSITION_IN_VEHICLE", y="Status", z="Count", text_auto=True, title="RQ4 – Heatmap: Vehicle Positions Resulting in Death", color_continuous_scale="Reds")
        f.update_xaxes(tickangle=45)
        figs["rq4"] = style_figure(f, "RQ4 – Heatmap: Vehicle Positions Resulting in Death")
    else: figs["rq4"] = None

    # RQ5
    if "rq5" in data_dict:
        f = px.bar(data_dict["rq5"], x="AgeGroup", y="Count", color="TIME_OF_DAY", barmode="group", title="RQ5 – Driver Age Groups vs Day/Night Collisions")
        figs["rq5"] = style_figure(f, "RQ5 – Driver Age Groups vs Day/Night Collisions")
    else: figs["rq5"] = None

    # RQ6
    if "rq6" in data_dict:
        f = px.bar(data_dict["rq6"], y="COMPLAINT", x="Count", orientation='h', title="RQ6 – Top Complaints for Injured Pedestrians", color="Count", color_continuous_scale="Magma")
        figs["rq6"] = style_figure(f, "RQ6 – Top Complaints for Injured Pedestrians")
    else: figs["rq6"] = None

    # RQ7
    if "rq7" in data_dict:
        df7 = data_dict["rq7"].copy()
        df7["PERSON_SEX"] = df7["PERSON_SEX"].astype(str)
        df7["CONTRIBUTING FACTOR VEHICLE 1"] = df7["CONTRIBUTING FACTOR VEHICLE 1"].astype(str)
        f = px.bar(df7, x="PERSON_SEX", y="Count", color="CONTRIBUTING FACTOR VEHICLE 1", text="CONTRIBUTING FACTOR VEHICLE 1", title="RQ7 – Top 5 Contributing Factors by Sex")
        f.update_traces(textposition="inside", texttemplate="%{text}<br>(%{y})")
        figs["rq7"] = style_figure(f, "RQ7 – Top 5 Contributing Factors by Sex")
    else: figs["rq7"] = None

    # RQ8
    if "rq8" in data_dict:
        df8 = data_dict["rq8"].copy()
        df8["SAFETY_EQUIPMENT"] = df8["SAFETY_EQUIPMENT"].astype(str)
        f = px.treemap(df8, path=["SAFETY_EQUIPMENT"], values="Count", title="RQ8 – Safety Equipment Used by Drivers", color="Count", color_continuous_scale="Teal")
        figs["rq8"] = style_figure(f, "RQ8 – Safety Equipment Used by Drivers")
    else: figs["rq8"] = None

    # RQ9
    if "rq9" in data_dict:
        f = px.funnel(data_dict["rq9"], y="BODILY_INJURY", x="Count", title="RQ9 – Most Common Bodily Injuries for Pedestrians", color="Count", color_discrete_sequence=px.colors.sequential.Plasma)
        figs["rq9"] = style_figure(f, "RQ9 – Most Common Bodily Injuries for Pedestrians")
    else: figs["rq9"] = None

    # RQ10
    if "rq10" in data_dict:
        f = px.line(data_dict["rq10"], x="HOUR", y="CrashCount", title="RQ10 – Crash Frequency by Hour of Day", markers=True)
        f.update_traces(line_color="#00d2ff", line_width=3)
        f.update_xaxes(tickmode='linear', tick0=0, dtick=1)
        figs["rq10"] = style_figure(f, "RQ10 – Crash Frequency by Hour of Day")
    else: figs["rq10"] = None
    
    return figs

STATIC_FIGS = build_static_figures(static_data)


# =========================
# Dash app Setup
# =========================

app = Dash(
    __name__,
    external_stylesheets=["assets/styles.css", dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

server = app.server

app.title = "NYC Motor Vehicle Collisions"

TOGGLE_BUTTON_STYLE = {"position": "fixed", "top": "20px", "right": "24px", "left": "auto", "zIndex": 1100}
SIDEBAR_STYLE = {"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "280px", "zIndex": 1000, "padding": "20px", "overflowY": "auto"}
SIDEBAR_HIDDEN = {**SIDEBAR_STYLE, "transform": "translateX(-100%)", "transition": "transform 0.3s ease-in-out"}
CONTENT_STYLE = {"marginLeft": "0px", "minHeight": "100vh", "transition": "margin-left 0.3s ease-in-out", "padding": "24px"}
CONTENT_WITH_SIDEBAR = {**CONTENT_STYLE, "marginLeft": "280px"}

app.layout = html.Div(
    id="app",
    children=[
        html.Button("☰ Filters", id="toggle-button", style=TOGGLE_BUTTON_STYLE, n_clicks=0),
        html.Div(
            id="sidebar",
            style=SIDEBAR_STYLE,
            children=[
                html.Div("NYC Collisions", className="sidebar-title"),
                dbc.Button("Generate Report", id="generate-report-btn", color="success", className="mb-3 w-100 ios-primary-btn", n_clicks=0),
                html.Hr(className="sidebar-divider"),
                html.H4("Filters", className="filters-title"),
                html.Div([dcc.Input(id="search-input", type="text", placeholder="Try: Brooklyn 2022...", debounce=True, className="ios-search-input")], style={"marginBottom": "12px"}),
                create_filters_grid(df_with_year, filter_config, sidebar=True),
            ],
        ),
        html.Div(
            id="content",
            style=CONTENT_WITH_SIDEBAR,
            children=[
                html.Div(className="ios-top-bar", children=[html.Div("NYC Collision Insight", className="ios-top-title"), html.Div("Live · Interactive · Filtered", className="ios-top-sub")]),
                html.Div(
                    id="main-content",
                    children=[
                        html.H1("NYC Motor Vehicle Collisions Analysis"),
                        html.P("Use filters or search, then tap 'Generate Report'.", className="lead-text"),
                        html.Div(id="report-display-area", className="report-card", style={"display": "none"}),
                        html.Div(id="main-dashboard-charts"),
                    ],
                ),
            ],
        ),
    ],
)

@app.callback(
    [Output("sidebar", "style"), Output("content", "style"), Output("toggle-button", "style")],
    [Input("toggle-button", "n_clicks")],
    prevent_initial_call=False,
)
def toggle_sidebar(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return {**SIDEBAR_HIDDEN}, {**CONTENT_STYLE, "marginLeft": "0px"}, {**TOGGLE_BUTTON_STYLE, "zIndex": 1001}
    else:
        return SIDEBAR_STYLE, CONTENT_WITH_SIDEBAR, {**TOGGLE_BUTTON_STYLE, "zIndex": 1001}

@app.callback(
    [Output(f"{cfg.get('id') or cfg['column'].lower()}-filter", "value", allow_duplicate=True) for cfg in filter_config],
    [Input("search-input", "value")],
    prevent_initial_call="initial_duplicate",
)
def apply_search_to_filters(search_value):
    if not search_value:
        return [None] * len(filter_config)
    import re
    tokens = [t.lower() for t in re.findall(r"[\w]+", str(search_value))]
    def _find_matches_for_column(column_name):
        if df_with_year is None or column_name not in df_with_year.columns: return None
        # Only check unique values, super fast
        unique_vals = df_with_year[column_name].dropna().unique()
        matched = []
        for v in unique_vals:
            label = str(v).lower()
            if any(t in label for t in tokens):
                matched.append(v)
        return matched if matched else None
    return [_find_matches_for_column(cfg["column"]) for cfg in filter_config]

@app.callback(
    [Output(f"{cfg.get('id') or cfg['column'].lower()}-filter", "value", allow_duplicate=True) for cfg in filter_config],
    [Input(f"{cfg.get('id') or cfg['column'].lower()}-filter", "value") for cfg in filter_config],
    prevent_initial_call=True,
)
def expand_all_selections(*values):
    outputs = []
    for i, val in enumerate(values):
        col_name = filter_config[i]["column"]
        if val and "__ALL__" in (val if isinstance(val, list) else [val]):
            outputs.append(df_with_year[col_name].dropna().unique().tolist() if df_with_year is not None else val)
        else:
            outputs.append(val)
    return outputs

@app.callback(
    [Output("report-display-area", "children", allow_duplicate=True), Output("report-display-area", "style", allow_duplicate=True)],
    [Input("generate-report-btn", "n_clicks")],
    [State(f"{cfg.get('id') or cfg['column'].lower()}-filter", "value") for cfg in filter_config],
    prevent_initial_call=True,
)
def generate_report_in_page(n_open, *filter_values):
    if n_open is None or n_open == 0:
        return no_update, no_update

    try:
        # 1. Optimized Filtering (NO COPIES)
        mask = pd.Series([True] * len(df_with_year), index=df_with_year.index)
        
        filter_active = False
        for i, val in enumerate(filter_values):
            if val:
                filter_active = True
                col_name = filter_config[i]["column"]
                valid_vals = val if isinstance(val, list) else [val]
                col_mask = df_with_year[col_name].isin(valid_vals)
                if pd.Series([v is None for v in valid_vals]).any():
                     col_mask |= df_with_year[col_name].isna()
                mask &= col_mask

        # Create Views/Slices (Not deep copies)
        filtered_df = df_with_year[mask] if filter_active else df_with_year
        
        if filtered_df.empty:
            return html.Div([dbc.Alert("No data available for filters.", color="warning")]), {"display": "block"}

        # Deduplicate collisions for crash-level charts
        # Use boolean indexing to create a crash mask instead of a full drop_duplicates copy if possible
        # For safety and existing logic, we do drop_duplicates but limit columns if possible, 
        # but here we stick to standard drop_duplicates to ensure logic consistency.
        crash_level_df = filtered_df.drop_duplicates(subset=['COLLISION_ID'], keep='first')
        
        # Stats (Aggregation)
        total_crashes = len(crash_level_df)
        total_injured = crash_level_df["NUMBER OF PERSONS INJURED"].sum() if "NUMBER OF PERSONS INJURED" in crash_level_df.columns else 0
        total_killed = crash_level_df["NUMBER OF PERSONS KILLED"].sum() if "NUMBER OF PERSONS KILLED" in crash_level_df.columns else 0

        overview_charts = []
        
        # MAP (Dynamic)
        if {"LATITUDE", "LONGITUDE"}.issubset(crash_level_df.columns):
            map_data = crash_level_df.dropna(subset=["LATITUDE", "LONGITUDE"])
            if len(map_data) > 5000: map_data = map_data.sample(5000, random_state=42)
            if not map_data.empty:
                fig_map = px.scatter_map(
                    map_data, lat="LATITUDE", lon="LONGITUDE", map_style="carto-darkmatter", zoom=9,
                    center=dict(lat=40.730610, lon=-73.935242), title="Collision Location Map"
                )
                fig_map.update_traces(marker=dict(size=5, opacity=0.7, color="red"))
                overview_charts.append(dcc.Graph(figure=style_figure(fig_map), className="ios-chart"))

        # LINE (Year) - Aggregated
        if "YEAR" in crash_level_df.columns:
            # groupby().size() is much faster than counting straight in plotting
            year_counts = crash_level_df.groupby("YEAR", observed=True).size().reset_index(name="Crashes")
            fig_year = px.line(year_counts, x="YEAR", y="Crashes", title="Crashes Trend by Year", markers=True)

        # BAR (Month) - Aggregated
        if "MONTH" in crash_level_df.columns:
            month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            month_counts = crash_level_df.groupby("MONTH", observed=True).size().reindex(month_order).reset_index(name="Crashes")
            fig_month = px.bar(month_counts, x="MONTH", y="Crashes", title="Crashes by Month", color="Crashes")

        if "YEAR" in crash_level_df.columns and "MONTH" in crash_level_df.columns:
             overview_charts.append(dbc.Row([
                 dbc.Col(dcc.Graph(figure=style_figure(fig_year), className="ios-chart"), md=6),
                 dbc.Col(dcc.Graph(figure=style_figure(fig_month), className="ios-chart"), md=6)
             ], className="mb-4"))

        # LINE (Hour) - Aggregated
        if "HOUR" in crash_level_df.columns:
            hour_counts = crash_level_df.groupby("HOUR", observed=True).size().reset_index(name="Crashes")
            fig_hour = px.line(hour_counts, x="HOUR", y="Crashes", title="Crashes by Hour of Day (Filtered)", markers=True)
            fig_hour.update_xaxes(tickmode='linear', tick0=0, dtick=1)
            overview_charts.append(dcc.Graph(figure=style_figure(fig_hour), className="ios-chart"))

        # BAR (Top Factors) - Aggregated
        factor_col = "CONTRIBUTING FACTOR VEHICLE 1"
        if factor_col in filtered_df.columns:
            # Filter out bad values efficiently
            factors = filtered_df[~filtered_df[factor_col].astype(str).str.upper().str.contains("UNSPECIFIED|UNKNOWN", na=False)]
            if not factors.empty:
                f_counts = factors.groupby(factor_col, observed=True).size().reset_index(name="Count").sort_values("Count", ascending=False).head(5)
                fig_f = px.bar(f_counts, x=factor_col, y="Count", title="Top 5 Contributing Factors (Excl. Unspecified)", color=factor_col)
                overview_charts.append(dcc.Graph(figure=style_figure(fig_f), className="ios-chart"))
        
        # PIE (Sex) - Aggregated
        if "PERSON_SEX" in filtered_df.columns:
            sex_counts = filtered_df.groupby("PERSON_SEX", observed=True).size().reset_index(name="Count")
            fig_pie = px.pie(sex_counts, names="PERSON_SEX", values="Count", title="Person Sex Distribution (Filtered)")
            overview_charts.append(dcc.Graph(figure=style_figure(fig_pie), className="ios-chart"))

        # BAR (Streets) - Aggregated
        if "BOROUGH" in crash_level_df.columns and "STREET NAME" in crash_level_df.columns:
            street_data = crash_level_df[crash_level_df["STREET NAME"].astype(str).str.strip() != ""]
            street_counts = street_data.groupby(["BOROUGH", "STREET NAME"], observed=True).size().reset_index(name="Crashes")
            top_streets = street_counts.sort_values(["BOROUGH", "Crashes"], ascending=[True, False]).groupby("BOROUGH", observed=True).head(3).reset_index(drop=True)
            
            if not top_streets.empty:
                top_streets["Rank"] = top_streets.groupby("BOROUGH").cumcount().astype(str)
                fig_streets = px.bar(top_streets, x="BOROUGH", y="Crashes", color="Rank", barmode="group", text="STREET NAME", title="Top 3 High-Crash Streets per Borough (Side-by-Side)")
                fig_streets.update_traces(textposition="inside", textangle=-90)
                fig_streets.update_layout(showlegend=False)
                overview_charts.append(dcc.Graph(figure=style_figure(fig_streets), className="ios-chart"))

        # DOUGHNUT (Borough) - Aggregated
        if "BOROUGH" in crash_level_df.columns:
            boro_counts = crash_level_df.groupby("BOROUGH", observed=True).size().reset_index(name="Crashes")
            boro_counts["BOROUGH"] = boro_counts["BOROUGH"].astype(str)
            fig_doughnut = px.pie(boro_counts, names="BOROUGH", values="Crashes", title="Crash Distribution by Borough", hole=0.4, color_discrete_sequence=px.colors.qualitative.Bold)
            fig_doughnut.update_traces(textinfo="percent+label")
            overview_charts.append(dcc.Graph(figure=style_figure(fig_doughnut), className="ios-chart"))

        # HEATMAP (Position vs Bodily Injury) - Aggregated
        if "POSITION_IN_VEHICLE" in filtered_df.columns and "BODILY_INJURY" in filtered_df.columns:
             bad_vals = ["UNKNOWN", "DOES NOT APPLY", "N/A", "NOT APPLICABLE"]
             hm_data = filtered_df[~filtered_df["POSITION_IN_VEHICLE"].astype(str).str.upper().isin(bad_vals)]
             hm_data = hm_data[~hm_data["BODILY_INJURY"].astype(str).str.upper().isin(bad_vals)]
             
             if not hm_data.empty:
                 # Aggregate BEFORE passing to Plotly
                 hm_data_grouped = hm_data.groupby(["POSITION_IN_VEHICLE", "BODILY_INJURY"], observed=True).size().reset_index(name="Count")
                 # Filter top 10 to match original logic visually
                 top_pos = hm_data_grouped.groupby("POSITION_IN_VEHICLE")["Count"].sum().nlargest(10).index
                 top_bod = hm_data_grouped.groupby("BODILY_INJURY")["Count"].sum().nlargest(10).index
                 hm_data_grouped = hm_data_grouped[hm_data_grouped["POSITION_IN_VEHICLE"].isin(top_pos) & hm_data_grouped["BODILY_INJURY"].isin(top_bod)]
                 
                 hm_data_grouped["POSITION_IN_VEHICLE"] = hm_data_grouped["POSITION_IN_VEHICLE"].astype(str)
                 hm_data_grouped["BODILY_INJURY"] = hm_data_grouped["BODILY_INJURY"].astype(str)

                 fig_hm = px.density_heatmap(hm_data_grouped, x="POSITION_IN_VEHICLE", y="BODILY_INJURY", z="Count", text_auto=True, title="Heatmap: Position vs Bodily Injury (Filtered)", color_continuous_scale="Viridis")
                 overview_charts.append(dcc.Graph(figure=style_figure(fig_hm), className="ios-chart"))

        # TREEMAP (Safety) - Aggregated
        if "PERSON_TYPE" in filtered_df.columns and "SAFETY_EQUIPMENT" in filtered_df.columns:
             sb_data = filtered_df[~filtered_df["PERSON_TYPE"].astype(str).str.upper().isin(["UNKNOWN"])]
             sb_data = sb_data[~sb_data["SAFETY_EQUIPMENT"].astype(str).str.upper().isin(["UNKNOWN", "NONE", "OTHER"])]
             
             if not sb_data.empty:
                 sb_counts = sb_data.groupby(["PERSON_TYPE", "SAFETY_EQUIPMENT"], observed=True).size().reset_index(name="Count")
                 top_pt = sb_counts.groupby("PERSON_TYPE")["Count"].sum().nlargest(10).index
                 sb_counts = sb_counts[sb_counts["PERSON_TYPE"].isin(top_pt)]
                 
                 sb_counts["PERSON_TYPE"] = sb_counts["PERSON_TYPE"].astype(str)
                 sb_counts["SAFETY_EQUIPMENT"] = sb_counts["SAFETY_EQUIPMENT"].astype(str)

                 fig_sb = px.treemap(sb_counts, path=['PERSON_TYPE', 'SAFETY_EQUIPMENT'], values='Count', color='Count', title="Safety Equipment by Person Type (Filtered)", color_continuous_scale="Teal")
                 fig_sb.update_traces(textinfo="label+value+percent parent")
                 overview_charts.append(dcc.Graph(figure=style_figure(fig_sb), className="ios-chart"))

        # TREEMAP (Injury) - Aggregated
        if "PERSON_TYPE" in filtered_df.columns and "BODILY_INJURY" in filtered_df.columns:
             exclude_vals = ["UNKNOWN", "DOES NOT APPLY", "N/A", "NOT APPLICABLE", "UNSPECIFIED"]
             bi_data = filtered_df[~filtered_df["PERSON_TYPE"].astype(str).str.upper().isin(exclude_vals)]
             bi_data = bi_data[~bi_data["BODILY_INJURY"].astype(str).str.upper().isin(exclude_vals)]
             
             if not bi_data.empty:
                 bi_counts = bi_data.groupby(["PERSON_TYPE", "BODILY_INJURY"], observed=True).size().reset_index(name="Count")
                 bi_counts["PERSON_TYPE"] = bi_counts["PERSON_TYPE"].astype(str)
                 bi_counts["BODILY_INJURY"] = bi_counts["BODILY_INJURY"].astype(str)

                 fig_bi = px.treemap(bi_counts, path=['PERSON_TYPE', 'BODILY_INJURY'], values='Count', color='Count', title="Bodily Injury by Person Type (Filtered)", color_continuous_scale="Reds")
                 fig_bi.update_traces(textinfo="label+value+percent parent")
                 overview_charts.append(dcc.Graph(figure=style_figure(fig_bi), className="ios-chart"))

        # SIDE-BY-SIDE HEATMAPS - Aggregated
        if {"PERSON_INJURY", "BODILY_INJURY", "POSITION_IN_VEHICLE"}.issubset(filtered_df.columns):
             exclude_vals = ["UNKNOWN", "DOES NOT APPLY", "N/A", "NOT APPLICABLE", "UNSPECIFIED"]
             row_content = []

             # Heatmap 1
             h1_data = filtered_df[~filtered_df["PERSON_INJURY"].astype(str).str.upper().isin(exclude_vals)]
             h1_data = h1_data[~h1_data["BODILY_INJURY"].astype(str).str.upper().isin(exclude_vals)]
             
             if not h1_data.empty:
                 h1_agg = h1_data.groupby(["PERSON_INJURY", "BODILY_INJURY"], observed=True).size().reset_index(name="Count")
                 top_pi = h1_agg.groupby("PERSON_INJURY")["Count"].sum().nlargest(15).index
                 top_bi = h1_agg.groupby("BODILY_INJURY")["Count"].sum().nlargest(15).index
                 h1_agg = h1_agg[h1_agg["PERSON_INJURY"].isin(top_pi) & h1_agg["BODILY_INJURY"].isin(top_bi)]
                 fig_h1 = px.density_heatmap(h1_agg, x="BODILY_INJURY", y="PERSON_INJURY", z="Count", text_auto=True, title="Person Injury vs Bodily Injury", color_continuous_scale="Blues")
                 row_content.append(dbc.Col(dcc.Graph(figure=style_figure(fig_h1), className="ios-chart"), md=6))

             # Heatmap 2
             h2_data = filtered_df[~filtered_df["PERSON_INJURY"].astype(str).str.upper().isin(exclude_vals)]
             h2_data = h2_data[~h2_data["POSITION_IN_VEHICLE"].astype(str).str.upper().isin(exclude_vals)]
             
             if not h2_data.empty:
                 h2_agg = h2_data.groupby(["PERSON_INJURY", "POSITION_IN_VEHICLE"], observed=True).size().reset_index(name="Count")
                 top_pi2 = h2_agg.groupby("PERSON_INJURY")["Count"].sum().nlargest(15).index
                 top_pos = h2_agg.groupby("POSITION_IN_VEHICLE")["Count"].sum().nlargest(15).index
                 h2_agg = h2_agg[h2_agg["PERSON_INJURY"].isin(top_pi2) & h2_agg["POSITION_IN_VEHICLE"].isin(top_pos)]
                 fig_h2 = px.density_heatmap(h2_agg, x="POSITION_IN_VEHICLE", y="PERSON_INJURY", z="Count", text_auto=True, title="Person Injury vs Position in Vehicle", color_continuous_scale="Oranges")
                 row_content.append(dbc.Col(dcc.Graph(figure=style_figure(fig_h2), className="ios-chart"), md=6))

             if row_content: overview_charts.append(dbc.Row(row_content, className="mb-4"))

        # SUNBURST - Aggregated
        if {"PERSON_TYPE", "PERSON_INJURY", "COMPLAINT"}.issubset(filtered_df.columns):
             sb_cols = ["PERSON_TYPE", "PERSON_INJURY", "COMPLAINT"]
             exclude_vals = ["UNKNOWN", "DOES NOT APPLY", "N/A", "NOT APPLICABLE", "UNSPECIFIED", "NONE"]
             
             # Filter efficiently
             mask_sb = ~filtered_df["PERSON_TYPE"].astype(str).str.upper().isin(exclude_vals)
             mask_sb &= ~filtered_df["PERSON_INJURY"].astype(str).str.upper().isin(exclude_vals)
             mask_sb &= ~filtered_df["COMPLAINT"].astype(str).str.upper().isin(exclude_vals)
             sb_data = filtered_df[mask_sb]
             
             if not sb_data.empty:
                 sb_grouped = sb_data.groupby(sb_cols, observed=True).size().reset_index(name="Count")
                 top_complaints = sb_grouped.groupby("COMPLAINT")["Count"].sum().nlargest(20).index
                 sb_grouped = sb_grouped[sb_grouped["COMPLAINT"].isin(top_complaints)]
                 fig_sun = px.sunburst(sb_grouped, path=["PERSON_TYPE", "PERSON_INJURY", "COMPLAINT"], values="Count", title="Hierarchy: Person Type → Injury → Complaint", color="Count", color_continuous_scale="RdBu", branchvalues="total")
                 overview_charts.append(dcc.Graph(figure=style_figure(fig_sun), className="ios-chart"))

        # BOX PLOT (Age) - Raw data is required for true box plot stats, but we filter first
        if "PERSON_AGE" in filtered_df.columns:
             age_data = filtered_df[(filtered_df["PERSON_AGE"] >= 0) & (filtered_df["PERSON_AGE"] <= 110)]
             if not age_data.empty:
                 fig_box = px.box(age_data, y="PERSON_AGE", title="Age Distribution (Filtered)", points="outliers")
                 overview_charts.append(dcc.Graph(figure=style_figure(fig_box), className="ios-chart"))

        # RQs (Static)
        rq_sections = []
        # We use the pre-built global static figures
        questions = {
            "rq1": "RQ1: Which street has the most crashes in each borough?", "rq2": "RQ2: Which day of the week has the most crashes?",
            "rq3": "RQ3: Which bodily injuries most frequently result in death?", "rq4": "RQ4: Which vehicle positions most frequently result in death?",
            "rq5": "RQ5: Are young drivers more involved in night-time crashes?", "rq6": "RQ6: What is the most common complaint for injured pedestrians?",
            "rq7": "RQ7: Top 5 Contributing Factors for Female and Male", "rq8": "RQ8: What safety equipment is most frequently used by drivers?",
            "rq9": "RQ9: What are the most common bodily injuries faced by pedestrians?", "rq10": "RQ10: What is the most dangerous time of day for crashes?"
        }
        
        for key, title in questions.items():
            fig = STATIC_FIGS.get(key)
            if fig:
                rq_sections.append(html.Div([html.H5(title, className="section-title mb-2"), dcc.Graph(figure=fig, className="ios-chart"), html.Small("Note: This visualization is based on the full dataset (All Time).", className="text-muted")], className="mb-4"))
            else:
                rq_sections.append(html.Div(dbc.Alert(f"{key.upper()}: No data available (Static).", color="warning"), className="mb-4"))

        # Build Report
        report_children = [
            dbc.Row([dbc.Col(dbc.Button("Hide Report", id="hide-report-div-btn", color="secondary", className="float-end ios-secondary-btn"), width={"size": 3, "offset": 9})], className="mb-3"),
            html.H3("Collision Analysis Report", className="report-title mb-4"),
            html.H4("Summary Statistics", className="section-title mb-3"),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("Total Crashes"), dbc.CardBody(html.H2(f"{total_crashes:,}"))], className="ios-stat-card ios-stat-primary"), md=4),
                dbc.Col(dbc.Card([dbc.CardHeader("Persons Injured"), dbc.CardBody(html.H2(f"{int(total_injured):,}"))], className="ios-stat-card ios-stat-warning"), md=4),
                dbc.Col(dbc.Card([dbc.CardHeader("Persons Killed"), dbc.CardBody(html.H2(f"{int(total_killed):,}"))], className="ios-stat-card ios-stat-danger"), md=4),
            ], className="mb-4"),
            html.Hr(),
            html.H4("Core Visualizations", className="section-title mb-3"),
        ]
        
        for item in overview_charts:
             if isinstance(item, dbc.Row): report_children.append(item)
             else: report_children.append(html.Div(item, className="mb-4"))

        if rq_sections:
            report_children.append(html.Hr())
            report_children.append(html.H4("Research Question Visualizations", className="section-title mb-3"))
            report_children.extend(rq_sections)

        return html.Div(report_children), {"display": "block"}
    
    except Exception as e:
        error_message = f"Report Generation Failed: {e}. Try clearing filters or refreshing."
        print(f"FATAL CALLBACK ERROR: {e}")
        return html.Div([html.H3("Report Error", className="text-danger"), dbc.Alert(error_message, color="danger"), dbc.Button("Hide Report", id="hide-report-div-btn", color="secondary")]), {"display": "block"}

@app.callback(
    [Output("report-display-area", "children", allow_duplicate=True), Output("report-display-area", "style", allow_duplicate=True)],
    [Input("hide-report-div-btn", "n_clicks")],
    prevent_initial_call=True,
)
def hide_report_in_page(n_close):
    if n_close and n_close > 0:
        return None, {"display": "none"}
    return no_update, no_update

if __name__ == "__main__":
    app.server.run(debug=True)