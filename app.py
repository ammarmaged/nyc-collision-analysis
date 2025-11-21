import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, no_update
from pathlib import Path

from components import create_filters_grid, get_filter_config_with_year

# =========================
# Data loading
# =========================

DATA_PATH = Path(__file__).parent / "integrated_dataset_cleaned_final.parquet"


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

# =========================
# Global Plotly dark styling helper
# =================================

def style_figure(fig, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f5f5f7"),
    )
    if title is not None:
        fig.update_layout(title=dict(text=title, font=dict(size=16)))
    return fig


# ==============================================================================
# STATIC DATA CALCULATIONS (RQs 1 - 10)
# Calculated ONCE on full data. These will never disappear due to filters.
# ==============================================================================

# --- RQ1: Most crash-prone street per borough (BAR) ---
STATIC_RQ1_FIG = None
if "STREET NAME" in df_with_year.columns and "BOROUGH" in df_with_year.columns and "COLLISION_ID" in df_with_year.columns:
    static_df_1 = df_with_year.dropna(subset=["STREET NAME", "BOROUGH"])
    static_df_1 = static_df_1[static_df_1["STREET NAME"].astype(str).str.strip() != ""]
    
    street_counts = (
        static_df_1.drop_duplicates(subset=["COLLISION_ID"])
        .groupby(["BOROUGH", "STREET NAME"], observed=True)
        .size()
        .reset_index(name="CrashCount")
    )
    
    top_streets = (
        street_counts.sort_values("CrashCount", ascending=False)
        .groupby("BOROUGH", observed=True)
        .head(1)
        .reset_index(drop=True)
    )

    STATIC_RQ1_FIG = px.bar(
        top_streets,
        x="BOROUGH",
        y="CrashCount",
        text="STREET NAME",
        color="BOROUGH",
        title="RQ1 – Street with Most Crashes per Borough (All Time)",
    )
    STATIC_RQ1_FIG.update_traces(textposition="auto")
    STATIC_RQ1_FIG = style_figure(STATIC_RQ1_FIG, "RQ1 – Street with Most Crashes per Borough (All Time)")


# --- RQ2: Crash Distribution by Day of Week (PIE) ---
STATIC_RQ2_FIG = None
if "CRASH_DATETIME" in df_with_year.columns and "COLLISION_ID" in df_with_year.columns:
    static_df_2 = df_with_year.copy()
    static_df_2["CRASH_DATETIME"] = pd.to_datetime(static_df_2["CRASH_DATETIME"], errors="coerce")
    static_df_2["DAY_OF_WEEK"] = static_df_2["CRASH_DATETIME"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    rq2_data = (
        static_df_2.drop_duplicates(subset=["COLLISION_ID"])
        .groupby("DAY_OF_WEEK", observed=True)
        .size()
        .reindex(day_order)
        .fillna(0)
        .reset_index(name="CrashCount")
    )

    if not rq2_data.empty:
        STATIC_RQ2_FIG = px.pie(
            rq2_data,
            names="DAY_OF_WEEK",
            values="CrashCount",
            title="RQ2 – Crash Distribution by Day of Week",
        )
        STATIC_RQ2_FIG = style_figure(STATIC_RQ2_FIG, "RQ2 – Crash Distribution by Day of Week")


# --- RQ3: Bodily Injury vs Person Injury (HEATMAP - KILLED FOCUS) ---
STATIC_RQ3_FIG = None
if "BODILY_INJURY" in df_with_year.columns and "PERSON_INJURY" in df_with_year.columns:
    static_df_3 = df_with_year[
        df_with_year["PERSON_INJURY"].astype(str).str.upper() == "KILLED"
    ].dropna(subset=["BODILY_INJURY"])
    
    if not static_df_3.empty:
        rq3_data = (
            static_df_3.groupby(["BODILY_INJURY"], observed=True)
            .size()
            .reset_index(name="Count")
        )
        rq3_data["Status"] = "Killed"
        
        # Ensure strings
        rq3_data["BODILY_INJURY"] = rq3_data["BODILY_INJURY"].astype(str)

        STATIC_RQ3_FIG = px.density_heatmap(
            rq3_data,
            x="BODILY_INJURY",
            y="Status",
            z="Count",
            text_auto=True,
            title="RQ3 – Heatmap: Bodily Injuries Resulting in Death (All Time)",
            color_continuous_scale="Reds"
        )
        STATIC_RQ3_FIG = style_figure(STATIC_RQ3_FIG, "RQ3 – Heatmap: Bodily Injuries Resulting in Death (All Time)")


# --- RQ4: Vehicle Position vs Person Injury (HEATMAP - KILLED FOCUS) ---
STATIC_RQ4_FIG = None
if "POSITION_IN_VEHICLE" in df_with_year.columns and "PERSON_INJURY" in df_with_year.columns:
    static_df_4 = df_with_year[
        df_with_year["PERSON_INJURY"].astype(str).str.upper() == "KILLED"
    ].dropna(subset=["POSITION_IN_VEHICLE"])

    exclude_values = ["UNKNOWN", "DOES NOT APPLY", "NOT APPLICABLE", "N/A", "OTHER"]
    static_df_4 = static_df_4[
        ~static_df_4["POSITION_IN_VEHICLE"].astype(str).str.upper().isin(exclude_values)
    ]

    if not static_df_4.empty:
        rq4_data = (
            static_df_4.groupby(["POSITION_IN_VEHICLE"], observed=True)
            .size()
            .reset_index(name="Count")
        )
        rq4_data["Status"] = "Killed"
        
        # Ensure strings
        rq4_data["POSITION_IN_VEHICLE"] = rq4_data["POSITION_IN_VEHICLE"].astype(str)

        STATIC_RQ4_FIG = px.density_heatmap(
            rq4_data,
            x="POSITION_IN_VEHICLE",
            y="Status",
            z="Count",
            text_auto=True,
            title="RQ4 – Heatmap: Vehicle Positions Resulting in Death (All Time)",
            color_continuous_scale="Reds"
        )
        STATIC_RQ4_FIG.update_xaxes(tickangle=45)
        STATIC_RQ4_FIG = style_figure(STATIC_RQ4_FIG, "RQ4 – Heatmap: Vehicle Positions Resulting in Death (All Time)")


# --- RQ5: Driver Age Groups vs Day/Night Collisions (BAR CHART) ---
STATIC_RQ5_FIG = None
rq5_cols = {"PERSON_TYPE", "PERSON_AGE", "CRASH_DATETIME"}
if rq5_cols.issubset(df_with_year.columns):
    static_df_5 = df_with_year.copy()
    static_df_5 = static_df_5[
        static_df_5["PERSON_TYPE"].astype(str).str.upper().str.contains("DRIVER")
    ]
    
    if not static_df_5.empty:
        static_df_5["HOUR"] = static_df_5["CRASH_DATETIME"].dt.hour
        static_df_5["TIME_OF_DAY"] = static_df_5["HOUR"].apply(
            lambda h: "Night" if (pd.notna(h) and (h >= 20 or h < 6)) else "Day"
        )
        
        def age_bucket(a):
            try:
                a = float(a)
            except Exception:
                return "Unknown"
            if a < 18: return "<18"
            elif a <= 25: return "18–25"
            elif a <= 40: return "26–40"
            elif a <= 60: return "41–60"
            else: return "60+"

        static_df_5["AgeGroup"] = static_df_5["PERSON_AGE"].apply(age_bucket)
        
        rq5_data = (
            static_df_5.groupby(["AgeGroup", "TIME_OF_DAY"], observed=True)
            .size()
            .reset_index(name="Count")
        )
        
        if not rq5_data.empty:
            STATIC_RQ5_FIG = px.bar(
                rq5_data,
                x="AgeGroup",
                y="Count",
                color="TIME_OF_DAY",
                barmode="group",
                title="RQ5 – Driver Age Groups vs Day/Night Collisions (All Time)",
            )
            STATIC_RQ5_FIG = style_figure(STATIC_RQ5_FIG, "RQ5 – Driver Age Groups vs Day/Night Collisions (All Time)")


# --- RQ6: Most Frequent Complaint for Injured Pedestrians (BAR CHART) ---
STATIC_RQ6_FIG = None
rq6_cols = {"PERSON_TYPE", "PERSON_INJURY", "COMPLAINT"}
if rq6_cols.issubset(df_with_year.columns):
    static_df_6 = df_with_year[
        df_with_year["PERSON_TYPE"].astype(str).str.upper() == "PEDESTRIAN"
    ].copy()
    
    static_df_6 = static_df_6[
        ~static_df_6["PERSON_INJURY"].astype(str).str.upper().isin(["NO APPARENT INJURY", "UNKNOWN"])
    ].dropna(subset=["COMPLAINT"])

    exclude_complaints = ["DOES NOT APPLY", "UNKNOWN", "NOT APPLICABLE", "N/A"]
    static_df_6 = static_df_6[
        ~static_df_6["COMPLAINT"].astype(str).str.upper().isin(exclude_complaints)
    ]
    
    if not static_df_6.empty:
        rq6_data = (
            static_df_6.groupby("COMPLAINT", observed=True)
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=True)
            .tail(10) 
        )
        
        # --- FIX: Point 'y' to the COMPLAINT column name, NOT the index ---
        rq6_data = rq6_data.reset_index() # Ensure COMPLAINT is a column, not index
        
        STATIC_RQ6_FIG = px.bar(
            rq6_data,
            y="COMPLAINT", # Correct column name
            x="Count",
            orientation='h',
            title="RQ6 – Top Complaints for Injured Pedestrians (All Time)",
            color="Count",
            color_continuous_scale="Magma"
        )
        STATIC_RQ6_FIG = style_figure(STATIC_RQ6_FIG, "RQ6 – Top Complaints for Injured Pedestrians (All Time)")


# --- RQ7: Top 5 Contributing Factors for Female and Male (STACKED BAR) ---
STATIC_RQ7_FIG = None
factor_col_7 = "CONTRIBUTING FACTOR VEHICLE 1"
if "PERSON_SEX" in df_with_year.columns and factor_col_7 in df_with_year.columns:
    static_df_7 = df_with_year[
        df_with_year["PERSON_SEX"].astype(str).str.upper().isin(["F", "M", "FEMALE", "MALE"])
    ].copy()
    
    static_df_7 = static_df_7[
        ~static_df_7[factor_col_7].astype(str).str.upper().str.contains("UNSPECIFIED|UNKNOWN", na=False)
    ]
    
    if not static_df_7.empty:
        rq7_data = (
            static_df_7.groupby(["PERSON_SEX", factor_col_7], observed=True)
            .size()
            .reset_index(name="Count")
        )
        
        rq7_data = (
            rq7_data.sort_values(["PERSON_SEX", "Count"], ascending=[True, False])
            .groupby("PERSON_SEX")
            .head(5)
        )
        
        # Ensure strings
        rq7_data["PERSON_SEX"] = rq7_data["PERSON_SEX"].astype(str)
        rq7_data[factor_col_7] = rq7_data[factor_col_7].astype(str)

        STATIC_RQ7_FIG = px.bar(
            rq7_data,
            x="PERSON_SEX",
            y="Count",
            color=factor_col_7, 
            text=factor_col_7,  
            title="RQ7 – Top 5 Contributing Factors by Sex (Stacked)",
        )
        STATIC_RQ7_FIG.update_traces(textposition="inside", texttemplate="%{text}<br>(%{y})")
        STATIC_RQ7_FIG = style_figure(STATIC_RQ7_FIG, "RQ7 – Top 5 Contributing Factors by Sex (Stacked)")


# --- RQ8: Safety Equipment Used by Drivers (TREEMAP) ---
STATIC_RQ8_FIG = None
if "SAFETY_EQUIPMENT" in df_with_year.columns and "PERSON_TYPE" in df_with_year.columns:
    static_df_8 = df_with_year[
        df_with_year["PERSON_TYPE"].astype(str).str.upper().str.contains("DRIVER")
    ].dropna(subset=["SAFETY_EQUIPMENT"])
    
    if not static_df_8.empty:
        rq8_data = (
            static_df_8.groupby("SAFETY_EQUIPMENT", observed=True)
            .size()
            .reset_index(name="Count")
        )
        
        # Ensure strings
        rq8_data["SAFETY_EQUIPMENT"] = rq8_data["SAFETY_EQUIPMENT"].astype(str)

        STATIC_RQ8_FIG = px.treemap(
            rq8_data,
            path=["SAFETY_EQUIPMENT"],
            values="Count",
            title="RQ8 – Safety Equipment Used by Drivers (All Time)",
            color="Count",
            color_continuous_scale="Teal"
        )
        STATIC_RQ8_FIG = style_figure(STATIC_RQ8_FIG, "RQ8 – Safety Equipment Used by Drivers (All Time)")


# --- RQ9: Most Bodily Injury for Pedestrians (FUNNEL CHART) ---
STATIC_RQ9_FIG = None
if "PERSON_TYPE" in df_with_year.columns and "BODILY_INJURY" in df_with_year.columns:
    static_df_9 = df_with_year[
        df_with_year["PERSON_TYPE"].astype(str).str.upper() == "PEDESTRIAN"
    ].dropna(subset=["BODILY_INJURY"])

    exclude_injuries = ["UNKNOWN", "DOES NOT APPLY", "NOT APPLICABLE", "N/A", "UNSPECIFIED"]
    static_df_9 = static_df_9[
        ~static_df_9["BODILY_INJURY"].astype(str).str.upper().isin(exclude_injuries)
    ]

    if not static_df_9.empty:
        rq9_data = (
            static_df_9.groupby("BODILY_INJURY", observed=True)
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
            .head(10) 
        )
        
        # --- FIX: Use the actual column/index name for Y axis ---
        rq9_data = rq9_data.reset_index() # Ensure BODILY_INJURY is a column
        
        STATIC_RQ9_FIG = px.funnel(
            rq9_data,
            y="BODILY_INJURY", # Correct column name
            x="Count",
            title="RQ9 – Most Common Bodily Injuries for Pedestrians (All Time)",
            color="Count",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        STATIC_RQ9_FIG = style_figure(STATIC_RQ9_FIG, "RQ9 – Most Common Bodily Injuries for Pedestrians (All Time)")


# --- RQ10: Crash Frequency by Hour of Day (LINE CHART) ---
STATIC_RQ10_FIG = None
if "CRASH_DATETIME" in df_with_year.columns and "COLLISION_ID" in df_with_year.columns:
    # Use deduplicated for crash counts
    static_df_10 = df_with_year.drop_duplicates(subset=["COLLISION_ID"])
    static_df_10["HOUR"] = static_df_10["CRASH_DATETIME"].dt.hour
    
    rq10_data = (
        static_df_10.groupby("HOUR", observed=True)
        .size()
        .reset_index(name="CrashCount")
    )
    
    if not rq10_data.empty:
        STATIC_RQ10_FIG = px.line(
            rq10_data,
            x="HOUR",
            y="CrashCount",
            title="RQ10 – Crash Frequency by Hour of Day (All Time)",
            markers=True, # Adds dots at each data point
        )
        # Optional: Set a specific color that pops in Dark Mode (Cyan/Teal)
        STATIC_RQ10_FIG.update_traces(line_color="#00d2ff", line_width=3)
        
        STATIC_RQ10_FIG.update_xaxes(tickmode='linear', tick0=0, dtick=1)
        STATIC_RQ10_FIG = style_figure(STATIC_RQ10_FIG, "RQ10 – Crash Frequency by Hour of Day (All Time)")

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

TOGGLE_BUTTON_STYLE = {
    "position": "fixed", "top": "20px", "right": "24px", "left": "auto", "zIndex": 1100,
}

SIDEBAR_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "280px",
    "zIndex": 1000, "padding": "20px", "overflowY": "auto",
}

SIDEBAR_HIDDEN = {
    **SIDEBAR_STYLE, "transform": "translateX(-100%)", "transition": "transform 0.3s ease-in-out",
}

CONTENT_STYLE = {
    "marginLeft": "0px", "minHeight": "100vh", "transition": "margin-left 0.3s ease-in-out", "padding": "24px",
}

CONTENT_WITH_SIDEBAR = {**CONTENT_STYLE, "marginLeft": "280px"}

# =========================
# Layout
# =========================

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

# =========================
# Callbacks
# =========================

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
        # 1. Filter Data
        filtered_df = df_with_year.copy()
        for i, val in enumerate(filter_values):
            col_name = filter_config[i]["column"]
            if val:
                valid_vals = val if isinstance(val, list) else [val]
                filtered_df = filtered_df[
                    (filtered_df[col_name].isin(valid_vals)) | 
                    (filtered_df[col_name].isna() & pd.Series([v is None for v in valid_vals]).any())
                ]
        
        if filtered_df.empty:
            return html.Div([dbc.Alert("No data available for filters.", color="warning")]), {"display": "block"}

        # Crash Level Data
        crash_level_df = filtered_df.drop_duplicates(subset=['COLLISION_ID'], keep='first')
        
        # Stats
        total_crashes = len(crash_level_df)
        total_injured = crash_level_df["NUMBER OF PERSONS INJURED"].sum() if "NUMBER OF PERSONS INJURED" in crash_level_df else 0
        total_killed = crash_level_df["NUMBER OF PERSONS KILLED"].sum() if "NUMBER OF PERSONS KILLED" in crash_level_df else 0

        overview_charts = []
        
        # MAP (Dynamic)
        if {"LATITUDE", "LONGITUDE"}.issubset(crash_level_df.columns):
            map_data = crash_level_df.dropna(subset=["LATITUDE", "LONGITUDE"])
            if len(map_data) > 10000: map_data = map_data.sample(10000, random_state=42)
            if not map_data.empty:
                fig_map = px.scatter_map(
                    map_data, lat="LATITUDE", lon="LONGITUDE", map_style="carto-darkmatter", zoom=9,
                    center=dict(lat=40.730610, lon=-73.935242), title="Collision Location Map"
                )
                fig_map.update_traces(marker=dict(size=5, opacity=0.7, color="red"))
                overview_charts.append(dcc.Graph(figure=style_figure(fig_map), className="ios-chart"))

        # LINE (Dynamic)
        if "YEAR" in crash_level_df.columns:
            year_counts = crash_level_df["YEAR"].value_counts().sort_index().reset_index(name="Crashes")
            fig_year = px.line(year_counts, x="YEAR", y="Crashes", title="Crashes Trend by Year", markers=True)
            # Will append in combined row

        # BAR (Dynamic) - Month
        if "CRASH_DATETIME" in crash_level_df.columns:
            crash_level_df["CRASH_DATETIME"] = pd.to_datetime(crash_level_df["CRASH_DATETIME"], errors="coerce")
            crash_level_df["MONTH"] = crash_level_df["CRASH_DATETIME"].dt.month_name()
            month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            month_counts = crash_level_df["MONTH"].value_counts().reindex(month_order).reset_index(name="Crashes")
            fig_month = px.bar(month_counts, x="MONTH", y="Crashes", title="Crashes by Month", color="Crashes")
            # Will append in combined row

        # Combined Row for Year & Month (Side-by-Side)
        if "YEAR" in crash_level_df.columns and "CRASH_DATETIME" in crash_level_df.columns:
             overview_charts.append(dbc.Row([
                 dbc.Col(dcc.Graph(figure=style_figure(fig_year), className="ios-chart"), md=6),
                 dbc.Col(dcc.Graph(figure=style_figure(fig_month), className="ios-chart"), md=6)
             ], className="mb-4"))
        
        # LINE (Dynamic - Crashes by Hour)
        if "CRASH_DATETIME" in crash_level_df.columns:
            # Extract hour directly from the datetime column
            hour_counts = crash_level_df["CRASH_DATETIME"].dt.hour.value_counts().sort_index().reset_index()
            hour_counts.columns = ["HOUR", "Crashes"]
            
            fig_hour = px.line(
                hour_counts, 
                x="HOUR", 
                y="Crashes", 
                title="Crashes by Hour of Day (Filtered)", 
                markers=True
            )
            fig_hour.update_xaxes(tickmode='linear', tick0=0, dtick=1)
            overview_charts.append(dcc.Graph(figure=style_figure(fig_hour), className="ios-chart"))
        
        # BAR (Dynamic - Top Factors) - EXCLUDE Unspecified
        factor_col = "CONTRIBUTING FACTOR VEHICLE 1"
        if factor_col in filtered_df.columns:
            factors = filtered_df[~filtered_df[factor_col].astype(str).str.upper().str.contains("UNSPECIFIED|UNKNOWN", na=False)].copy()
            if not factors.empty:
                f_counts = factors[factor_col].astype(str).value_counts().head(5).reset_index(name="Count")
                f_counts.columns = [factor_col, "Count"]
                fig_f = px.bar(f_counts, x=factor_col, y="Count", title="Top 5 Contributing Factors (Excl. Unspecified)", color=factor_col)
                overview_charts.append(dcc.Graph(figure=style_figure(fig_f), className="ios-chart"))
        
        # PIE (Dynamic - Person Sex Distribution)
        if "PERSON_SEX" in filtered_df.columns:
            sex_counts = filtered_df["PERSON_SEX"].astype(str).value_counts().reset_index(name="Count")
            if not sex_counts.empty:
                sex_counts.columns = ["PERSON_SEX", "Count"]
                fig_pie = px.pie(
                    sex_counts, 
                    names="PERSON_SEX", 
                    values="Count", 
                    title="Person Sex Distribution (Filtered)"
                )
                overview_charts.append(dcc.Graph(figure=style_figure(fig_pie), className="ios-chart"))

        # 1. BAR (Dynamic - Top 3 Streets per Borough) - VERTICAL UPDATE
        if "BOROUGH" in crash_level_df.columns and "STREET NAME" in crash_level_df.columns:
            street_data = crash_level_df.dropna(subset=["BOROUGH", "STREET NAME"])
            street_data = street_data[street_data["STREET NAME"].astype(str).str.strip() != ""]
            
            street_counts = street_data.groupby(["BOROUGH", "STREET NAME"], observed=True).size().reset_index(name="Crashes")
            
            top_streets = (
                street_counts.sort_values(["BOROUGH", "Crashes"], ascending=[True, False])
                .groupby("BOROUGH")
                .head(3)
                .reset_index(drop=True)
            )
            
            if not top_streets.empty:
                # Sort by BOROUGH for grouping, then Crashes descending for visual order
                top_streets = top_streets.sort_values(["BOROUGH", "Crashes"], ascending=[True, False])
                
                fig_streets = px.bar(
                    top_streets,
                    x="BOROUGH", # Borough on X Axis
                    y="Crashes",
                    color="STREET NAME", # Color by street to separate bars side-by-side
                    text="STREET NAME",  # Label bars with street name
                    title="Top 3 High-Crash Streets per Borough",
                )
                # Hide legend (too many unique streets) and rotate text for readability
                fig_streets.update_layout(showlegend=False)
                fig_streets.update_traces(textposition="inside", textangle=-90)
                
                overview_charts.append(dcc.Graph(figure=style_figure(fig_streets), className="ios-chart"))

        # 2. DOUGHNUT (Dynamic - Crashes by Borough)
        if "BOROUGH" in crash_level_df.columns:
            boro_counts = crash_level_df["BOROUGH"].value_counts().reset_index(name="Crashes")
            if not boro_counts.empty:
                boro_counts["BOROUGH"] = boro_counts["BOROUGH"].astype(str)
                fig_doughnut = px.pie(
                    boro_counts, 
                    names="BOROUGH", 
                    values="Crashes", 
                    title="Crash Distribution by Borough", 
                    hole=0.4, # Creates the Doughnut effect
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig_doughnut.update_traces(textinfo="percent+label")
                overview_charts.append(dcc.Graph(figure=style_figure(fig_doughnut), className="ios-chart"))
        
        # HEATMAP (Dynamic - Position vs Bodily Injury)
        if "POSITION_IN_VEHICLE" in filtered_df.columns and "BODILY_INJURY" in filtered_df.columns:
             hm_data = filtered_df.dropna(subset=["POSITION_IN_VEHICLE", "BODILY_INJURY"])
             # Remove uninformative values
             bad_vals = ["UNKNOWN", "DOES NOT APPLY", "N/A", "NOT APPLICABLE"]
             hm_data = hm_data[~hm_data["POSITION_IN_VEHICLE"].astype(str).str.upper().isin(bad_vals)]
             hm_data = hm_data[~hm_data["BODILY_INJURY"].astype(str).str.upper().isin(bad_vals)]
             
             if not hm_data.empty:
                 # Limit to top categories
                 top_pos = hm_data["POSITION_IN_VEHICLE"].value_counts().head(10).index
                 top_bod = hm_data["BODILY_INJURY"].value_counts().head(10).index
                 
                 hm_data = hm_data[hm_data["POSITION_IN_VEHICLE"].isin(top_pos) & hm_data["BODILY_INJURY"].isin(top_bod)].copy()
                 hm_data["POSITION_IN_VEHICLE"] = hm_data["POSITION_IN_VEHICLE"].astype(str)
                 hm_data["BODILY_INJURY"] = hm_data["BODILY_INJURY"].astype(str)
                 
                 hm_data_grouped = hm_data.groupby(["POSITION_IN_VEHICLE", "BODILY_INJURY"], observed=True).size().reset_index(name="Count")

                 fig_hm = px.density_heatmap(
                     hm_data_grouped, 
                     x="POSITION_IN_VEHICLE", 
                     y="BODILY_INJURY", 
                     z="Count",
                     text_auto=True, 
                     title="Heatmap: Position vs Bodily Injury (Filtered)", 
                     color_continuous_scale="Viridis"
                 )
                 overview_charts.append(dcc.Graph(figure=style_figure(fig_hm), className="ios-chart"))

        # TREEMAP (Dynamic - Person Type vs Safety Equipment)
        if "PERSON_TYPE" in filtered_df.columns and "SAFETY_EQUIPMENT" in filtered_df.columns:
             sb_data = filtered_df.dropna(subset=["PERSON_TYPE", "SAFETY_EQUIPMENT"])
             # Exclude "Unknown" and other uninformative values
             sb_data = sb_data[~sb_data["PERSON_TYPE"].astype(str).str.upper().isin(["UNKNOWN"])]
             sb_data = sb_data[~sb_data["SAFETY_EQUIPMENT"].astype(str).str.upper().isin(["UNKNOWN", "NONE", "OTHER"])]
             
             if not sb_data.empty:
                 sb_data["PERSON_TYPE"] = sb_data["PERSON_TYPE"].astype(str)
                 sb_data["SAFETY_EQUIPMENT"] = sb_data["SAFETY_EQUIPMENT"].astype(str)
                 
                 sb_counts = sb_data.groupby(["PERSON_TYPE", "SAFETY_EQUIPMENT"], observed=True).size().reset_index(name="Count")
                 
                 # Limit to top Person Types to keep chart readable
                 top_pt = sb_counts.groupby("PERSON_TYPE")["Count"].sum().nlargest(10).index
                 sb_counts = sb_counts[sb_counts["PERSON_TYPE"].isin(top_pt)]

                 fig_sb = px.treemap(
                     sb_counts, 
                     path=['PERSON_TYPE', 'SAFETY_EQUIPMENT'], 
                     values='Count',
                     color='Count',  # <--- UPDATED: Color by Count for "Levels"
                     title="Safety Equipment by Person Type (Filtered)",
                     color_continuous_scale="Teal" # <--- UPDATED: Gradient Scale
                 )
                 fig_sb.update_traces(textinfo="label+value+percent parent")
                 overview_charts.append(dcc.Graph(figure=style_figure(fig_sb), className="ios-chart"))
        # TREEMAP (Dynamic - Person Type vs Bodily Injury)
        if "PERSON_TYPE" in filtered_df.columns and "BODILY_INJURY" in filtered_df.columns:
             bi_data = filtered_df.dropna(subset=["PERSON_TYPE", "BODILY_INJURY"])
             # Exclude uninformative values
             exclude_vals = ["UNKNOWN", "DOES NOT APPLY", "N/A", "NOT APPLICABLE", "UNSPECIFIED"]
             bi_data = bi_data[~bi_data["PERSON_TYPE"].astype(str).str.upper().isin(exclude_vals)]
             bi_data = bi_data[~bi_data["BODILY_INJURY"].astype(str).str.upper().isin(exclude_vals)]
             
             if not bi_data.empty:
                 bi_data["PERSON_TYPE"] = bi_data["PERSON_TYPE"].astype(str)
                 bi_data["BODILY_INJURY"] = bi_data["BODILY_INJURY"].astype(str)
                 
                 bi_counts = bi_data.groupby(["PERSON_TYPE", "BODILY_INJURY"], observed=True).size().reset_index(name="Count")

                 fig_bi = px.treemap(
                     bi_counts, 
                     path=['PERSON_TYPE', 'BODILY_INJURY'], 
                     values='Count',
                     color='Count', # <--- UPDATED: Color by Count for "Levels"
                     title="Bodily Injury by Person Type (Filtered)",
                     color_continuous_scale="Reds" # <--- UPDATED: Gradient Scale
                 )
                 fig_bi.update_traces(textinfo="label+value+percent parent")
                 overview_charts.append(dcc.Graph(figure=style_figure(fig_bi), className="ios-chart"))
        # BOX PLOT (Dynamic - Age Distribution)
        if "PERSON_AGE" in filtered_df.columns:
             age_data = filtered_df.dropna(subset=["PERSON_AGE"])
             age_data = age_data[(age_data["PERSON_AGE"] >= 0) & (age_data["PERSON_AGE"] <= 110)]
             
             if not age_data.empty:
                 fig_box = px.box(
                     age_data, 
                     y="PERSON_AGE", 
                     title="Age Distribution (Filtered)",
                     points="outliers" 
                 )
                 overview_charts.append(dcc.Graph(figure=style_figure(fig_box), className="ios-chart"))
        
        # SIDE-BY-SIDE HEATMAPS (Dynamic: Injury Analysis)
        if {"PERSON_INJURY", "BODILY_INJURY", "POSITION_IN_VEHICLE"}.issubset(filtered_df.columns):
             exclude_vals = ["UNKNOWN", "DOES NOT APPLY", "N/A", "NOT APPLICABLE", "UNSPECIFIED"]
             row_content = []

             # 1. Heatmap: Person Injury vs Bodily Injury
             h1_data = filtered_df.dropna(subset=["PERSON_INJURY", "BODILY_INJURY"])
             h1_data = h1_data[
                 ~h1_data["PERSON_INJURY"].astype(str).str.upper().isin(exclude_vals) &
                 ~h1_data["BODILY_INJURY"].astype(str).str.upper().isin(exclude_vals)
             ]
             
             if not h1_data.empty:
                 # Limit to top 15 categories for readability
                 top_pi = h1_data["PERSON_INJURY"].value_counts().head(15).index
                 top_bi = h1_data["BODILY_INJURY"].value_counts().head(15).index
                 h1_data = h1_data[h1_data["PERSON_INJURY"].isin(top_pi) & h1_data["BODILY_INJURY"].isin(top_bi)]
                 
                 fig_h1 = px.density_heatmap(
                     h1_data, 
                     x="BODILY_INJURY", 
                     y="PERSON_INJURY", 
                     text_auto=True, # Counts values inside cells
                     title="Person Injury vs Bodily Injury",
                     color_continuous_scale="Blues"
                 )
                 row_content.append(dbc.Col(dcc.Graph(figure=style_figure(fig_h1), className="ios-chart"), md=6))

             # 2. Heatmap: Person Injury vs Position in Vehicle
             h2_data = filtered_df.dropna(subset=["PERSON_INJURY", "POSITION_IN_VEHICLE"])
             h2_data = h2_data[
                 ~h2_data["PERSON_INJURY"].astype(str).str.upper().isin(exclude_vals) &
                 ~h2_data["POSITION_IN_VEHICLE"].astype(str).str.upper().isin(exclude_vals)
             ]
             
             if not h2_data.empty:
                 # Limit to top 15 categories
                 top_pi2 = h2_data["PERSON_INJURY"].value_counts().head(15).index
                 top_pos = h2_data["POSITION_IN_VEHICLE"].value_counts().head(15).index
                 h2_data = h2_data[h2_data["PERSON_INJURY"].isin(top_pi2) & h2_data["POSITION_IN_VEHICLE"].isin(top_pos)]

                 fig_h2 = px.density_heatmap(
                     h2_data, 
                     x="POSITION_IN_VEHICLE", 
                     y="PERSON_INJURY", 
                     text_auto=True, # Counts values inside cells
                     title="Person Injury vs Position in Vehicle",
                     color_continuous_scale="Oranges"
                 )
                 row_content.append(dbc.Col(dcc.Graph(figure=style_figure(fig_h2), className="ios-chart"), md=6))

             # Append the row if we have at least one chart
             if row_content:
                 overview_charts.append(dbc.Row(row_content, className="mb-4"))

    # SUNBURST (Dynamic: Person Type -> Injury -> Complaint)
        if {"PERSON_TYPE", "PERSON_INJURY", "COMPLAINT"}.issubset(filtered_df.columns):
             sb_cols = ["PERSON_TYPE", "PERSON_INJURY", "COMPLAINT"]
             sb_data = filtered_df.dropna(subset=sb_cols)
             
             # Filter out uninformative values for cleaner visualization
             exclude_vals = ["UNKNOWN", "DOES NOT APPLY", "N/A", "NOT APPLICABLE", "NONE"]
             for col in sb_cols:
                 sb_data = sb_data[~sb_data[col].astype(str).str.upper().isin(exclude_vals)]
             
             if not sb_data.empty:
                 # To ensure performance and readability, we limit to the most frequent combinations if data is huge
                 # Grouping first to reduce rows
                 sb_grouped = sb_data.groupby(sb_cols, observed=True).size().reset_index(name="Count")
                 
                 # If too many unique complaints, keep only top 20 overall to avoid clutter
                 top_complaints = sb_grouped.groupby("COMPLAINT")["Count"].sum().nlargest(20).index
                 sb_grouped = sb_grouped[sb_grouped["COMPLAINT"].isin(top_complaints)]
                 
                 fig_sun = px.sunburst(
                     sb_grouped,
                     path=["PERSON_TYPE", "PERSON_INJURY", "COMPLAINT"],
                     values="Count",
                     title="Hierarchy: Person Type → Injury → Complaint",
                     color="Count",
                     color_continuous_scale="RdBu",
                     branchvalues="total"
                 )
                 fig_sun = style_figure(fig_sun, "Hierarchy: Person Type → Injury → Complaint")
                 overview_charts.append(dcc.Graph(figure=fig_sun, className="ios-chart"))  


        # RQs
        rq_sections = []
        
        # STATIC RQs 1-10
        static_charts = [
            STATIC_RQ1_FIG, STATIC_RQ2_FIG, STATIC_RQ3_FIG, STATIC_RQ4_FIG, 
            STATIC_RQ5_FIG, STATIC_RQ6_FIG, STATIC_RQ7_FIG, STATIC_RQ8_FIG, 
            STATIC_RQ9_FIG, STATIC_RQ10_FIG
        ]
        
        questions = {
            1: "RQ1: Which street has the most crashes in each borough?",
            2: "RQ2: Which day of the week has the most crashes?",
            3: "RQ3: Which bodily injuries most frequently result in death?",
            4: "RQ4: Which vehicle positions most frequently result in death?",
            5: "RQ5: Are young drivers more involved in night-time crashes?",
            6: "RQ6: What is the most common complaint for injured pedestrians?",
            7: "RQ7: Top 5 Contributing Factors for Female and Male",
            8: "RQ8: What safety equipment is most frequently used by drivers?",
            9: "RQ9: What are the most common bodily injuries faced by pedestrians?",
            10: "RQ10: What is the most dangerous time of day for crashes?"
        }
        
        for i, fig in enumerate(static_charts, 1):
            if fig:
                rq_sections.append(html.Div([
                    html.H5(questions.get(i, f"RQ{i}"), className="section-title mb-2"),
                    dcc.Graph(figure=fig, className="ios-chart"),
                    html.Small("Note: This visualization is based on the full dataset (All Time) and does not change with filters.", className="text-muted")
                ], className="mb-4"))
            else:
                rq_sections.append(html.Div(dbc.Alert(f"RQ{i}: No data available (Static).", color="warning"), className="mb-4"))

        # 5. Build the full report layout
        report_children = [
            dbc.Row(
                [
                    dbc.Col(dbc.Button("Hide Report", id="hide-report-div-btn", color="secondary", className="float-end ios-secondary-btn"), width={"size": 3, "offset": 9}),
                ], className="mb-3"
            ),
            html.H3("Collision Analysis Report", className="report-title mb-4"),
            html.H4("Summary Statistics", className="section-title mb-3"),
            dbc.Row(
                [
                    dbc.Col(dbc.Card([dbc.CardHeader("Total Crashes"), dbc.CardBody(html.H2(f"{total_crashes:,}"))], className="ios-stat-card ios-stat-primary"), md=4),
                    dbc.Col(dbc.Card([dbc.CardHeader("Persons Injured"), dbc.CardBody(html.H2(f"{int(total_injured):,}"))], className="ios-stat-card ios-stat-warning"), md=4),
                    dbc.Col(dbc.Card([dbc.CardHeader("Persons Killed"), dbc.CardBody(html.H2(f"{int(total_killed):,}"))], className="ios-stat-card ios-stat-danger"), md=4),
                ], className="mb-4"
            ),
            html.Hr(),
            html.H4("Core Visualizations", className="section-title mb-3"),
        ]
        
        # Append charts - Year/Month row is appended directly, others wrapped in Divs
        for item in overview_charts:
             if isinstance(item, dbc.Row):
                 report_children.append(item)
             else:
                 report_children.append(html.Div(item, className="mb-4"))

        if rq_sections:
            report_children.append(html.Hr())
            report_children.append(html.H4("Research Question Visualizations", className="section-title mb-3"))
            report_children.extend(rq_sections)

        report_content = html.Div(report_children)
        return report_content, {"display": "block"}
    
    except Exception as e:
        error_message = f"Report Generation Failed: An unexpected error occurred during data processing. Error: {e}. Try clearing your filters."
        print(f"FATAL CALLBACK ERROR: {e}")
        error_content = html.Div(
            [
                html.H3("Report Error", className="report-title mb-4 text-danger"),
                dbc.Alert(error_message, color="danger"),
                dbc.Button("Hide Report", id="hide-report-div-btn", color="secondary", className="mt-3")
            ], className="report-card p-4"
        )
        return error_content, {"display": "block"}


# 5. Hide Report
@app.callback(
    [
        Output("report-display-area", "children", allow_duplicate=True),
        Output("report-display-area", "style", allow_duplicate=True),
    ],
    [Input("hide-report-div-btn", "n_clicks")],
    prevent_initial_call=True,
)
def hide_report_in_page(n_close):
    if n_close and n_close > 0:
        return None, {"display": "none"}
    return no_update, no_update


if __name__ == "__main__":
    app.run(debug=True)