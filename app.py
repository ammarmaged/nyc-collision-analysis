import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, no_update
from pathlib import Path

from components import create_filters_grid, get_filter_config_with_year

# =========================
# Data loading
# =========================

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

# =========================
# Global Plotly dark styling helper
# =========================

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


# =========================
# Dash app
# =========================

app = Dash(
    __name__,
    external_stylesheets=["assets/styles.css", dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "NYC Motor Vehicle Collisions – iPhone 17 Pro Edition"

TOGGLE_BUTTON_STYLE = {
    "position": "fixed",
    "top": "20px",
    "right": "24px",   # move to right side
    "left": "auto",
    "zIndex": 1100,
}


SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "280px",
    "zIndex": 1000,
    "padding": "20px",
    "overflowY": "auto",
}

SIDEBAR_HIDDEN = {
    **SIDEBAR_STYLE,
    "transform": "translateX(-100%)",
    "transition": "transform 0.3s ease-in-out",
}

CONTENT_STYLE = {
    "marginLeft": "0px",
    "minHeight": "100vh",
    "transition": "margin-left 0.3s ease-in-out",
    "padding": "24px",
}

CONTENT_WITH_SIDEBAR = {
    **CONTENT_STYLE,
    "marginLeft": "280px",
}

# =========================
# Layout
# =========================

app.layout = html.Div(
    id="app",
    children=[
        # Toggle button
        html.Button(
            "☰ Filters",
            id="toggle-button",
            style=TOGGLE_BUTTON_STYLE,
            n_clicks=0,
        ),

        # Sidebar
        html.Div(
            id="sidebar",
            style=SIDEBAR_STYLE,
            children=[
                html.Div("NYC Collisions", className="sidebar-title"),

                # Generate Report Button (main requirement)
                dbc.Button(
                    "Generate Report",
                    id="generate-report-btn",
                    color="success",
                    className="mb-3 w-100 ios-primary-btn",
                    n_clicks=0,
                ),
                html.Hr(className="sidebar-divider"),

                html.H4("Filters", className="filters-title"),

                # Search input (search mode)
                html.Div(
                    [
                        dcc.Input(
                            id="search-input",
                            type="text",
                            placeholder="Try: Brooklyn 2022 pedestrian…",
                            debounce=True,
                            className="ios-search-input",
                        )
                    ],
                    style={"marginBottom": "12px"},
                ),

                # Filters Grid (auto-generated)
                create_filters_grid(df_with_year, filter_config, sidebar=True),
            ],
        ),

        # Main content
        html.Div(
            id="content",
            style=CONTENT_WITH_SIDEBAR,
            children=[
                # iOS-style top bar
                html.Div(
                    className="ios-top-bar",
                    children=[
                        html.Div("NYC Collision Insight", className="ios-top-title"),
                        html.Div(
                            "Live · Interactive · Filtered",
                            className="ios-top-sub",
                        ),
                    ],
                ),

                html.Div(
                    id="main-content",
                    children=[
                        html.H1("NYC Motor Vehicle Collisions Analysis"),
                        html.P(
                            "Use filters or the smart search, then tap “Generate Report” to see an interactive analysis dashboard.",
                            className="lead-text",
                        ),

                        # Report Display Area (initially hidden)
                        html.Div(
                            id="report-display-area",
                            className="report-card",
                            style={"display": "none"},
                        ),

                        # Placeholder for extra charts if needed
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


# 1. Toggle Sidebar
@app.callback(
    [Output("sidebar", "style"), Output("content", "style"), Output("toggle-button", "style")],
    [Input("toggle-button", "n_clicks")],
    prevent_initial_call=False,
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


# 2. Search -> Auto-apply filters
@app.callback(
    [Output(f"{cfg.get('id') or cfg['column'].lower()}-filter", "value", allow_duplicate=True) for cfg in filter_config],
    [Input("search-input", "value")],
    prevent_initial_call="initial_duplicate",
)
def apply_search_to_filters(search_value):
    if not search_value:
        return [None] * len(filter_config)

    import re

    def _format_label(v):
        if v is None:
            return ""
        try:
            import pandas as _pd
            if isinstance(v, (_pd.Timestamp,)):
                return v.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

        s = str(v).strip()
        while s.startswith("_"):
            s = s[1:]
        s = s.replace("_", " ")
        try:
            if "." in s:
                f = float(s)
                if f.is_integer():
                    return str(int(f))
        except Exception:
            pass
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
                if label == token and v not in matched:
                    matched.append(v)
        for token in tokens:
            for label, words, v in candidates:
                if token in words and v not in matched:
                    matched.append(v)
        return matched if matched else None

    outputs = []
    for cfg in filter_config:
        outputs.append(_find_matches_for_column(cfg["column"]))
    return outputs


# 3. Expand "All" Selections
@app.callback(
    [Output(f"{cfg.get('id') or cfg['column'].lower()}-filter", "value", allow_duplicate=True) for cfg in filter_config],
    [Input(f"{cfg.get('id') or cfg['column'].lower()}-filter", "value") for cfg in filter_config],
    prevent_initial_call=True,
)
def expand_all_selections(*values):
    outputs = []
    for i, val in enumerate(values):
        col_name = filter_config[i]["column"]
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


# 4. Generate Report (with all research visualizations)
@app.callback(
    [
        Output("report-display-area", "children", allow_duplicate=True),
        Output("report-display-area", "style", allow_duplicate=True),
    ],
    [Input("generate-report-btn", "n_clicks")],
    [State(f"{cfg.get('id') or cfg['column'].lower()}-filter", "value") for cfg in filter_config],
    prevent_initial_call=True,
)
def generate_report_in_page(n_open, *filter_values):
    if n_open is None or n_open == 0:
        return no_update, no_update

    # 1. Filter Data
    filtered_df = df_with_year.copy()

    for i, val in enumerate(filter_values):
        col_name = filter_config[i]["column"]
        if val:
            valid_vals = val if isinstance(val, list) else [val]
            filtered_df = filtered_df[filtered_df[col_name].isin(filtered_df[col_name].dropna().unique())]
            filtered_df = filtered_df[filtered_df[col_name].isin(valid_vals)]

    # If everything filtered out
    if filtered_df is None or filtered_df.empty:
        report_content = html.Div(
            [
                dbc.Row(
                    dbc.Col(
                        dbc.Button(
                            "Hide Report",
                            id="hide-report-div-btn",
                            color="secondary",
                            className="float-end ios-secondary-btn",
                        ),
                        width={"size": 3, "offset": 9},
                    ),
                    className="mb-3",
                ),
                html.H3("Collision Analysis Report", className="report-title mb-4"),
                dbc.Alert(
                    "No data available for the selected filters. Try relaxing your filters or clearing the search.",
                    color="warning",
                    className="ios-alert",
                ),
            ]
        )
        return report_content, {"display": "block"}

    # 2. Derived time columns for RQs
    if "CRASH_DATETIME" in filtered_df.columns:
        filtered_df["CRASH_DATETIME"] = pd.to_datetime(
            filtered_df["CRASH_DATETIME"], errors="coerce"
        )
        filtered_df["HOUR"] = filtered_df["CRASH_DATETIME"].dt.hour
        filtered_df["DAY_OF_WEEK"] = filtered_df["CRASH_DATETIME"].dt.day_name()
        # Day/Night flag
        filtered_df["TIME_OF_DAY"] = filtered_df["HOUR"].apply(
            lambda h: "Night" if (pd.notna(h) and (h >= 20 or h < 6)) else "Day"
        )

    # Severity flag
    if "NUMBER OF PERSONS KILLED" in filtered_df.columns:
        filtered_df["ANY_FATAL"] = filtered_df["NUMBER OF PERSONS KILLED"] > 0
    else:
        filtered_df["ANY_FATAL"] = False

    # 3. Summary Stats
    total_crashes = len(filtered_df)
    total_injured = (
        filtered_df["NUMBER OF PERSONS INJURED"].sum()
        if "NUMBER OF PERSONS INJURED" in filtered_df.columns
        else 0
    )
    total_killed = (
        filtered_df["NUMBER OF PERSONS KILLED"].sum()
        if "NUMBER OF PERSONS KILLED" in filtered_df.columns
        else 0
    )

    # ========== CORE OVERVIEW VISUALS ==========
    overview_charts = []

    # Map: all collisions in filtered set
    if {"LATITUDE", "LONGITUDE"}.issubset(filtered_df.columns):
        map_data = filtered_df.dropna(subset=["LATITUDE", "LONGITUDE"])
        MAX_MAP_POINTS = 10000
        if len(map_data) > MAX_MAP_POINTS:
            map_data = map_data.sample(MAX_MAP_POINTS, random_state=42)
        if not map_data.empty:
            fig_map = px.scatter_mapbox(
                map_data,
                lat="LATITUDE",
                lon="LONGITUDE",
                mapbox_style="carto-darkmatter",  # DARK TILESET
                zoom=9,
                center=dict(lat=40.730610, lon=-73.935242),
                title="Collision Location Map (Sampled if Very Large)",
            )
            fig_map.update_traces(
                marker=dict(size=5, opacity=0.7, color="red"),
                selector=dict(mode="markers"),
            )
            fig_map.update_layout(
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                hoverlabel=dict(
                    bgcolor="white", font_size=12, font_family="SF Pro Text"
                ),
            )
            fig_map = style_figure(fig_map, "Collision Location Map (Sampled if Very Large)")
            overview_charts.append(dcc.Graph(figure=fig_map, className="ios-chart"))

    # Line: Crashes by Year
    if "YEAR" in filtered_df.columns:
        year_counts = filtered_df["YEAR"].value_counts().sort_index().reset_index()
        year_counts.columns = ["Year", "Crashes"]
        fig_year = px.line(
            year_counts,
            x="Year",
            y="Crashes",
            title="Crashes Trend by Year",
            markers=True,
        )
        fig_year = style_figure(fig_year, "Crashes Trend by Year")
        overview_charts.append(dcc.Graph(figure=fig_year, className="ios-chart"))

    # Bar: Top contributing factors
    factor_column = "CONTRIBUTING FACTOR VEHICLE 1"
    if factor_column in filtered_df.columns:
        non_unspecified_factors = filtered_df[
            ~filtered_df[factor_column]
            .astype(str)
            .str.upper()
            .str.contains("UNSPECIFIED|UNKNOWN", na=False)
        ]
        if not non_unspecified_factors.empty:
            factor_counts = (
                non_unspecified_factors[factor_column]
                .value_counts()
                .head(5)
                .reset_index()
            )
            factor_counts.columns = ["Factor", "Count"]
            fig_factor = px.bar(
                factor_counts,
                x="Factor",
                y="Count",
                title="Top 5 Contributing Factors",
                color="Factor",
            )
            fig_factor = style_figure(fig_factor, "Top 5 Contributing Factors")
            overview_charts.append(dcc.Graph(figure=fig_factor, className="ios-chart"))

    # Pie: Injury type distribution
    injury_column = "PERSON_INJURY"
    if injury_column in filtered_df.columns:
        injury_counts_filtered = filtered_df[
            filtered_df[injury_column] != "NO APPARENT INJURY"
        ][injury_column].value_counts().reset_index()
        injury_counts_filtered.columns = ["Injury Type", "Count"]
        if not injury_counts_filtered.empty:
            fig_pie = px.pie(
                injury_counts_filtered,
                names="Injury Type",
                values="Count",
                title="Injury Type Distribution (Excluding No Injury)",
            )
            fig_pie = style_figure(
                fig_pie, "Injury Type Distribution (Excluding No Injury)"
            )
            overview_charts.append(dcc.Graph(figure=fig_pie, className="ios-chart"))

    # ========== RESEARCH QUESTION VISUALS ==========
    rq_sections = []

    # RQ1: Severity index over years by borough
    if {"YEAR", "BOROUGH"}.issubset(filtered_df.columns):
        rq1 = (
            filtered_df.groupby(["YEAR", "BOROUGH"], dropna=True)
            .agg(
                crashes=("COLLISION_ID", "nunique")
                if "COLLISION_ID" in filtered_df.columns
                else ("YEAR", "size"),
                injured=(
                    "NUMBER OF PERSONS INJURED",
                    "sum",
                )
                if "NUMBER OF PERSONS INJURED" in filtered_df.columns
                else ("YEAR", "size"),
                killed=(
                    "NUMBER OF PERSONS KILLED",
                    "sum",
                )
                if "NUMBER OF PERSONS KILLED" in filtered_df.columns
                else ("YEAR", "size"),
            )
            .reset_index()
        )
        rq1["Severity Index"] = rq1["injured"] + 3 * rq1["killed"]
        fig_rq1 = px.line(
            rq1,
            x="YEAR",
            y="Severity Index",
            color="BOROUGH",
            title="RQ1 – Severity Index Over Time by Borough",
        )
        fig_rq1 = style_figure(fig_rq1, "RQ1 – Severity Index Over Time by Borough")
        rq_sections.append(
            html.Div(
                [
                    html.H5(
                        "RQ1: How has crash severity changed over the years across boroughs?",
                        className="section-title mb-2",
                    ),
                    dcc.Graph(figure=fig_rq1, className="ios-chart"),
                ],
                className="mb-4",
            )
        )

    # RQ2: Pedestrian serious injuries by day/hour (heatmap)
    if {
        "PERSON_TYPE",
        "PERSON_INJURY",
        "DAY_OF_WEEK",
        "HOUR",
    }.issubset(filtered_df.columns):
        ped_serious = filtered_df[
            (filtered_df["PERSON_TYPE"].astype(str).str.upper() == "PEDESTRIAN")
            & (filtered_df["PERSON_INJURY"] != "NO APPARENT INJURY")
        ].copy()
        if not ped_serious.empty:
            day_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            ped_serious["DAY_OF_WEEK"] = pd.Categorical(
                ped_serious["DAY_OF_WEEK"], categories=day_order, ordered=True
            )
            rq2 = (
                ped_serious.groupby(["DAY_OF_WEEK", "HOUR"])
                .size()
                .reset_index(name="Count")
            )
            fig_rq2 = px.density_heatmap(
                rq2,
                x="HOUR",
                y="DAY_OF_WEEK",
                z="Count",
                title="RQ2 – Pedestrian Serious Injuries by Hour & Day of Week",
                color_continuous_scale="OrRd",
            )
            fig_rq2 = style_figure(
                fig_rq2, "RQ2 – Pedestrian Serious Injuries by Hour & Day of Week"
            )
            rq_sections.append(
                html.Div(
                    [
                        html.H5(
                            "RQ2: When are pedestrians most likely to be seriously injured?",
                            className="section-title mb-2",
                        ),
                        dcc.Graph(figure=fig_rq2, className="ios-chart"),
                    ],
                    className="mb-4",
                )
            )

    # RQ3: Factors – fatal vs non-fatal
    if factor_column in filtered_df.columns:
        temp = filtered_df.copy()
        temp["SeverityGroup"] = temp["ANY_FATAL"].map(
            {True: "Fatal Collision", False: "Non-Fatal Collision"}
        )
        rq3 = (
            temp.groupby(["SeverityGroup", factor_column], dropna=True)
            .size()
            .reset_index(name="Count")
        )
        fatal_factors = (
            rq3[rq3["SeverityGroup"] == "Fatal Collision"]
            .sort_values("Count", ascending=False)[factor_column]
            .head(6)
        )
        rq3 = rq3[rq3[factor_column].isin(fatal_factors)]
        if not rq3.empty:
            fig_rq3 = px.bar(
                rq3,
                x=factor_column,
                y="Count",
                color="SeverityGroup",
                barmode="group",
                title="RQ3 – Contributing Factors: Fatal vs Non-Fatal Collisions",
            )
            fig_rq3 = style_figure(
                fig_rq3,
                "RQ3 – Contributing Factors: Fatal vs Non-Fatal Collisions",
            )
            rq_sections.append(
                html.Div(
                    [
                        html.H5(
                            "RQ3: Are some factors more associated with fatal collisions?",
                            className="section-title mb-2",
                        ),
                        dcc.Graph(figure=fig_rq3, className="ios-chart"),
                    ],
                    className="mb-4",
                )
            )

    # RQ4: Safety equipment vs injury severity (vehicle occupants)
    if {
        "PERSON_TYPE",
        "SAFETY_EQUIPMENT",
        "PERSON_INJURY",
    }.issubset(filtered_df.columns):
        occ = filtered_df[
            filtered_df["PERSON_TYPE"].astype(str).str.upper().str.contains("OCCUP")
        ].copy()
        if not occ.empty:
            rq4 = (
                occ.groupby(["SAFETY_EQUIPMENT", "PERSON_INJURY"])
                .size()
                .reset_index(name="Count")
            )
            fig_rq4 = px.bar(
                rq4,
                x="SAFETY_EQUIPMENT",
                y="Count",
                color="PERSON_INJURY",
                title="RQ4 – Injury Severity vs Safety Equipment (Occupants)",
            )
            fig_rq4 = style_figure(
                fig_rq4,
                "RQ4 – Injury Severity vs Safety Equipment (Occupants)",
            )
            rq_sections.append(
                html.Div(
                    [
                        html.H5(
                            "RQ4: How does injury severity differ by safety equipment?",
                            className="section-title mb-2",
                        ),
                        dcc.Graph(figure=fig_rq4, className="ios-chart"),
                    ],
                    className="mb-4",
                )
            )

    # RQ5: Age groups vs day/night for drivers
    if {
        "PERSON_TYPE",
        "PERSON_AGE",
        "TIME_OF_DAY",
    }.issubset(filtered_df.columns):
        drivers = filtered_df[
            filtered_df["PERSON_TYPE"].astype(str).str.upper().str.contains("DRIVER")
        ].copy()
        if not drivers.empty:
            def age_bucket(a):
                try:
                    a = float(a)
                except Exception:
                    return "Unknown"
                if a < 18:
                    return "<18"
                elif a <= 25:
                    return "18–25"
                elif a <= 40:
                    return "26–40"
                elif a <= 60:
                    return "41–60"
                else:
                    return "60+"

            drivers["AgeGroup"] = drivers["PERSON_AGE"].apply(age_bucket)
            rq5 = (
                drivers.groupby(["AgeGroup", "TIME_OF_DAY"])
                .size()
                .reset_index(name="Count")
            )
            fig_rq5 = px.bar(
                rq5,
                x="AgeGroup",
                y="Count",
                color="TIME_OF_DAY",
                barmode="group",
                title="RQ5 – Driver Age Groups vs Day/Night Collisions",
            )
            fig_rq5 = style_figure(
                fig_rq5, "RQ5 – Driver Age Groups vs Day/Night Collisions"
            )
            rq_sections.append(
                html.Div(
                    [
                        html.H5(
                            "RQ5: Are young drivers more involved in night-time crashes?",
                            className="section-title mb-2",
                        ),
                        dcc.Graph(figure=fig_rq5, className="ios-chart"),
                    ],
                    className="mb-4",
                )
            )

    # RQ6: Pedestrian collision share by borough & year
    if {"YEAR", "BOROUGH", "PERSON_TYPE"}.issubset(filtered_df.columns):
        ped_mask = (
            filtered_df["PERSON_TYPE"].astype(str).str.upper() == "PEDESTRIAN"
        )
        grp_all = (
            filtered_df.groupby(["YEAR", "BOROUGH"])
            .size()
            .reset_index(name="Total")
        )
        grp_ped = (
            filtered_df[ped_mask]
            .groupby(["YEAR", "BOROUGH"])
            .size()
            .reset_index(name="PedestrianCrashes")
        )
        rq6 = grp_all.merge(grp_ped, on=["YEAR", "BOROUGH"], how="left")
        rq6["PedestrianCrashes"] = rq6["PedestrianCrashes"].fillna(0)
        rq6["PedestrianShare"] = rq6["PedestrianCrashes"] / rq6["Total"]
        fig_rq6 = px.line(
            rq6,
            x="YEAR",
            y="PedestrianShare",
            color="BOROUGH",
            title="RQ6 – Pedestrian Collision Share Over Time by Borough",
        )
        fig_rq6 = style_figure(
            fig_rq6, "RQ6 – Pedestrian Collision Share Over Time by Borough"
        )
        rq_sections.append(
            html.Div(
                [
                    html.H5(
                        "RQ6: Which boroughs have higher pedestrian collision shares?",
                        className="section-title mb-2",
                    ),
                    dcc.Graph(figure=fig_rq6, className="ios-chart"),
                ],
                className="mb-4",
            )
        )

    # RQ7: Pedestrian severity vs sex
    if {
        "PERSON_TYPE",
        "PERSON_SEX",
        "PERSON_INJURY",
    }.issubset(filtered_df.columns):
        ped2 = filtered_df[
            filtered_df["PERSON_TYPE"].astype(str).str.upper() == "PEDESTRIAN"
        ].copy()
        if not ped2.empty:
            rq7 = (
                ped2.groupby(["PERSON_SEX", "PERSON_INJURY"])
                .size()
                .reset_index(name="Count")
            )
            fig_rq7 = px.bar(
                rq7,
                x="PERSON_SEX",
                y="Count",
                color="PERSON_INJURY",
                barmode="stack",
                title="RQ7 – Pedestrian Injury Severity by Sex",
            )
            fig_rq7 = style_figure(
                fig_rq7, "RQ7 – Pedestrian Injury Severity by Sex"
            )
            rq_sections.append(
                html.Div(
                    [
                        html.H5(
                            "RQ7: Do male vs female pedestrians show different severity patterns?",
                            className="section-title mb-2",
                        ),
                        dcc.Graph(figure=fig_rq7, className="ios-chart"),
                    ],
                    className="mb-4",
                )
            )

    # RQ8: Seat position vs severe injury rate
    if {
        "POSITION_IN_VEHICLE",
        "PERSON_INJURY",
    }.issubset(filtered_df.columns):
        occ2 = filtered_df[
            filtered_df["PERSON_TYPE"].astype(str).str.upper().str.contains("OCCUP")
        ].copy()
        if not occ2.empty:
            occ2["Severe"] = ~occ2["PERSON_INJURY"].isin(
                ["NO APPARENT INJURY", "UNKNOWN"]
            )
            rq8 = (
                occ2.groupby("POSITION_IN_VEHICLE")
                .agg(
                    Total=("POSITION_IN_VEHICLE", "size"),
                    Severe=("Severe", "sum"),
                )
                .reset_index()
            )
            rq8["SevereRate"] = rq8["Severe"] / rq8["Total"]
            fig_rq8 = px.bar(
                rq8,
                x="POSITION_IN_VEHICLE",
                y="SevereRate",
                title="RQ8 – Severe Injury Rate by Seat Position",
            )
            fig_rq8 = style_figure(
                fig_rq8, "RQ8 – Severe Injury Rate by Seat Position"
            )
            rq_sections.append(
                html.Div(
                    [
                        html.H5(
                            "RQ8: Are some vehicle positions riskier than others?",
                            className="section-title mb-2",
                        ),
                        dcc.Graph(figure=fig_rq8, className="ios-chart"),
                    ],
                    className="mb-4",
                )
            )

    # RQ9: Factors – pedestrians vs cyclists
    if {
        "PERSON_TYPE",
        factor_column,
    }.issubset(filtered_df.columns):
        ped_cyc = filtered_df[
            filtered_df["PERSON_TYPE"].astype(str).str.upper().isin(
                ["PEDESTRIAN", "BICYCLIST"]
            )
        ].copy()
        if not ped_cyc.empty:
            ped_cyc[factor_column] = ped_cyc[factor_column].astype(str)
            rq9 = (
                ped_cyc.groupby(["PERSON_TYPE", factor_column])
                .size()
                .reset_index(name="Count")
            )
            top_factors = (
                rq9.groupby(factor_column)["Count"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
                .index
            )
            rq9 = rq9[rq9[factor_column].isin(top_factors)]
            fig_rq9 = px.bar(
                rq9,
                x=factor_column,
                y="Count",
                color="PERSON_TYPE",
                barmode="group",
                title="RQ9 – Key Factors: Pedestrians vs Cyclists",
            )
            fig_rq9 = style_figure(
                fig_rq9, "RQ9 – Key Factors: Pedestrians vs Cyclists"
            )
            rq_sections.append(
                html.Div(
                    [
                        html.H5(
                            "RQ9: Do pedestrians and cyclists face different risk factors?",
                            className="section-title mb-2",
                        ),
                        dcc.Graph(figure=fig_rq9, className="ios-chart"),
                    ],
                    className="mb-4",
                )
            )

    # RQ10: High severity hotspots map
    if {"LATITUDE", "LONGITUDE", "PERSON_INJURY"}.issubset(filtered_df.columns):
        severe_mask = filtered_df["PERSON_INJURY"].isin(
            ["KILLED", "INJURED", "SERIOUS INJURY"]
        ) | filtered_df.get("ANY_FATAL", False)
        severe_map = filtered_df[severe_mask].dropna(subset=["LATITUDE", "LONGITUDE"])
        if not severe_map.empty:
            MAX_HOTSPOT_POINTS = 8000
            if len(severe_map) > MAX_HOTSPOT_POINTS:
                severe_map = severe_map.sample(MAX_HOTSPOT_POINTS, random_state=42)
            fig_rq10 = px.scatter_mapbox(
                severe_map,
                lat="LATITUDE",
                lon="LONGITUDE",
                mapbox_style="carto-darkmatter",
                zoom=9,
                center=dict(lat=40.730610, lon=-73.935242),
                title="RQ10 – High Severity Collision Hotspots",
            )
            fig_rq10.update_traces(
                marker=dict(size=6, opacity=0.8, color="orange"),
                selector=dict(mode="markers"),
            )
            fig_rq10 = style_figure(
                fig_rq10, "RQ10 – High Severity Collision Hotspots"
            )
            rq_sections.append(
                html.Div(
                    [
                        html.H5(
                            "RQ10: Where are high-severity collision hotspots located?",
                            className="section-title mb-2",
                        ),
                        dcc.Graph(figure=fig_rq10, className="ios-chart"),
                    ],
                    className="mb-4",
                )
            )

    # 5. Build the full report layout
    report_children = [
        dbc.Row(
            dbc.Col(
                dbc.Button(
                    "Hide Report",
                    id="hide-report-div-btn",
                    color="secondary",
                    className="float-end ios-secondary-btn",
                ),
                width={"size": 3, "offset": 9},
            ),
            className="mb-3",
        ),
        html.H3("Collision Analysis Report", className="report-title mb-4"),
        html.H4("Summary Statistics", className="section-title mb-3"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Total Crashes"),
                            dbc.CardBody(html.H2(f"{total_crashes:,}")),
                        ],
                        className="ios-stat-card ios-stat-primary",
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Persons Injured"),
                            dbc.CardBody(html.H2(f"{int(total_injured):,}")),
                        ],
                        className="ios-stat-card ios-stat-warning",
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Persons Killed"),
                            dbc.CardBody(html.H2(f"{int(total_killed):,}")),
                        ],
                        className="ios-stat-card ios-stat-danger",
                    ),
                    md=4,
                ),
            ],
            className="mb-4",
        ),
        html.Hr(),
        html.H4("Core Visualizations", className="section-title mb-3"),
    ]

    for ch in overview_charts:
        report_children.append(html.Div(ch, className="mb-4"))

    if rq_sections:
        report_children.append(html.Hr())
        report_children.append(
            html.H4(
                "Research Question Visualizations",
                className="section-title mb-3",
            )
        )
        report_children.extend(rq_sections)

    report_content = html.Div(report_children)

    return report_content, {"display": "block"}


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
