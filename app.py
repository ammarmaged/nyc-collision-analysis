
import pandas as pd
from dash import Dash, html, dcc, Input, Output
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

app = Dash(__name__, external_stylesheets=["assets/styles.css"])
app.title = "NYC Motor Vehicle Collisions Analysis"

# Minimal inline styles (most styling now in CSS)
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
}

SIDEBAR_HIDDEN = {
    **SIDEBAR_STYLE,
    "transform": "translateX(-100%)"
}

CONTENT_STYLE = {
    "marginLeft": "0px",
    "minHeight": "100vh",
}

CONTENT_WITH_SIDEBAR = {
    **CONTENT_STYLE,
    "marginLeft": "280px"
}

# Layout with dynamic sidebar
app.layout = html.Div([
    # Toggle button - MUST be first for z-index layering
    html.Button(
        "☰ Filters",
        id="toggle-button",
        style=TOGGLE_BUTTON_STYLE,
        n_clicks=0
    ),
    
    # Sidebar
    html.Div([
        html.Div("NYC Collisions", className="sidebar-title"),
        html.H4("Filters", className="filters-title"),
        # Search input for automatic filter application
        html.Div([
            dcc.Input(
                id="search-input",
                type="text",
                placeholder="Type to search (e.g. 2011, injured, Manhattan)...",
                debounce=True,
                style={"width": "100%", "padding": "8px", "marginBottom": "12px", "borderRadius": "4px"}
            )
        ], style={"marginBottom": "12px"}),
        create_filters_grid(df_with_year, filter_config, sidebar=True)
    ], id="sidebar", style=SIDEBAR_STYLE),
    
    # Main content
    html.Div([
        html.H1("NYC Motor Vehicle Collisions Analysis"),
        html.Div(id="main-content")
    ], id="content", style=CONTENT_WITH_SIDEBAR)
], style={"display": "flex", "flexDirection": "column"})


# Callback to toggle sidebar
@app.callback(
    [Output("sidebar", "style"),
     Output("content", "style"),
     Output("toggle-button", "style")],
    [Input("toggle-button", "n_clicks")],
    prevent_initial_call=False
)
def toggle_sidebar(n_clicks):
    if n_clicks % 2 == 1:  # Odd clicks = hidden
        sidebar_hidden = {**SIDEBAR_HIDDEN}
        content = {**CONTENT_STYLE, "marginLeft": "0px"}
        button = {**TOGGLE_BUTTON_STYLE, "zIndex": 1001}  # Keep button on top
        return sidebar_hidden, content, button
    else:  # Even clicks = visible
        button = {**TOGGLE_BUTTON_STYLE, "zIndex": 1001}  # Keep button on top
        return SIDEBAR_STYLE, CONTENT_WITH_SIDEBAR, button


# -----------------------------
# Search -> auto-apply filters
# -----------------------------
# This callback listens to the search input and attempts to find matching
# values for each filter. It returns a value for each dropdown (or None to clear).
@app.callback(
    [
        Output("year-filter", "value"),
        Output("borough-filter", "value"),
        Output("person_type-filter", "value"),
        Output("person_injury-filter", "value"),
        Output("person_sex-filter", "value"),
        Output("position_in_vehicle-filter", "value"),
    ],
    [Input("search-input", "value")]
)
def apply_search_to_filters(search_value):
    # If no search or empty, clear all filters
    if not search_value:
        return [None, None, None, None, None, None]

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

    # token set - words and numbers only, lowercased
    tokens = [t.lower() for t in re.findall(r"[\w]+", str(search_value))]

    # Helper to pick a matching value from a column
    def _find_matches_for_column(column_name):
        """
        Return a list of candidate values (original types) matching any token.
        Matches exact token first, then substring matches. Preserves candidate order
        and avoids duplicates.
        """
        if df_with_year is None or column_name not in df_with_year.columns:
            return None
        unique_vals = df_with_year[column_name].dropna().unique()
        # build list of (label_lower, original_value)
        candidates = []
        for v in unique_vals:
            label = _format_label(v).lower()
            candidates.append((label, v))

        matched = []

        # exact matches first
        for token in tokens:
            for label, v in candidates:
                if label == token and v not in matched:
                    matched.append(v)

        # substring matches next
        for token in tokens:
            for label, v in candidates:
                if token in label and v not in matched:
                    matched.append(v)

        if not matched:
            return None
        return matched

    # Evaluate for each filter in the same order as outputs (returns lists or None)
    year_val = _find_matches_for_column("YEAR")
    borough_val = _find_matches_for_column("BOROUGH")
    person_type_val = _find_matches_for_column("PERSON_TYPE")
    person_injury_val = _find_matches_for_column("PERSON_INJURY")
    person_sex_val = _find_matches_for_column("PERSON_SEX")
    position_val = _find_matches_for_column("POSITION_IN_VEHICLE")

    return [year_val, borough_val, person_type_val, person_injury_val, person_sex_val, position_val]


if __name__ == "__main__":
    app.run(debug=True)


@app.callback(
    [
        Output("year-filter", "value"),
        Output("borough-filter", "value"),
        Output("person_type-filter", "value"),
        Output("person_injury-filter", "value"),
        Output("person_sex-filter", "value"),
        Output("position_in_vehicle-filter", "value"),
    ],
    [
        Input("year-filter", "value"),
        Input("borough-filter", "value"),
        Input("person_type-filter", "value"),
        Input("person_injury-filter", "value"),
        Input("person_sex-filter", "value"),
        Input("position_in_vehicle-filter", "value"),
    ],
    prevent_initial_call=True
)
def expand_all_selections(year_v, borough_v, ptype_v, pinjury_v, psex_v, position_v):
    """If a dropdown contains the special value "__ALL__", replace it
    with the full list of available values from `df_with_year` for that column.
    Returns the (possibly) expanded lists for each dropdown.
    """
    def _expand(val, column_name):
        if val is None:
            return None
        # ensure we work with lists (multi=True)
        vals = val if isinstance(val, (list, tuple)) else [val]
        if "__ALL__" in vals:
            if df_with_year is None or column_name not in df_with_year.columns:
                return None
            all_vals = df_with_year[column_name].dropna().unique().tolist()
            return all_vals
        # return original shape (list for multi)
        return list(vals)

    return [
        _expand(year_v, "YEAR"),
        _expand(borough_v, "BOROUGH"),
        _expand(ptype_v, "PERSON_TYPE"),
        _expand(pinjury_v, "PERSON_INJURY"),
        _expand(psex_v, "PERSON_SEX"),
        _expand(position_v, "POSITION_IN_VEHICLE"),
    ]



