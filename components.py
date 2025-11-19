"""
Reusable UI components for the NYC Collision Analysis Dashboard.
Scalable dropdown filters and other UI building blocks.
"""

from dash import dcc, html
import pandas as pd


# ============================================================================
# DROPDOWN FILTER BUILDERS
# ============================================================================


def create_dropdown_filter(df: pd.DataFrame, column_name: str, label: str = None, id_suffix: str = None):
    """
    Create a reusable dropdown filter for any column.
    """
    if df is None or column_name not in df.columns:
        return dcc.Dropdown(
            id=f"{id_suffix or column_name.lower()}-filter",
            options=[],
            placeholder=f"No {column_name} data available",
            disabled=True,
            className="custom-dropdown",
        )

    unique_vals = df[column_name].dropna().unique()

    try:
        unique_vals = sorted(unique_vals)
    except TypeError:
        unique_vals = list(unique_vals)

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

    options = [{"label": "All", "value": "__ALL__"}] + [
        {"label": _format_label(val), "value": val} for val in unique_vals
    ]

    display_label = label or column_name.replace("_", " ").title()
    filter_id = f"{id_suffix or column_name.lower()}-filter"

    return dcc.Dropdown(
        id=filter_id,
        options=options,
        placeholder=f"Select {display_label}",
        clearable=True,
        multi=True,
        className="custom-dropdown",
        style={
            "minWidth": "200px",
            "width": "100%",
        },
    )


def create_filters_grid(df: pd.DataFrame, columns_config: list, sidebar: bool = False):
    """
    Create a grid of dropdown filters.
    """
    filters = []

    for config in columns_config:
        column_name = config.get("column")
        label = config.get("label")
        id_suffix = config.get("id")

        if column_name:
            dropdown = create_dropdown_filter(df, column_name, label, id_suffix)

            if sidebar:
                filters.append(
                    html.Div(
                        [
                            html.Label(
                                label or column_name.replace("_", " ").title(),
                                style={
                                    "fontWeight": "600",
                                    "marginBottom": "6px",
                                    "display": "block",
                                    "fontSize": "12px",
                                },
                                className="filter-label",
                            ),
                            dropdown,
                        ],
                        className="filter-item",
                    )
                )
            else:
                filters.append(
                    html.Div(
                        [
                            html.Label(
                                label or column_name.replace("_", " ").title(),
                                style={
                                    "fontWeight": "bold",
                                    "marginBottom": "5px",
                                    "display": "block",
                                },
                            ),
                            dropdown,
                        ],
                        style={
                            "flex": "1",
                            "minWidth": "200px",
                            "marginRight": "20px",
                            "marginBottom": "15px",
                        },
                    )
                )

    if sidebar:
        # Vertical stack for sidebar with iOS-style spacing
        return html.Div(filters, className="filters-grid")
    else:
        # Horizontal wrapping grid for other pages if needed
        return html.Div(
            filters,
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "15px",
                "marginBottom": "20px",
                "padding": "15px",
            },
        )


# ============================================================================
# FILTER CONFIGURATION
# ============================================================================

FILTER_COLUMNS = [
    {"column": "BOROUGH", "label": "Borough"},
    {"column": "PERSON_TYPE", "label": "Person Type"},
    {"column": "PERSON_INJURY", "label": "Injury Type"},
    {"column": "PERSON_SEX", "label": "Person Sex"},
    {"column": "POSITION_IN_VEHICLE", "label": "Position in Vehicle"},
    {"column": "SAFETY_EQUIPMENT", "label": "Safety Equipment"},
    {"column": "BODILY_INJURY", "label": "Bodily Injury"},
]


def get_filter_config_with_year(df: pd.DataFrame):
    """
    Return filter configuration including YEAR if available.
    """
    config = FILTER_COLUMNS.copy()

    if df is not None and "CRASH_DATETIME" in df.columns:
        df_copy = df.copy()
        df_copy["CRASH_DATETIME"] = pd.to_datetime(df_copy["CRASH_DATETIME"], errors="coerce")
        df_copy["YEAR"] = df_copy["CRASH_DATETIME"].dt.year

        config.insert(0, {"column": "YEAR", "label": "Year"})

        return config, df_copy

    return config, df
