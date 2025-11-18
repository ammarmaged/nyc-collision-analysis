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
    
    Args:
        df: DataFrame containing the column
        column_name: Name of the column to filter on
        label: Display label (defaults to column_name if not provided)
        id_suffix: Suffix for the component ID (defaults to column_name if not provided)
        
    Returns:
        dcc.Dropdown component
    """
    if df is None or column_name not in df.columns:
        return dcc.Dropdown(
            id=f"{id_suffix or column_name.lower()}-filter",
            options=[],
            placeholder=f"No {column_name} data available",
            disabled=True
        )
    
    # Get unique values, sorted
    unique_vals = df[column_name].dropna().unique()

    # Try to sort if possible (handles both strings and numbers)
    try:
        unique_vals = sorted(unique_vals)
    except TypeError:
        unique_vals = list(unique_vals)

    def _format_label(v):
        # Convert NaNs or None to empty string
        if v is None:
            return ""
        # If it's a pandas Timestamp or datetime-like, format nicely
        try:
            import pandas as _pd
            if isinstance(v, (_pd.Timestamp,)):
                return v.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

        s = str(v).strip()
        # Remove leading underscores and surrounding whitespace
        while s.startswith("_"):
            s = s[1:]
        s = s.replace("_", " ")

        # If numeric string like '2011.0', convert to int-like representation
        try:
            if "." in s:
                f = float(s)
                if f.is_integer():
                    return str(int(f))
        except Exception:
            pass

        return s

    options = [
        {"label": "All", "value": "__ALL__"},
    ] + [
        {"label": _format_label(val), "value": val}
        for val in unique_vals
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
            "padding": "8px",
            "borderRadius": "4px",
            "border": "1px solid #ccc",
            "fontSize": "13px"
        }
    )


def create_filters_grid(df: pd.DataFrame, columns_config: list, sidebar: bool = False):
    """
    Create a grid of dropdown filters.
    
    Args:
        df: DataFrame containing the data
        columns_config: List of dicts with keys:
                       - 'column': column name (required)
                       - 'label': display label (optional)
                       - 'id': custom ID suffix (optional)
        sidebar: If True, format filters for sidebar display (vertical stack)
                       
    Example:
        columns_config = [
            {'column': 'BOROUGH', 'label': 'Borough'},
            {'column': 'PERSON_TYPE', 'label': 'Person Type'},
            {'column': 'PERSON_INJURY', 'label': 'Injury Type'},
        ]
        
    Returns:
        html.Div with a responsive grid of dropdowns
    """
    filters = []
    
    for config in columns_config:
        column_name = config.get('column')
        label = config.get('label')
        id_suffix = config.get('id')
        
        if column_name:
            dropdown = create_dropdown_filter(df, column_name, label, id_suffix)
            
            if sidebar:
                # Sidebar styling: vertical stack with larger spacing
                filters.append(
                    html.Div([
                        html.Label(label or column_name.replace("_", " ").title(), 
                                  style={
                                      "fontWeight": "bold",
                                      "marginBottom": "8px",
                                      "display": "block",
                                      "color": "#ecf0f1",
                                      "fontSize": "13px"
                                  }),
                        dropdown
                    ], style={
                        "marginBottom": "20px",
                        "width": "100%"
                    })
                )
            else:
                # Grid styling: horizontal wrapping
                filters.append(
                    html.Div([
                        html.Label(label or column_name.replace("_", " ").title(), 
                                  style={"fontWeight": "bold", "marginBottom": "5px", "display": "block"}),
                        dropdown
                    ], style={"flex": "1", "minWidth": "200px", "marginRight": "20px", "marginBottom": "15px"})
                )
    
    if sidebar:
        # Vertical stack for sidebar
        return html.Div(
            filters,
            style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "10px"
            }
        )
    else:
        # Horizontal wrapping grid
        return html.Div(
            filters,
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "15px",
                "marginBottom": "20px",
                "padding": "15px",
                "backgroundColor": "#f8f9fa",
                "borderRadius": "5px"
            }
        )


# ============================================================================
# FILTER CONFIGURATION (Easy to maintain and extend)
# ============================================================================

FILTER_COLUMNS = [
    {'column': 'BOROUGH', 'label': 'Borough'},
    {'column': 'PERSON_TYPE', 'label': 'Person Type'},
    {'column': 'PERSON_INJURY', 'label': 'Injury Type'},
    {'column': 'PERSON_SEX', 'label': 'Person Sex'},
    {'column': 'POSITION_IN_VEHICLE', 'label': 'Position in Vehicle'},
    {'column': 'SAFETY_EQUIPMENT', 'label': 'Safety Equipment'},
    {'column': 'BODILY_INJURY', 'label': 'Bodily Injury'},
]

# Add YEAR filter if CRASH_DATETIME exists
def get_filter_config_with_year(df: pd.DataFrame):
    """
    Return filter configuration including YEAR if available.
    """
    config = FILTER_COLUMNS.copy()
    
    # Extract year from CRASH_DATETIME if it exists
    if df is not None and "CRASH_DATETIME" in df.columns:
        df_copy = df.copy()
        df_copy["CRASH_DATETIME"] = pd.to_datetime(df_copy["CRASH_DATETIME"], errors="coerce")
        df_copy["YEAR"] = df_copy["CRASH_DATETIME"].dt.year
        
        # Add YEAR at the beginning
        config.insert(0, {'column': 'YEAR', 'label': 'Year'})
        
        return config, df_copy
    
    return config, df


