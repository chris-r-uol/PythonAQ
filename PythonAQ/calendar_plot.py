import pandas as pd
import plotly.graph_objects as go
from plotly_calplot import calplot

def calendar(
    data: pd.DataFrame,
    value_column: str,
    date_column: str = 'date_time',
    aggregation_method: str = 'mean',
    colorscale: str = 'Viridis'
) -> go.Figure:
    """
    Creates a calendar heatmap visualization of the provided data.

    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame containing the data to plot.
    value_column : str
        The name of the column containing the values to aggregate and plot.
    date_column : str, optional
        The name of the column containing datetime information. Default is 'date'.
    aggregation_method : str, optional
        The aggregation method to use ('mean', 'sum', etc.). Default is 'mean'.
    colorscale : str, optional
        The colorscale to use for the plot. Default is 'Viridis'.

    Returns:
    --------
    go.Figure
        A Plotly Figure object containing the calendar heatmap.
    """
    # Check if the date column exists
    if date_column not in data.columns:
        raise ValueError(f"Date column '{date_column}' not found in the DataFrame.")

    # Check if the value column exists
    if value_column not in data.columns:
        raise ValueError(f"Value column '{value_column}' not found in the DataFrame.")

    # Create a copy to avoid modifying the original DataFrame
    df = data.copy()

    # Ensure the date column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e:
            raise ValueError(f"Unable to convert column '{date_column}' to datetime: {e}")

    # Resample data to daily frequency using the specified aggregation method
    daily_data = df.resample('D', on=date_column)[value_column].agg(aggregation_method).reset_index()

    # Determine the number of years to adjust the plot height
    n_years = df[date_column].dt.year.nunique()

    # Create the calendar heatmap
    fig = calplot(
        daily_data,
        x=date_column,
        y=value_column,
        colorscale=colorscale,
        dark_theme=False
    )

    # Adjust the figure height based on the number of years
    fig.update_layout(height=270 * n_years)

    return fig