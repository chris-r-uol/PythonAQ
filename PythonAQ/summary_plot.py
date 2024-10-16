import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def summary_plot(df, colours = {'ts':'dodgerblue', 'rug':'crimson', 'histogram':'orange'}):
    """
    Creates a comprehensive data summary plot using Plotly for the given DataFrame
    and returns summary statistics as a separate DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data.
    - colours (dict): Dictionary of colours with keys ts, rug, histogram
    
    Returns:
    - fig (plotly.graph_objects.Figure): The resulting Plotly figure.
    - summary_df (pd.DataFrame): DataFrame containing summary statistics for each column.
    """
    
    # Ensure 'date_time' is in datetime format
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # Remove specified columns
    columns_to_remove = ['site', 'code', 'date']
    df_clean = df.drop(columns=columns_to_remove, errors='ignore')
    
    # Identify columns of interest and sort them for consistent ordering
    columns_of_interest = sorted(df_clean.columns.drop('date_time'))
    
    # Determine the number of columns to plot
    num_cols = len(columns_of_interest)
    
    # Initialize list to store summary statistics
    summary_list = []
    
    # Create subplots: 2 columns per variable (Time Series + Histogram)
    # Each variable gets its own row
    subplot_titles = []
    for col in columns_of_interest:
        subplot_titles.append(f"{col} Time Series")
        subplot_titles.append(f"{col} Histogram")
    
    fig = make_subplots(
        rows=num_cols, cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        column_widths=[0.7, 0.3],
        specs=[[{"type": "scatter"}, {"type": "histogram"}] for _ in range(num_cols)]
    )
    
    # Iterate over each column of interest and add traces
    for idx, col in enumerate(columns_of_interest, start=1):
        # Extract relevant data
        df_col = df_clean[['date_time', col]].copy()
        
        # Compute summary statistics
        total_points = len(df_col)
        non_missing = df_col[col].notna().sum()
        missing = df_col[col].isna().sum()
        capture_rate = (non_missing / total_points) * 100
        min_val = df_col[col].min()
        max_val = df_col[col].max()
        mean_val = df_col[col].mean()
        median_val = df_col[col].median()
        percentile_95 = df_col[col].quantile(0.95)
        
        # Append summary statistics to the list
        summary_list.append({
            'Column': col,
            'Capture Rate (%)': round(capture_rate, 2),
            'Missing Points': missing,
            'Min': round(min_val, 2),
            'Max': round(max_val, 2),
            'Mean': round(mean_val, 2),
            'Median': round(median_val, 2),
            '95th Percentile': round(percentile_95, 2)
        })
        
        # Time Series Trace
        fig.add_trace(
            go.Scatter(
                x=df_col['date_time'],
                y=df_col[col],
                mode='lines',
                name=f"{col} Value",
                line=dict(color=colours['ts'])
            ),
            row=idx, col=1
        )
        
        # Rug Plot for Missing Data
        missing_dates = df_col['date_time'][df_col[col].isna()]
        if not missing_dates.empty:
            fig.add_trace(
                go.Scatter(
                    x=missing_dates,
                    y=[df_col[col].min() - (df_col[col].max() - df_col[col].min()) * 0.05] * len(missing_dates),
                    mode='markers',
                    name='Missing Data',
                    marker=dict(color=colours['rug'], size=5, symbol='line-ns-open'),
                    showlegend=False
                ),
                row=idx, col=1
            )
        
        # Histogram Trace
        fig.add_trace(
            go.Histogram(
                x=df_col[col],
                name=f"{col} Histogram",
                marker_color=colours['histogram'],
                opacity=0.75,
                nbinsx=30
            ),
            row=idx, col=2
        )
    
    # Update layout for better appearance
    fig.update_layout(
        height=300 * num_cols,  # Adjust height based on number of plots
        width=1200,
        title_text="Data Summary Plot",
        showlegend=False
    )
    
    # Update x-axis and y-axis titles for time series and histograms
    for idx, col in enumerate(columns_of_interest, start=1):
        # Time Series Plot Axes
        fig.update_xaxes(title_text="Date Time", row=idx, col=1)
        fig.update_yaxes(title_text=col, row=idx, col=1)
        
        # Histogram Axes
        fig.update_xaxes(title_text=col, row=idx, col=2)
        fig.update_yaxes(title_text="Count", row=idx, col=2)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_list)
    
    return fig, summary_df