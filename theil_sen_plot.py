import pandas as pd
import numpy as np
from sklearn.linear_model import TheilSenRegressor
import plotly.graph_objects as go

def theil_sen_plot(
    df,
    date_col='date_time',
    pollutant_col='pollutant',
    agg_freq=None,
    title='Theil-Sen Regression Plot',
    yaxis_title='Concentration',
    width=1000,
    height=600
):
    """
    Plots time series data with Theil-Sen regression analysis.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the data.
    - date_col (str): Name of the date/time column in df.
    - pollutant_col (str): Name of the pollutant column to analyze.
    - agg_freq (str): Resampling frequency (e.g., 'D' for daily, 'M' for monthly). If None, no aggregation.
    - title (str): Title of the plot.
    - yaxis_title (str): Label for the y-axis.
    - width (int): Width of the figure.
    - height (int): Height of the figure.

    Returns:
    - fig (plotly.graph_objects.Figure): The resulting plot.
    """
    # Ensure date_col is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Select only date_col and pollutant_col
    df = df[[date_col, pollutant_col]]

    # Drop rows with missing values in date_col or pollutant_col
    df.dropna(subset=[date_col, pollutant_col], inplace=True)

    # Sort the DataFrame by date
    df.sort_values(by=date_col, inplace=True)

    # Aggregate data if aggregation frequency is specified
    if agg_freq is not None:
        df.set_index(date_col, inplace=True)
        # Resample and compute mean only on the pollutant_col
        df = df.resample(agg_freq)[[pollutant_col]].mean()
        df.reset_index(inplace=True)

        # Drop NaNs in pollutant_col after resampling
        df.dropna(subset=[pollutant_col], inplace=True)

    # Convert date to numeric for regression (ordinal)
    df['date_numeric'] = df[date_col].map(pd.Timestamp.toordinal)

    # Drop any rows with NaNs in 'date_numeric' or 'pollutant_col'
    df.dropna(subset=['date_numeric', pollutant_col], inplace=True)

    # Check if there's any data left
    if df.empty:
        raise ValueError("No data available for regression after removing NaN values.")

    # Prepare data for regression
    X = df['date_numeric'].values.reshape(-1, 1)
    y = df[pollutant_col].values

    # Perform Theil-Sen regression
    ts_regressor = TheilSenRegressor()
    ts_regressor.fit(X, y)
    y_pred = ts_regressor.predict(X)

    # Calculate slope per year
    days_in_year = 365.25
    slope_per_year = ts_regressor.coef_[0] * days_in_year
    intercept = ts_regressor.intercept_

    # Calculate upper and lower bounds (approximate using residuals)
    residuals = y - y_pred
    residual_std = np.std(residuals)
    y_upper = y_pred + 1.96 * residual_std
    y_lower = y_pred - 1.96 * residual_std

    # Create figure
    fig = go.Figure()

    # Plot original data
    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[pollutant_col],
            mode='lines+markers',
            name='Data',
            line=dict(color='dodgerblue'),
            marker=dict(size=5),
            showlegend=True
        )
    )

    # Plot regression line
    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=y_pred,
            mode='lines',
            name='Theil-Sen Regression',
            line=dict(color='red'),
            showlegend=True
        )
    )

    # Plot upper confidence bound
    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=y_upper,
            mode='lines',
            name='Upper Bound',
            line=dict(color='red', dash='dash'),
            showlegend=False
        )
    )

    # Plot lower confidence bound and fill area between bounds
    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=y_lower,
            mode='lines',
            name='Lower Bound',
            line=dict(color='red', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.1)',
            showlegend=False
        )
    )

    # Add rate per year as annotation
    rate_text = f"Rate of Change: {slope_per_year:.3f} units/year"
    fig.add_annotation(
        x=df[date_col].iloc[len(df) // 2],
        y=max(y_pred)+10,
        text=rate_text,
        showarrow=False,
        font=dict(size=12, color='darkgreen'),
        bgcolor='rgba(255,255,255,0.7)',
        bordercolor='darkgreen',
        borderwidth=1
    )

    # Alternate background colors for each year
    years = df[date_col].dt.year.unique()
    shapes = []
    for i, year in enumerate(years):
        start_date = pd.Timestamp(year=year, month=1, day=1)
        end_date = pd.Timestamp(year=year + 1, month=1, day=1)
        color = 'rgba(70, 70, 70, 0.2)' if i % 2 == 0 else 'rgba(255, 255, 255, 0)'
        shapes.append(
            dict(
                type='rect',
                xref='x',
                yref='paper',
                x0=start_date,
                y0=0,
                x1=end_date,
                y1=1,
                fillcolor=color,
                opacity=0.5,
                layer='below',
                line_width=0,
            )
        )

    fig.update_layout(
        legend=dict(
            x=0.95,
            y=0.95,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.5)'),  # Semi-transparent background,
        title=title,
        xaxis_title='Date',
        yaxis_title=yaxis_title,
        width=width,
        height=height,
        shapes=shapes,
        hovermode='x unified',
        template='plotly_white'
    )

    return fig
