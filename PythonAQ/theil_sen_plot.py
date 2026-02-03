import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pymannkendall as mk
from scipy import stats # Scipy handles the CI for Theil-Sen correctly
from statsmodels.tsa.seasonal import STL

def theil_sen_plot(
    df,
    date_col='date_time',
    pollutant_col='pollutant',
    agg_freq=None,
    deseason=False,
    title='Theil-Sen Regression Plot',
    yaxis_title='Concentration',
    width=1000,
    height=600,
    deseason_data=None,
    alpha=0.05
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, pollutant_col]].dropna(subset=[date_col, pollutant_col])
    df.sort_values(by=date_col, inplace=True)

    if agg_freq is not None:
        df.set_index(date_col, inplace=True)
        df = df.resample(agg_freq)[[pollutant_col]].mean()
        df.reset_index(inplace=True)
        df.dropna(subset=[pollutant_col], inplace=True)

    # Deseasonalization (STL Decomposition)
    if deseason and len(df) >= 24:
        # Period 12 for monthly data
        stl = STL(df[pollutant_col], period=12, robust=True).fit()
        # openair uses trend + remainder for 'deseasoned'
        df['target'] = stl.trend + stl.resid
    else:
        df['target'] = df[pollutant_col]

    # Numeric date for regression (Days since 1970 to match R logic)
    df['date_numeric'] = (df[date_col] - pd.Timestamp("1970-01-01")).dt.days
    
    y_values = df['target'].values
    x_values = df['date_numeric'].values

    # --- 1. Get Significance from Mann-Kendall ---
    mk_res = mk.original_test(y_values, alpha=alpha)
    p_val = mk_res.p
    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "+" if p_val < 0.1 else ""

    # --- 2. Get Slope and Confidence Intervals from Scipy ---
    # stats.theilslopes returns: slope, intercept, lo_slope, up_slope
    # 'alpha' in scipy is the confidence level (e.g., 0.95), not the significance level.
    res = stats.theilslopes(y_values, x_values, alpha=(1 - alpha))
    slope, intercept, lo_slope, up_slope = res

    # Convert to annual rates
    slope_year = slope * 365.25
    upper_year = up_slope * 365.25
    lower_year = lo_slope * 365.25

    intercept_lo = np.median(y_values - lo_slope * x_values)
    intercept_up = np.median(y_values - up_slope * x_values)

    # Predict values for plotting
    y_pred = (slope * x_values) + intercept
    # Scipy's theilslopes provides the slope CI, we map that to the lines
    y_upper = (up_slope * x_values) + intercept_up
    y_lower = (lo_slope * x_values) + intercept_lo

    # Create figure
    fig = go.Figure()

    # Data Trace
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df['target'],
        mode='lines+markers', name='Data',
        line=dict(color='cornflowerblue', width=1.5),
        marker=dict(color='cornflowerblue', size=6)
    ))

    # Regression Line
    fig.add_trace(go.Scatter(
        x=df[date_col], y=y_pred,
        mode='lines', name='Sen-Theil Trend',
        line=dict(color='red', width=2)
    ))

    # Upper/Lower CI lines with Shading
    fig.add_trace(go.Scatter(
        x=df[date_col], y=y_upper,
        mode='lines', line=dict(color='red', dash='dot', width=1),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=df[date_col], y=y_lower,
        mode='lines', line=dict(color='red', dash='dot', width=1),
        fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)',
        name='CI', showlegend=False
    ))

    # Summary Annotation (Replicates openair style)
    rate_text = f"Trend: {slope_year:.3f} [{lower_year:.3f}, {upper_year:.3f}] units/year {stars}"
    fig.add_annotation(
        xref="paper", yref="paper", x=0.5, y=1.05,
        text=rate_text, showarrow=False,
        font=dict(size=14, color='darkgreen'),
        bgcolor='white', bordercolor='darkgreen', borderwidth=1
    )

    # Background shading for years
    years = df[date_col].dt.year.unique()
    for i, year in enumerate(years):
        if i % 2 == 0:
            fig.add_vrect(
                x0=f"{year}-01-01", x1=f"{year+1}-01-01",
                fillcolor="rgba(128, 128, 128, 0.1)", layer="below", line_width=0,
            )

    fig.update_layout(
        title=title, xaxis_title='Date', yaxis_title=yaxis_title,
        width=width, height=height, template='plotly_white',
        hovermode='x unified',
        legend=dict(
            x=1,
            y=1,
            xanchor='right',
            yanchor='top'
        )
    )

    return fig
