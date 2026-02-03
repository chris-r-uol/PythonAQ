import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.tsa.seasonal import STL
import pymannkendall as mk
from scipy import stats
import statsmodels.api as sm

def smooth_trend_plot(
    df,
    date_col='date_time',
    pollutant_col='pollutant',
    avg_freq='MS', 
    deseason=False,
    alpha=0.05,
    title='Non-parametric Smooth Trend (GAM)',
    width=1000,
    height=600
):
    # 1. Data Prep
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, pollutant_col]].dropna()
    df.sort_values(by=date_col, inplace=True)

    # 2. Monthly Averaging
    df = df.set_index(date_col).resample(avg_freq).mean().reset_index()
    df.dropna(inplace=True)

    # 3. Deseasonalization
    if deseason and len(df) >= 24:
        stl = STL(df[pollutant_col], period=12, robust=True).fit()
        df['target'] = stl.trend + stl.resid
    else:
        df['target'] = df[pollutant_col]

    # 4. Numeric X for GAM
    start_date = df[date_col].min()
    df['days_since_start'] = (df[date_col] - start_date).dt.days
    X = df[['days_since_start']] # Must be 2D for BSplines
    y = df['target']

    # 5. Fit GAM using statsmodels
    # Fixed API: 'df' instead of 'df_rb'
    bs = BSplines(X, df=[5], degree=[4])
    # Adding sm.add_constant ensures the 'baseline' concentration is captured
    gam_model = GLMGam(y, sm.add_constant(X), smoother=bs).fit()
    
    # 6. Generate Smooth Prediction Grid
    x_grid_val = np.linspace(X.values.min(), X.values.max(), 200).flatten()
    
    # We must add a constant to our prediction grid to match the model structure
    x_grid_predict = sm.add_constant(x_grid_val)
    
    date_grid = [start_date + pd.Timedelta(days=int(d)) for d in x_grid_val]
    
    # Get predictions
    # exog_smooth takes the spline part, exog takes the constant part
    predictions = gam_model.get_prediction(exog=x_grid_predict, exog_smooth=x_grid_val)
    y_pred = predictions.predicted_mean
    y_std = predictions.se_mean 
    
    z = stats.norm.ppf(1 - alpha/2)
    y_lower = y_pred - z * y_std
    y_upper = y_pred + z * y_std

    # 7. Significance Check
    mk_res = mk.original_test(df['target'])
    stars = "***" if mk_res.p < 0.001 else "**" if mk_res.p < 0.01 else "*" if mk_res.p < 0.05 else ""

    # 8. Create Plot
    fig = go.Figure()

    # Alternate Background Shading (openair style)
    years = df[date_col].dt.year.unique()
    for i, year in enumerate(years):
        if i % 2 == 0:
            fig.add_vrect(
                x0=f"{year}-01-01", x1=f"{year+1}-01-01",
                fillcolor="rgba(128, 128, 128, 0.1)", layer="below", line_width=0,
            )

    # Monthly Means
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df['target'],
        mode='lines+markers', name='Monthly Mean',
        line=dict(color='indianred', width=1.5),
        marker=dict(color='indianred', size=6)
    ))

    # GAM Trend Line
    fig.add_trace(go.Scatter(
        x=date_grid, y=y_pred,
        mode='lines', name='GAM Trend',
        line=dict(color='indianred', width=3)
    ))

    # Shaded Confidence Interval
    fig.add_trace(go.Scatter(
        x=date_grid, y=y_upper,
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=date_grid, y=y_lower,
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(205, 92, 92, 0.2)',
        name=f'{(1-alpha)*100}% CI', showlegend=True
    ))

    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        text=f"Trend Significance: {stars if stars else 'N.S.'} (p={mk_res.p:.3f})",
        showarrow=False, font=dict(color="indianred", size=12),
        bgcolor="white", bordercolor="indianred", borderwidth=1
    )

    fig.update_layout(
        title=title, xaxis_title='Date', yaxis_title=f"Concentration ({pollutant_col})",
        template='plotly_white', width=width, height=height,
        hovermode='x unified',
        legend=dict(x=1, y=1, xanchor='right', yanchor='top', bgcolor='rgba(255,255,255,0.6)')
    )

    return fig
