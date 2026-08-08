"""Flexible scatter plots. Port of openair's ``scatterPlot``."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from .text import quick_text
from .faceting import conditionable

__all__ = ['scatter_plot']


def _lowess(x, y, frac=0.3, n_points=200):
    """LOWESS smoother, via statsmodels when present, else a binned fallback."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        order = np.argsort(x)
        bins = np.array_split(order, min(n_points, max(1, len(x) // 10)))
        return (np.array([x[b].mean() for b in bins if len(b)]),
                np.array([y[b].mean() for b in bins if len(b)]))
    fitted = lowess(y, x, frac=frac, return_sorted=True)
    return fitted[:, 0], fitted[:, 1]


@conditionable
def scatter_plot(df, x, y, method='scatter', colour_by=None, linear=False,
                 smooth=False, frac=0.3, bins=40, one_to_one=False,
                 colorscale='Viridis', marker_size=5, opacity=0.6,
                 title=None, width=900, height=700):
    """Scatter, hexbin or density plot of two variables, with optional fits.

    Parameters:
    - df (pd.DataFrame): Input data.
    - x, y (str): Column names for the axes.
    - method (str): 'scatter', 'hexbin' (2-D histogram) or 'density'.
    - colour_by (str or None): Column used to colour points, for 'scatter'.
    - linear (bool): Add a least-squares line, annotated with slope and R^2.
    - smooth (bool): Add a LOWESS smooth line.
    - frac (float): LOWESS smoothing span.
    - bins (int): Bin count for 'hexbin' and 'density'.
    - one_to_one (bool): Draw the 1:1 line, useful for model evaluation.
    - colorscale (str): Plotly colour scale name.
    - marker_size (int), opacity (float): Marker styling for 'scatter'.
    - title (str or None): Plot title.
    - width, height (int): Figure size in pixels.

    Returns:
    - fig (go.Figure): The scatter plot.
    - summary (pd.DataFrame): Fit statistics; empty if no fit was requested.
    """
    for column in (x, y):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
    if method not in ('scatter', 'hexbin', 'density'):
        raise ValueError("method must be 'scatter', 'hexbin' or 'density'.")

    columns = [x, y] + ([colour_by] if colour_by else [])
    data = df[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        raise ValueError(f"No complete '{x}'/'{y}' pairs to plot.")

    xv = data[x].to_numpy(dtype=float)
    yv = data[y].to_numpy(dtype=float)

    fig = go.Figure()
    if method == 'scatter':
        marker = dict(size=marker_size, opacity=opacity)
        if colour_by:
            marker.update(color=data[colour_by], colorscale=colorscale,
                          showscale=True, colorbar=dict(title=colour_by))
        else:
            marker.update(color='steelblue')
        fig.add_trace(go.Scattergl(
            x=xv, y=yv, mode='markers', name='data', marker=marker,
            hovertemplate=f'{x}: %{{x:.2f}}<br>{y}: %{{y:.2f}}<extra></extra>',
        ))
    else:
        fig.add_trace(go.Histogram2d(
            x=xv, y=yv, nbinsx=bins, nbinsy=bins, colorscale=colorscale,
            colorbar=dict(title='count'),
            histnorm='probability density' if method == 'density' else None,
        ))

    records = []
    if linear:
        fit = stats.linregress(xv, yv)
        grid = np.linspace(xv.min(), xv.max(), 100)
        fig.add_trace(go.Scatter(
            x=grid, y=fit.slope * grid + fit.intercept, mode='lines',
            name='linear fit', line=dict(color='crimson', width=2),
        ))
        records.append({
            'fit': 'linear', 'slope': fit.slope, 'intercept': fit.intercept,
            'r': fit.rvalue, 'r_squared': fit.rvalue ** 2,
            'p_value': fit.pvalue, 'std_err': fit.stderr, 'n': len(xv),
        })
        fig.add_annotation(
            xref='paper', yref='paper', x=0.02, y=0.98, showarrow=False,
            text=(f'y = {fit.slope:.3f}x + {fit.intercept:.3f}<br>'
                  f'R&#178; = {fit.rvalue ** 2:.3f}, n = {len(xv)}'),
            align='left', bgcolor='rgba(255,255,255,0.75)',
            bordercolor='crimson', borderwidth=1,
        )

    if smooth:
        sx, sy = _lowess(xv, yv, frac=frac)
        fig.add_trace(go.Scatter(
            x=sx, y=sy, mode='lines', name='smooth',
            line=dict(color='darkorange', width=3),
        ))

    if one_to_one:
        low = float(min(xv.min(), yv.min()))
        high = float(max(xv.max(), yv.max()))
        fig.add_trace(go.Scatter(
            x=[low, high], y=[low, high], mode='lines', name='1:1',
            line=dict(color='grey', width=1, dash='dash'),
        ))

    fig.update_layout(
        title=title or f'{quick_text(y)} vs {quick_text(x)}',
        xaxis_title=quick_text(x), yaxis_title=quick_text(y),
        template='plotly_white', width=width, height=height,
    )
    return fig, pd.DataFrame(records)
