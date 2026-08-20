"""Relationships between two pollutants over time. Port of openair's ``linearRelation``.

Fits y on x separately in each time period and plots how the slope moves. The
slope between two pollutants is usually more informative than either series on
its own, because it describes the source rather than the amount: the NOx to
NO2 relationship changes when fleets change, and the SO2 to NOx ratio
separates combustion of one fuel from another. A change in slope is a change
in what is emitting, not merely in how much.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .text import quick_text

__all__ = ['linear_relation']

_PERIODS = {'year': 'YS', 'season': 'QS-DEC', 'month': 'MS', 'week': 'W',
            'day': 'D', 'hour': 'h'}


def linear_relation(df, x='NOx', y='NO2', period='month', date_col='date_time',
                    min_points=20, intercept=True, condition=None,
                    colour='#1f77b4', title=None, width=1000, height=520):
    """Fit y ~ x within each period and plot the slope over time.

    Parameters:
    - df (pd.DataFrame): Input data.
    - x, y (str): Predictor and response columns.
    - period (str): 'year', 'season', 'month', 'week', 'day' or 'hour'.
    - date_col (str): Datetime column.
    - min_points (int): Complete pairs a period needs before it is fitted.
      Periods with fewer are dropped rather than fitted badly.
    - intercept (bool): Fit an intercept. False forces the line through the
      origin, which is the right choice only when y must be zero when x is.
    - condition (str or None): Column to split by, giving one line per level.
    - colour (str): Line colour when `condition` is None.
    - title (str or None): Plot title.
    - width, height (int): Figure size in pixels.

    Returns:
    - fig (go.Figure): Slope against time, with a 95% confidence band.
    - summary (pd.DataFrame): Per period: slope, intercept, standard error,
      r squared and n.

    Notes:
    - The confidence band is the standard error of the slope within each
      period. It describes the fit, not the measurement: a tight band on a
      badly specified relationship is still tight.
    - The regression is ordinary least squares, which assumes the error is in
      y alone. Where both pollutants are measured with comparable relative
      error the slope is biased towards zero, so read changes in it rather
      than its absolute value.
    """
    for col in (x, y):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in the DataFrame.")
    if period not in _PERIODS:
        raise ValueError(f'period must be one of {sorted(_PERIODS)}.')
    if condition is not None and condition not in df.columns:
        raise ValueError(f"Column '{condition}' not found in the DataFrame.")
    if min_points < 3:
        raise ValueError('min_points must be at least 3.')

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    groups = ([(None, data)] if condition is None
              else list(data.groupby(condition, observed=True, sort=True)))

    rows = []
    for level, frame in groups:
        frame = frame.dropna(subset=[x, y, date_col])
        if frame.empty:
            continue
        for stamp, block in frame.groupby(
                pd.Grouper(key=date_col, freq=_PERIODS[period])):
            fit = _fit(block[x].to_numpy(float), block[y].to_numpy(float),
                       intercept, min_points)
            if fit is None:
                continue
            rows.append({'date': stamp, 'level': level, **fit})

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError(
            f'No {period} had at least {min_points} complete {x}/{y} pairs.')

    fig = go.Figure()
    levels = [None] if condition is None else sorted(summary['level'].unique(),
                                                     key=str)
    palette = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']
    for index, level in enumerate(levels):
        block = (summary if level is None
                 else summary[summary['level'] == level]).sort_values('date')
        line_colour = colour if level is None else palette[index % len(palette)]
        name = 'slope' if level is None else str(level)
        upper = block['slope'] + 1.96 * block['slope_se']
        lower = block['slope'] - 1.96 * block['slope_se']
        fig.add_trace(go.Scatter(
            x=list(block['date']) + list(block['date'][::-1]),
            y=list(upper) + list(lower[::-1]),
            fill='toself', fillcolor=_translucent(line_colour),
            line=dict(width=0), hoverinfo='skip', showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=block['date'], y=block['slope'], mode='lines+markers', name=name,
            line=dict(color=line_colour, width=2), marker=dict(size=6),
            customdata=np.column_stack([block['r_squared'], block['n']]),
            hovertemplate=('%{x|%Y-%m-%d}<br>slope %{y:.3f}'
                           '<br>r² %{customdata[0]:.3f}'
                           '<br>n %{customdata[1]}<extra></extra>'),
        ))

    fig.update_layout(
        title=(title if title is not None
               else f'{quick_text(y)} against {quick_text(x)} by {period}'),
        xaxis_title='', yaxis_title=f'slope, {quick_text(y)} per {quick_text(x)}',
        template='plotly_white', width=width, height=height,
        showlegend=condition is not None,
    )
    return fig, summary


def _fit(x, y, intercept, min_points):
    """Least squares fit of one period, or None if it cannot be fitted."""
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = len(x)
    if n < min_points:
        return None
    # A period where the predictor never varies has no slope to estimate, and
    # the normal equations are singular rather than merely imprecise.
    if np.ptp(x) == 0:
        return None

    design = np.column_stack([x, np.ones(n)]) if intercept else x[:, None]
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    slope = float(coefficients[0])
    constant = float(coefficients[1]) if intercept else 0.0

    residual = y - design @ coefficients
    dof = n - design.shape[1]
    if dof <= 0:
        return None
    variance = float(residual @ residual) / dof
    # Standard error of the slope from the diagonal of the covariance matrix.
    gram_inverse = np.linalg.pinv(design.T @ design)
    slope_se = float(np.sqrt(max(variance * gram_inverse[0, 0], 0.0)))

    total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - float(residual @ residual) / total if total > 0 else np.nan
    return {'slope': slope, 'intercept': constant, 'slope_se': slope_se,
            'r_squared': r_squared, 'n': n}


def _translucent(colour, alpha=0.18):
    if isinstance(colour, str) and colour.startswith('#') and len(colour) == 7:
        r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
        return f'rgba({r},{g},{b},{alpha})'
    return 'rgba(120,120,120,0.18)'
