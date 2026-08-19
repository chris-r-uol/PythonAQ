"""Rolling multiple regression. Port of openair's ``runRegression``.

Fits the same model repeatedly over a sliding window, so that a coefficient
becomes a series rather than a single number. A relationship estimated once
over four years is an average over whatever changed during them; this shows
whether it held still.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .text import quick_text

__all__ = ['run_regression']


def run_regression(df, y='NO2', x=('NOx', 'ws', 'temp'), window=168, step=24,
                   date_col='date_time', min_points=None, intercept=True,
                   standardise=False, title=None, width=1000, panel_height=240):
    """Regress y on several predictors over a sliding window.

    Parameters:
    - df (pd.DataFrame): Input data.
    - y (str): Response column.
    - x (list of str): Predictor columns.
    - window (int): Window width in observations. The default of 168 is one
      week of hourly data.
    - step (int): Observations between successive fits. The default of 24
      advances a day at a time, which is enough for a weekly window.
    - date_col (str): Datetime column, used to order the data and label fits.
    - min_points (int or None): Complete rows a window needs to be fitted.
      None requires three quarters of the window.
    - intercept (bool): Fit an intercept.
    - standardise (bool): Divide each predictor by its overall standard
      deviation first, so coefficients are comparable between predictors
      measured in different units. Changes the scale of the coefficients, not
      the fit.
    - title (str or None): Plot title.
    - width (int), panel_height (int): Figure width and per-panel height.

    Returns:
    - fig (go.Figure): One panel per predictor, coefficient against time, with
      a 95% confidence band.
    - summary (pd.DataFrame): One row per window: the coefficients, their
      standard errors, r squared and n.

    Notes:
    - Successive windows overlap by `window - step` observations, so
      neighbouring points share most of their data and are not independent.
      The series will look smoother than the evidence warrants; read the level
      and large moves, not the wiggles.
    - Correlated predictors split their shared effect between them
      arbitrarily, and the split can move from window to window even when the
      underlying relationship does not. Two coefficients that mirror each
      other are the usual sign of this.
    """
    predictors = [x] if isinstance(x, str) else list(x)
    for col in [y, *predictors]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in the DataFrame.")
    if window < len(predictors) + 2:
        raise ValueError(
            f'window must exceed the number of predictors; {window} is too '
            f'small for {len(predictors)} of them.')
    if step < 1:
        raise ValueError('step must be at least 1.')

    data = df[[date_col, y, *predictors]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)

    if standardise:
        for col in predictors:
            spread = data[col].std(ddof=1)
            if spread and np.isfinite(spread):
                data[col] = data[col] / spread

    if min_points is None:
        min_points = max(int(0.75 * window), len(predictors) + 2)

    names = (['intercept'] if intercept else []) + predictors
    rows = []
    for start in range(0, len(data) - window + 1, step):
        block = data.iloc[start:start + window]
        fit = _fit_window(block, y, predictors, intercept, min_points)
        if fit is None:
            continue
        # Labelled at the centre of the window: labelling at the start or end
        # would shift every feature by half a window against the series it is
        # meant to explain.
        rows.append({'date': block[date_col].iloc[len(block) // 2], **fit})

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError(
            f'No window of {window} rows had {min_points} complete cases. '
            'Widen the window, lower min_points, or check for gaps.')

    fig = make_subplots(rows=len(predictors), cols=1, shared_xaxes=True,
                        vertical_spacing=0.06,
                        subplot_titles=[quick_text(p) for p in predictors])
    palette = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']
    for index, name in enumerate(predictors):
        colour = palette[index % len(palette)]
        upper = summary[name] + 1.96 * summary[f'{name}_se']
        lower = summary[name] - 1.96 * summary[f'{name}_se']
        fig.add_trace(go.Scatter(
            x=list(summary['date']) + list(summary['date'][::-1]),
            y=list(upper) + list(lower[::-1]), fill='toself',
            fillcolor=_translucent(colour), line=dict(width=0),
            hoverinfo='skip', showlegend=False,
        ), row=index + 1, col=1)
        fig.add_trace(go.Scatter(
            x=summary['date'], y=summary[name], mode='lines', name=name,
            line=dict(color=colour, width=2), showlegend=False,
            hovertemplate=('%{x|%Y-%m-%d %H:%M}<br>coefficient %{y:.4g}'
                           '<extra></extra>'),
        ), row=index + 1, col=1)
        # Zero is the reference: a coefficient whose band crosses it is not
        # distinguishable from no relationship at all in that window.
        fig.add_hline(y=0, line=dict(color='rgba(120,120,120,0.6)', width=1,
                                     dash='dot'), row=index + 1, col=1)

    fig.update_layout(
        title=(title if title is not None
               else f'Rolling regression of {quick_text(y)} '
                    f'({window}-point window)'),
        template='plotly_white', width=width,
        height=panel_height * len(predictors) + 120, showlegend=False,
    )
    return fig, summary


def _fit_window(block, y, predictors, intercept, min_points):
    """Ordinary least squares on one window, or None if it cannot be fitted."""
    frame = block[[y, *predictors]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(frame)
    if n < min_points:
        return None

    response = frame[y].to_numpy(dtype=float)
    columns = [frame[p].to_numpy(dtype=float) for p in predictors]
    design = np.column_stack(([np.ones(n)] if intercept else []) + columns)
    if design.shape[0] <= design.shape[1]:
        return None
    # A predictor that never moves within the window carries no information,
    # and makes the normal equations singular rather than merely imprecise.
    if any(np.ptp(column) == 0 for column in columns):
        return None

    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coefficients
    dof = n - design.shape[1]
    if dof <= 0:
        return None
    variance = float(residual @ residual) / dof
    covariance = variance * np.linalg.pinv(design.T @ design)
    errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    names = (['intercept'] if intercept else []) + predictors
    result = {'n': n}
    for name, value, error in zip(names, coefficients, errors):
        result[name] = float(value)
        result[f'{name}_se'] = float(error)

    total = float(((response - response.mean()) ** 2).sum())
    result['r_squared'] = (1.0 - float(residual @ residual) / total
                           if total > 0 else np.nan)
    return result


def _translucent(colour, alpha=0.18):
    if isinstance(colour, str) and colour.startswith('#') and len(colour) == 7:
        r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
        return f'rgba({r},{g},{b},{alpha})'
    return 'rgba(120,120,120,0.18)'
