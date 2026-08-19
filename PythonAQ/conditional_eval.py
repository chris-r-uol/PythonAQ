"""Conditioned model evaluation. Port of openair's ``conditionalEval``.

`conditional_quantile` shows *that* a model is wrong and where in its range.
This asks *why*, by splitting the same bins by other variables: if the bias
grows with wind speed, or only appears at low temperature, the failure has a
name and something can be done about it. Aggregate statistics cannot
distinguish a model that is uniformly mediocre from one that is excellent
except under a condition that matters.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .text import quick_text

__all__ = ['conditional_eval']

_STATISTICS = ('MB', 'NMB', 'MGE', 'RMSE', 'r', 'IOA')


def conditional_eval(df, obs='obs', mod='mod', variables=None, bins=10,
                     min_count=10, statistic='MB', date_col='date_time',
                     title=None, width=1000, panel_height=260):
    """Break model error down by bins of the modelled value, and by other variables.

    Parameters:
    - df (pd.DataFrame): Data containing the observed, modelled and
      conditioning columns.
    - obs (str), mod (str): Observed and modelled value columns.
    - variables (list of str or None): Other columns to summarise within each
      bin, e.g. ['ws', 'wd', 'temp']. None uses none, leaving only the error
      panel.
    - bins (int): Number of quantile bins of the modelled value. Quantile bins
      rather than equal-width ones, so each carries a similar number of points
      instead of the top bins holding three.
    - min_count (int): Bins with fewer complete pairs are dropped.
    - statistic (str): Error statistic for the first panel; one of
      'MB', 'NMB', 'MGE', 'RMSE', 'r', 'IOA'.
    - date_col (str): Ignored except to be excluded from `variables`.
    - title (str or None): Plot title.
    - width (int), panel_height (int): Figure width and per-panel height.

    Returns:
    - fig (go.Figure): The error statistic against modelled value, then one
      panel per conditioning variable.
    - summary (pd.DataFrame): Per bin: the bin centre, n, every statistic in
      `_STATISTICS`, and the mean of each conditioning variable.

    Notes:
    - The conditioning panels show each variable's mean within the bin *minus
      its overall mean*, so that variables on different scales share an axis
      and the reference is always zero. A panel that stays near zero is not
      implicated; one that trends with the error is.
    - This is association, not attribution. A variable that tracks the bias
      may be the cause, or may merely accompany it - wind speed and boundary
      layer depth move together, and this cannot tell them apart.
    - Wind direction is averaged as a plain mean here, which is wrong across
      north. Read the wind direction panel only as a rough indication, or
      condition on a sector column instead.
    """
    for column in (obs, mod):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
    if statistic not in _STATISTICS:
        raise ValueError(f'statistic must be one of {list(_STATISTICS)}.')
    if bins < 2:
        raise ValueError('bins must be at least 2.')

    variables = list(variables or [])
    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ValueError(f'Conditioning column(s) not found: {missing}')

    data = df[[obs, mod, *variables]].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[obs, mod]).copy()
    if len(data) < min_count:
        raise ValueError(
            f'Only {len(data)} complete {obs}/{mod} pairs; need at least '
            f'{min_count}.')

    # Quantile bins on the modelled value. duplicates='drop' because a heavily
    # tied series (many zeros, say) has fewer distinct edges than bins asked
    # for, and the alternative is an exception the caller cannot act on.
    data['_bin'] = pd.qcut(data[mod], bins, duplicates='drop')
    if data['_bin'].cat.categories.empty:
        raise ValueError('The modelled values are too heavily tied to bin.')

    rows = []
    for interval, block in data.groupby('_bin', observed=True):
        if len(block) < min_count:
            continue
        row = {'bin_centre': float(interval.mid), 'bin': str(interval),
               'n': len(block)}
        row.update(_statistics(block[obs].to_numpy(float),
                               block[mod].to_numpy(float)))
        for name in variables:
            row[name] = float(block[name].mean())
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError(
            f'No bin had at least {min_count} pairs. Lower bins or min_count.')

    panels = 1 + len(variables)
    fig = make_subplots(
        rows=panels, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=([f'{statistic} of {quick_text(mod)}']
                        + [quick_text(v) for v in variables]),
    )
    fig.add_trace(go.Scatter(
        x=summary['bin_centre'], y=summary[statistic], mode='lines+markers',
        line=dict(color='#d62728', width=2), marker=dict(size=7),
        customdata=summary['n'],
        hovertemplate=('modelled %{x:.3g}<br>' + statistic
                       + ' %{y:.3g}<br>n %{customdata}<extra></extra>'),
        showlegend=False,
    ), row=1, col=1)
    # Perfect on this statistic is zero for the error measures and one for the
    # agreement measures; draw whichever applies so the panel has a reference.
    fig.add_hline(y=1.0 if statistic in ('r', 'IOA') else 0.0,
                  line=dict(color='rgba(120,120,120,0.6)', width=1, dash='dot'),
                  row=1, col=1)

    palette = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#17becf']
    for index, name in enumerate(variables):
        overall = float(data[name].mean())
        fig.add_trace(go.Scatter(
            x=summary['bin_centre'], y=summary[name] - overall,
            mode='lines+markers', line=dict(color=palette[index % len(palette)],
                                            width=2),
            marker=dict(size=6), showlegend=False,
            customdata=summary[name],
            hovertemplate=('modelled %{x:.3g}<br>mean %{customdata:.3g}'
                           '<br>anomaly %{y:+.3g}<extra></extra>'),
        ), row=index + 2, col=1)
        fig.add_hline(y=0.0, line=dict(color='rgba(120,120,120,0.6)', width=1,
                                       dash='dot'), row=index + 2, col=1)

    fig.update_xaxes(title_text=f'modelled {quick_text(mod)}', row=panels, col=1)
    fig.update_layout(
        title=(title if title is not None
               else f'Conditioned evaluation of {quick_text(mod)} '
                    f'against {quick_text(obs)}'),
        template='plotly_white', width=width,
        height=panel_height * panels + 120, showlegend=False,
    )
    return fig, summary


def _statistics(observed, modelled):
    """The evaluation statistics for one bin.

    Deliberately duplicates the formulas rather than calling `mod_stats`, which
    works on a DataFrame and would mean rebuilding one per bin.
    """
    residual = modelled - observed
    absolute = np.abs(residual)
    total = float(observed.sum())
    result = {
        'MB': float(residual.mean()),
        'MGE': float(absolute.mean()),
        'NMB': float(residual.sum() / total) if total else np.nan,
        'RMSE': float(np.sqrt(np.mean(residual ** 2))),
    }
    if len(observed) > 2 and observed.std() > 0 and modelled.std() > 0:
        result['r'] = float(np.corrcoef(observed, modelled)[0, 1])
    else:
        result['r'] = np.nan

    # Willmott's index of agreement, piecewise as in openair: the denominator
    # switches at twice the observed mean absolute deviation, which keeps the
    # index bounded when the model is very poor.
    deviation = float(np.abs(observed - observed.mean()).sum())
    error = float(absolute.sum())
    if deviation == 0:
        result['IOA'] = np.nan
    elif error <= 2 * deviation:
        result['IOA'] = 1.0 - error / (2 * deviation)
    else:
        result['IOA'] = 2 * deviation / error - 1.0
    return result
