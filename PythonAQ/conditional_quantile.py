"""Conditional quantile plots. Port of openair's ``conditionalQuantile``.

``mod_stats`` says how well a model does overall; this says *where* it fails.
Modelled values are binned, and within each bin the spread of the corresponding
observations is drawn. A model can have a respectable correlation while being
badly wrong at high concentrations, which is exactly the range that matters,
and that shows up here as the median line peeling away from the 1:1 line at the
top of the range.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .faceting import conditionable
from .text import quick_text

__all__ = ['conditional_quantile']


@conditionable
def conditional_quantile(df, obs='obs', mod='mod', bins=20, min_count=10,
                         quantiles=(0.1, 0.25, 0.75, 0.9), show_histograms=True,
                         colour='#1f77b4', title=None, xlab=None, ylab=None,
                         width=820, height=700):
    """Plot the distribution of observations conditioned on modelled values.

    Parameters:
    - df (pd.DataFrame): Data containing both columns.
    - obs (str), mod (str): Observed and modelled value columns.
    - bins (int or sequence): Number of equal-width bins across the modelled
      range, or explicit edges.
    - min_count (int): Bins with fewer pairs than this are dropped, since a
      quantile of three points is not informative.
    - quantiles (sequence): Two pairs, drawn as nested bands around the median.
      Defaults to the 10/90 and 25/75 intervals openair uses.
    - show_histograms (bool): Overlay the marginal distributions of both
      variables, which show where the data actually is.
    - colour (str): Base colour for the bands.
    - title, xlab, ylab (str or None): Labels; generated when None.
    - width, height (int): Figure size in pixels.

    Returns:
    - fig (go.Figure): The conditional quantile plot.
    - summary (pd.DataFrame): Median and the requested quantiles per bin.
    """
    for column in (obs, mod):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
    if len(quantiles) != 4:
        raise ValueError('quantiles must give four values: two nested pairs.')

    pair = df[[mod, obs]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < min_count:
        raise ValueError(
            f'Only {len(pair)} complete pairs; need at least min_count={min_count}.'
        )

    edges = (np.linspace(pair[mod].min(), pair[mod].max(), int(bins) + 1)
             if np.isscalar(bins) else np.asarray(bins, dtype=float))
    assigned = pd.cut(pair[mod], bins=edges, include_lowest=True, labels=False)

    low_outer, low_inner, high_inner, high_outer = sorted(quantiles)
    rows = []
    for index in range(len(edges) - 1):
        values = pair.loc[assigned == index, obs]
        if len(values) < min_count:
            continue
        rows.append({
            'mod': (edges[index] + edges[index + 1]) / 2.0,
            'n': len(values),
            'median': values.median(),
            'lower_outer': values.quantile(low_outer),
            'lower_inner': values.quantile(low_inner),
            'upper_inner': values.quantile(high_inner),
            'upper_outer': values.quantile(high_outer),
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError(
            f'No bin has at least {min_count} pairs. Use fewer bins or lower '
            f'min_count.'
        )

    fig = go.Figure()
    x = summary['mod'].tolist()

    for lower, upper, opacity, name in (
        ('lower_outer', 'upper_outer', 0.18,
         f'{low_outer:.0%}-{high_outer:.0%}'),
        ('lower_inner', 'upper_inner', 0.35,
         f'{low_inner:.0%}-{high_inner:.0%}'),
    ):
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=summary[upper].tolist() + summary[lower].tolist()[::-1],
            fill='toself', fillcolor=_with_alpha(colour, opacity),
            line=dict(width=0), name=name, hoverinfo='skip',
        ))

    fig.add_trace(go.Scatter(
        x=x, y=summary['median'], mode='lines', name='median',
        line=dict(color=colour, width=2.5),
        hovertemplate='modelled %{x:.1f}<br>observed median %{y:.1f}<extra></extra>',
    ))

    # The 1:1 line is the reference: a perfect model puts the median on it.
    limits = [float(min(summary['mod'].min(), summary['lower_outer'].min())),
              float(max(summary['mod'].max(), summary['upper_outer'].max()))]
    fig.add_trace(go.Scatter(
        x=limits, y=limits, mode='lines', name='1:1',
        line=dict(color='black', width=1, dash='dash'),
    ))

    if show_histograms:
        _add_marginals(fig, pair[mod], pair[obs], edges, limits)

    fig.update_layout(
        title=title or f'Conditional quantiles: {quick_text(obs)} given '
                       f'{quick_text(mod)}',
        xaxis_title=xlab or f'modelled {quick_text(mod)}',
        yaxis_title=ylab or f'observed {quick_text(obs)}',
        template='plotly_white', width=width, height=height,
        xaxis=dict(range=limits), yaxis=dict(range=limits),
        legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top',
                    bgcolor='rgba(255,255,255,0.7)'),
    )
    return fig, summary


def _with_alpha(colour, alpha):
    """Turn a hex or rgb colour into an rgba string."""
    if colour.startswith('#'):
        hex_value = colour.lstrip('#')
        r, g, b = (int(hex_value[i:i + 2], 16) for i in (0, 2, 4))
    elif colour.startswith('rgb'):
        r, g, b = (int(float(v)) for v in
                   colour[colour.index('(') + 1:colour.index(')')].split(',')[:3])
    else:  # named colour; fall back to a neutral grey rather than failing
        r, g, b = 100, 100, 100
    return f'rgba({r},{g},{b},{alpha})'


def _add_marginals(fig, modelled, observed, edges, limits):
    """Draw both marginal distributions along the bottom of the plot.

    Quantile bands say nothing about how many points support them, so the
    histograms show where the data actually is. They are scaled into the lower
    fifth of the panel rather than given their own axis, to keep the 1:1 line
    and the bands on one comparable scale.
    """
    span = limits[1] - limits[0]
    floor = limits[0]

    for values, name, colour in ((modelled, 'modelled', 'rgba(60,60,60,0.45)'),
                                 (observed, 'observed', 'rgba(200,80,40,0.45)')):
        counts, bin_edges = np.histogram(values, bins=edges)
        if counts.max() == 0:
            continue
        heights = counts / counts.max() * span * 0.18
        centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        fig.add_trace(go.Bar(
            x=centres, y=heights, base=floor, name=f'{name} (n)',
            marker=dict(color=colour), width=np.diff(bin_edges) * 0.45,
            offset=0 if name == 'modelled' else np.diff(bin_edges).mean() * 0.45,
            hovertemplate=f'{name}: %{{customdata}} points<extra></extra>',
            customdata=counts, showlegend=True,
        ))
    fig.update_layout(barmode='overlay')
