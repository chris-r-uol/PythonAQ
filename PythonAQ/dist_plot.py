"""Distribution plots. Port of openair's ``distPlot``.

Concentrations are not normally distributed - they are bounded below at zero,
right-skewed, and often a mixture of a background and a source. A mean hides
all of that. This shows the shape, and stacks several pollutants or several
conditioning levels on one pair of axes so their shapes can be compared.
"""

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go

from .faceting import conditionable
from .text import quick_text

__all__ = ['dist_plot']


@conditionable
def dist_plot(df, pollutant='NO2', kind='density', bins=50, bandwidth=None,
              colours=None, title=None, xaxis_title=None, log_x=False,
              normalise=True, width=900, height=520):
    """Plot the distribution of one or more pollutants.

    Parameters:
    - df (pd.DataFrame): Input data.
    - pollutant (str or list): Column, or several to overlay.
    - kind (str): 'density' for a kernel density estimate, 'histogram' for
      counts, or 'cdf' for the cumulative distribution.
    - bins (int): Histogram bins. Ignored by the other kinds.
    - bandwidth (float or None): Kernel bandwidth for 'density'. None uses
      Silverman's rule.
    - colours (list or None): Series colours.
    - title (str or None): Plot title.
    - xaxis_title (str or None): X axis label; defaults to the pollutant name,
      or 'concentration' when several are shown.
    - log_x (bool): Plot the distribution of the logarithm instead. Often the
      only way to see the bulk and the tail of a concentration at once.
    - normalise (bool): For 'histogram', plot proportions rather than counts,
      so series of different lengths are comparable.
    - width, height (int): Figure size in pixels.

    Returns:
    - fig (go.Figure): The distribution plot.

    Notes:
    - A kernel density estimate spreads weight symmetrically around each
      observation, so near a hard boundary it puts mass where no data can be.
      For a concentration bounded at zero this shows up as a tail below zero.
      'histogram' or log_x avoid the problem rather than hiding it.
    """
    if kind not in ('density', 'histogram', 'cdf'):
        raise ValueError("kind must be 'density', 'histogram' or 'cdf'.")
    pollutants = [pollutant] if isinstance(pollutant, str) else list(pollutant)
    missing = [p for p in pollutants if p not in df.columns]
    if missing:
        raise ValueError(f'Column(s) not found in the DataFrame: {missing}')
    if bins < 2:
        raise ValueError('bins must be at least 2.')

    colours = colours or pcolors.qualitative.Plotly
    fig = go.Figure()
    drawn = 0

    for index, name in enumerate(pollutants):
        values = pd.to_numeric(df[name], errors='coerce').to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if log_x:
            # Zero and negative values have no logarithm. Dropping them is the
            # honest option; substituting a small number would invent a mode.
            values = values[values > 0]
            values = np.log10(values)
        if len(values) < 2:
            continue
        drawn += 1
        colour = colours[index % len(colours)]
        label = quick_text(name)

        if kind == 'histogram':
            counts, edges = np.histogram(values, bins=bins)
            heights = counts / counts.sum() if normalise and counts.sum() else counts
            centres = (edges[:-1] + edges[1:]) / 2.0
            fig.add_trace(go.Bar(
                x=centres, y=heights, name=label, marker_color=colour,
                opacity=0.6 if len(pollutants) > 1 else 0.85,
                hovertemplate=f'{label}<br>%{{x:.3g}}<br>%{{y:.4g}}<extra></extra>',
            ))
        elif kind == 'cdf':
            ordered = np.sort(values)
            fig.add_trace(go.Scatter(
                x=ordered, y=np.arange(1, len(ordered) + 1) / len(ordered),
                mode='lines', name=label, line=dict(color=colour, width=2),
                hovertemplate=f'{label}<br>%{{x:.3g}}<br>%{{y:.3f}}<extra></extra>',
            ))
        else:
            grid, density = _kde(values, bandwidth)
            fig.add_trace(go.Scatter(
                x=grid, y=density, mode='lines', name=label,
                line=dict(color=colour, width=2), fill='tozeroy',
                fillcolor=_translucent(colour),
                hovertemplate=f'{label}<br>%{{x:.3g}}<br>%{{y:.4g}}<extra></extra>',
            ))

    if not drawn:
        raise ValueError('No pollutant had enough finite values to plot.')

    default_x = (quick_text(pollutants[0]) if len(pollutants) == 1
                 else 'concentration')
    y_title = {'density': 'density',
               'histogram': 'proportion' if normalise else 'count',
               'cdf': 'cumulative proportion'}[kind]
    fig.update_layout(
        title=title if title is not None else f'Distribution of {default_x}',
        xaxis_title=(xaxis_title if xaxis_title is not None
                     else (f'log10({default_x})' if log_x else default_x)),
        yaxis_title=y_title, template='plotly_white',
        width=width, height=height, barmode='overlay',
        showlegend=len(pollutants) > 1,
    )
    return fig


def _kde(values, bandwidth=None):
    """Gaussian kernel density estimate on a regular grid."""
    n = len(values)
    spread = values.std(ddof=1)
    if bandwidth is None:
        # Silverman's rule, using the smaller of the standard deviation and a
        # scaled IQR so that one extreme value cannot widen the whole kernel.
        iqr = np.subtract(*np.percentile(values, [75, 25]))
        scale = min(spread, iqr / 1.349) if iqr > 0 else spread
        bandwidth = 0.9 * scale * n ** (-0.2)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1.0

    grid = np.linspace(values.min() - 3 * bandwidth,
                       values.max() + 3 * bandwidth, 512)
    # Evaluated in blocks: the full outer product is len(values) x 512, which
    # is hundreds of megabytes for a few years of hourly data.
    density = np.zeros_like(grid)
    for start in range(0, n, 4096):
        chunk = values[start:start + 4096]
        z = (grid[None, :] - chunk[:, None]) / bandwidth
        density += np.exp(-0.5 * z ** 2).sum(axis=0)
    density /= n * bandwidth * np.sqrt(2 * np.pi)
    return grid, density


def _translucent(colour, alpha=0.18):
    """Fill colour matching a line colour."""
    if isinstance(colour, str) and colour.startswith('#') and len(colour) == 7:
        r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
        return f'rgba({r},{g},{b},{alpha})'
    return 'rgba(120,120,120,0.18)'
