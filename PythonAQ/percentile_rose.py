"""Percentile roses. Port of openair's ``percentileRose``.

Shows how the distribution of a pollutant, rather than just its mean, varies
with wind direction.
"""

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go

__all__ = ['percentile_rose']


def percentile_rose(df, pollutant, wd_col='wd', percentile=(25, 50, 75, 90, 95),
                    direction_bins=36, smooth=True, fill=True,
                    colours='Blues', title=None, mean_line=True,
                    fig_width=800, fig_height=800):
    """Plot pollutant percentiles as a function of wind direction.

    Parameters:
    - df (pd.DataFrame): Data containing wind direction and the pollutant.
    - pollutant (str): Column to summarise.
    - wd_col (str): Wind direction column, in degrees.
    - percentile (sequence of float): Percentiles to draw, ascending.
    - direction_bins (int): Number of wind direction sectors.
    - smooth (bool): Wrap the series so the trace closes at north.
    - fill (bool): Fill between successive percentile rings.
    - colours (str): Named Plotly colour scale.
    - title (str or None): Plot title; generated from `pollutant` if None.
    - mean_line (bool): Overlay the directional mean as a dashed line.
    - fig_width, fig_height (int): Figure size in pixels.

    Returns:
    - fig (go.Figure): The percentile rose.
    - summary (pd.DataFrame): Percentile value per direction bin.
    """
    for column in (pollutant, wd_col):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

    percentiles = sorted(
        [percentile] if np.isscalar(percentile) else list(percentile)
    )
    if not percentiles or not all(0 <= p <= 100 for p in percentiles):
        raise ValueError('percentile values must lie between 0 and 100.')
    if direction_bins < 4:
        raise ValueError('direction_bins must be at least 4.')

    data = df[[wd_col, pollutant]].dropna()
    data = data[(data[wd_col] >= 0) & (data[wd_col] <= 360)]
    if data.empty:
        raise ValueError('No valid wind direction / pollutant pairs.')

    bin_size = 360.0 / direction_bins
    # Shift by half a sector so the first bin is centred on north.
    centres = np.arange(direction_bins) * bin_size
    shifted = (data[wd_col] + bin_size / 2.0) % 360
    data = data.assign(_bin=np.floor(shifted / bin_size).astype(int) % direction_bins)

    grouped = data.groupby('_bin')[pollutant]
    rows = []
    for index, centre in enumerate(centres):
        values = grouped.get_group(index) if index in grouped.groups else pd.Series(dtype=float)
        row = {'wd': centre, 'n': len(values),
               'mean': values.mean() if len(values) else np.nan}
        for p in percentiles:
            row[f'percentile.{p:g}'] = np.percentile(values, p) if len(values) else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)

    scale = pcolors.sample_colorscale(
        colours, np.linspace(0.35, 0.95, len(percentiles))
    )

    def _closed(values):
        """Repeat the first point so the polar trace closes."""
        return list(values) + [values[0]] if smooth else list(values)

    theta = _closed(list(summary['wd']))

    fig = go.Figure()
    # Draw largest percentile first so smaller ones layer on top.
    for i, p in reversed(list(enumerate(percentiles))):
        column = f'percentile.{p:g}'
        fig.add_trace(go.Scatterpolar(
            r=_closed(list(summary[column])), theta=theta,
            mode='lines', name=f'{p:g}th percentile',
            line=dict(color=scale[i], width=2),
            fill='toself' if fill else None,
            fillcolor=scale[i] if fill else None,
            hovertemplate='%{theta:.0f}deg: %{r:.1f}<extra>' + f'{p:g}th' + '</extra>',
        ))

    if mean_line:
        fig.add_trace(go.Scatterpolar(
            r=_closed(list(summary['mean'])), theta=theta,
            mode='lines', name='mean',
            line=dict(color='firebrick', width=2, dash='dash'),
        ))

    fig.update_layout(
        title=title or f'Percentile rose: {pollutant}',
        template='plotly_white', width=fig_width, height=fig_height,
        polar=dict(
            angularaxis=dict(direction='clockwise', rotation=90,
                             tickmode='array', tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                             ticktext=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
            radialaxis=dict(title=pollutant, angle=45),
        ),
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top'),
    )
    return fig, summary
