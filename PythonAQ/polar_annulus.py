"""Polar annulus plots. Port of openair's ``polarAnnulus``.

Wind direction runs around the annulus and a temporal variable runs outwards
through it, so a source that only shows up at certain times of day, or in
certain months, appears as an arc rather than being averaged away.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .text import quick_text

__all__ = ['polar_annulus']

# period -> (column builder, number of levels, tick labels, radially cyclic)
_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
_DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def _temporal_index(dates, period):
    """Map timestamps onto an integer level, plus labels for the radial axis.

    Returns (values, n_levels, tick_labels, cyclic). `cyclic` says whether the
    first and last levels are neighbours, which decides how the smoother
    handles the radial edges: hour 23 is adjacent to hour 0, but the last year
    of a record is not adjacent to the first.
    """
    if period == 'hour':
        return dates.dt.hour, 24, [f'{h:02d}' for h in range(24)], True
    if period == 'month':
        return dates.dt.month - 1, 12, _MONTHS, True
    if period == 'weekday':
        return dates.dt.dayofweek, 7, _DAYS, True
    if period == 'season':
        # Meteorological seasons, starting at winter so the ring reads DJF
        # outwards through the year.
        return (dates.dt.month % 12) // 3, 4, ['DJF', 'MAM', 'JJA', 'SON'], True
    if period in ('trend', 'year'):
        years = dates.dt.year
        levels = sorted(years.unique())
        lookup = {year: i for i, year in enumerate(levels)}
        return years.map(lookup), len(levels), [str(y) for y in levels], False
    raise ValueError(
        f"Unknown period '{period}'. Choose from 'hour', 'month', 'weekday', "
        "'season' or 'trend'."
    )


def _smooth_cyclic(grid, sigma, radially_cyclic):
    """Smooth the (level, direction) grid, wrapping where the axis is cyclic.

    Wind direction always wraps: 359 degrees is next to 1 degree, and smoothing
    without that would leave a seam at north. The radial axis wraps only for
    cyclic periods.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        return grid

    filled = np.where(np.isfinite(grid), grid, 0.0)
    weights = np.isfinite(grid).astype(float)

    radial_mode = 'wrap' if radially_cyclic else 'nearest'
    # Normalised convolution: smooth values and weights alike, then divide, so
    # that empty cells neither contribute nor drag their neighbours towards zero.
    numerator = gaussian_filter(filled, sigma=sigma, mode=[radial_mode, 'wrap'])
    denominator = gaussian_filter(weights, sigma=sigma, mode=[radial_mode, 'wrap'])
    with np.errstate(invalid='ignore', divide='ignore'):
        smoothed = numerator / denominator
    return np.where(denominator > 1e-6, smoothed, np.nan)


def polar_annulus(df, pollutant, period='hour', wd_col='wd',
                  date_col='date_time', wd_bins=72, resolution=400,
                  smooth=True, sigma=(1.0, 2.0), inner_radius=0.3,
                  min_count=1, colorscale='Spectral_r', vmin=None, vmax=None,
                  title=None, fig_width=760, fig_height=740):
    """Plot concentration by wind direction and a temporal period.

    Parameters:
    - df (pd.DataFrame): Data containing wind direction, a date and the pollutant.
    - pollutant (str): Concentration column.
    - period (str): Radial variable: 'hour' (default), 'month', 'weekday',
      'season' or 'trend' (one ring per year).
    - wd_col (str), date_col (str): Wind direction and datetime columns.
    - wd_bins (int): Wind direction sectors.
    - resolution (int): Pixels per axis of the rendered surface.
    - smooth (bool): Smooth the binned means, wrapping at the cyclic edges.
    - sigma (tuple): Gaussian widths as (radial, angular), in bins.
    - inner_radius (float): Size of the hole, as a fraction of the outer radius.
      A hole keeps the innermost ring from being squeezed to nothing.
    - min_count (int): Minimum observations for a bin to be used.
    - colorscale (str): Plotly colour scale.
    - vmin, vmax (float or None): Colour limits.
    - title (str or None): Plot title.
    - fig_width, fig_height (int): Figure size in pixels.

    Returns:
    - fig (go.Figure): The polar annulus.
    - summary (pd.DataFrame): Mean and count per (level, direction) bin.
    """
    for column in (pollutant, wd_col, date_col):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
    if not 0 <= inner_radius < 1:
        raise ValueError('inner_radius must be at least 0 and less than 1.')
    if wd_bins < 4:
        raise ValueError('wd_bins must be at least 4.')

    data = df[[date_col, wd_col, pollutant]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.dropna()
    data = data[(data[wd_col] >= 0) & (data[wd_col] <= 360)]
    if data.empty:
        raise ValueError('No complete date / wind direction / pollutant rows.')

    level, n_levels, tick_labels, radially_cyclic = _temporal_index(
        data[date_col], period
    )
    data['_level'] = np.asarray(level)

    bin_size = 360.0 / wd_bins
    # Offset by half a sector so the first bin is centred on north.
    data['_sector'] = (
        np.floor(((data[wd_col] + bin_size / 2.0) % 360) / bin_size)
        .astype(int) % wd_bins
    )

    grouped = data.groupby(['_level', '_sector'], observed=True)[pollutant]
    summary = grouped.agg(['mean', 'count']).reset_index()
    summary = summary.rename(columns={'_level': 'level', '_sector': 'sector'})
    summary['wd'] = summary['sector'] * bin_size
    summary['period'] = [tick_labels[i] for i in summary['level']]

    usable = summary[summary['count'] >= min_count]
    if usable.empty:
        raise ValueError(f'No bin has at least {min_count} observations.')

    grid = np.full((n_levels, wd_bins), np.nan)
    grid[usable['level'].to_numpy(), usable['sector'].to_numpy()] = \
        usable['mean'].to_numpy()
    if smooth:
        grid = _smooth_cyclic(grid, sigma, radially_cyclic)

    axis = np.linspace(-1.0, 1.0, resolution)
    grid_x, grid_y = np.meshgrid(axis, axis)
    radius = np.hypot(grid_x, grid_y)
    # Compass bearing: 0 at north, increasing clockwise.
    bearing = (np.degrees(np.arctan2(grid_x, grid_y))) % 360

    # Radius maps onto the temporal level, leaving the centre empty.
    span = 1.0 - inner_radius
    fraction = (radius - inner_radius) / span
    level_index = np.floor(fraction * n_levels).astype(int)
    sector_index = (
        np.floor(((bearing + bin_size / 2.0) % 360) / bin_size).astype(int) % wd_bins
    )

    inside = (radius >= inner_radius) & (radius <= 1.0)
    level_index = np.clip(level_index, 0, n_levels - 1)
    surface = np.where(inside, grid[level_index, sector_index], np.nan)

    if vmin is None:
        vmin = float(np.nanmin(surface)) if np.isfinite(surface).any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(surface)) if np.isfinite(surface).any() else 1.0

    label = quick_text(pollutant)
    fig = go.Figure(go.Heatmap(
        x=axis, y=axis, z=surface, colorscale=colorscale, zmin=vmin, zmax=vmax,
        zsmooth='best', connectgaps=False,
        colorbar=dict(title=label, thickness=20, len=0.75, y=0.5),
        hoverinfo='skip',
    ))

    _add_annulus_axes(fig, n_levels, tick_labels, inner_radius, period)

    fig.update_layout(
        title=title or f'{label} by wind direction and {period}',
        template='plotly_white', width=fig_width, height=fig_height,
        xaxis=dict(visible=False, scaleanchor='y', scaleratio=1,
                   range=[-1.25, 1.25], constrain='domain'),
        yaxis=dict(visible=False, range=[-1.25, 1.25], constrain='domain'),
        plot_bgcolor='white',
    )
    return fig, summary


def _add_annulus_axes(fig, n_levels, tick_labels, inner_radius, period):
    """Overlay the compass, the annulus edges and radial period labels."""
    for radius in (inner_radius, 1.0):
        fig.add_shape(type='circle', xref='x', yref='y',
                      x0=-radius, y0=-radius, x1=radius, y1=radius,
                      line=dict(color='rgba(110,110,110,0.55)', width=1),
                      layer='above')

    for bearing, label in zip([0, 90, 180, 270], ['N', 'E', 'S', 'W']):
        angle = np.deg2rad((-bearing + 90) % 360)
        fig.add_annotation(
            x=1.12 * np.cos(angle), y=1.12 * np.sin(angle),
            text=f'<b>{label}</b>', showarrow=False,
            font=dict(size=14, color='#333'), xref='x', yref='y',
        )

    # Label a few rings rather than all of them, which would be unreadable for
    # 24 hours, and place them along the north axis where the plot is emptiest.
    step = max(1, n_levels // 6)
    span = 1.0 - inner_radius
    for index in range(0, n_levels, step):
        radius = inner_radius + (index + 0.5) / n_levels * span
        fig.add_annotation(
            x=0.0, y=radius, text=tick_labels[index], showarrow=False,
            font=dict(size=9, color='rgba(50,50,50,0.9)'),
            bgcolor='rgba(255,255,255,0.7)', xref='x', yref='y',
        )

    fig.add_annotation(
        x=0.0, y=-1.19, text=f'radius: {period}', showarrow=False,
        font=dict(size=11, color='#555'), xref='x', yref='y',
    )
