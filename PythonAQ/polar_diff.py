"""Difference between two bivariate polar surfaces. Port of openair's ``polarDiff``.

Answers "what changed, and from which direction?" - the question behind most
before-and-after work: an intervention, a plant closure, a lockdown, or one
site measured against another. Fitting each period separately and reading the
two plots side by side does not do this, because the eye cannot subtract two
colour scales.
"""

import numpy as np
import plotly.graph_objects as go

from .polar_plot import (_add_polar_axes, _polar_surface, _prepare_polar_data,
                         _to_components)
from .text import quick_text

__all__ = ['polar_diff']


def polar_diff(
    before,
    after,
    ws_col='ws',
    wd_col='wd',
    conc_col='NO2',
    ws_bins=30,
    wd_bins=72,
    color_palette='RdBu_r',
    title=None,
    limit=None,
    fig_width=800,
    fig_height=800,
    min_count=3,
    n_splines=10,
    render='raster',
    resolution=300,
    n_contours=14,
    uncertainty=None,
    exclude_missing=True,
    exclude_distance=None,
    ws_limit='auto',
):
    """Plot `after` minus `before` as a polar surface.

    Parameters:
    - before, after (pd.DataFrame): The two periods, or two sites. Each needs
      wind speed, wind direction and concentration columns.
    - ws_col, wd_col, conc_col (str): Column names, shared by both frames.
    - ws_bins, wd_bins (int): Bins used when aggregating before each fit.
    - color_palette (str or list): A *diverging* scale. The default puts blue
      on decreases and red on increases.
    - title (str or None): Plot title. None builds one from `conc_col`.
    - limit (float or None): Colour scale extent. The scale always runs from
      -limit to +limit; None uses the largest absolute difference. Set it
      explicitly to compare several difference plots against each other.
    - fig_width, fig_height (int): Figure size in pixels.
    - min_count (int): Minimum observations per bin for it to inform a fit.
    - n_splines (int): Splines per dimension of each tensor-product smooth.
    - render (str): 'raster', 'contour' or 'tile', as for `polar_plot`.
    - resolution (int): Grid points per axis.
    - n_contours (int): Bands when render='contour'.
    - uncertainty (float or None): Confidence width used to blank untrustworthy
      regions of each surface before subtracting, e.g. 0.95.
    - exclude_missing (bool), exclude_distance (float or None): Coverage
      masking, as for `polar_plot`, applied to each period separately.
    - ws_limit (str or float): Radial extent. Resolved once from the two
      datasets pooled, so both surfaces are drawn to the same radius.

    Returns:
    - fig (go.Figure): The difference plot.

    Notes:
    - The colour scale is forced to be symmetric about zero. A diverging scale
      centred anywhere else reads as though the midpoint colour meant "no
      change" when it does not, which would misstate the sign of the result
      over part of the plot.
    - A cell is drawn only where *both* periods have support. Wind sectors
      sampled in one period and not the other are left blank rather than
      being reported as a large change against nothing.
    - Each period is smoothed independently before subtraction, so a
      difference smaller than the noise in either fit is not meaningful. Use
      `uncertainty` to blank the regions where that is a real risk.
    """
    for name, frame in (('before', before), ('after', after)):
        for col in (ws_col, wd_col, conc_col):
            if col not in frame.columns:
                raise ValueError(f"Column '{col}' not found in the {name} DataFrame.")
    if render not in ('raster', 'contour', 'tile'):
        raise ValueError("render must be 'raster', 'contour' or 'tile'.")
    if limit is not None and limit <= 0:
        raise ValueError('limit must be positive.')

    first, first_max = _prepare_polar_data(before, ws_col, wd_col, conc_col, ws_limit)
    second, second_max = _prepare_polar_data(after, ws_col, wd_col, conc_col, ws_limit)
    # One radius for both. Taking each period's own 99th percentile would draw
    # the two surfaces on different grids, and subtracting those would compare
    # different wind speeds to each other.
    ws_max = max(first_max, second_max)

    surfaces = []
    for data in (first, second):
        grid_u, grid_v, Z, _, _ = _polar_surface(
            data, ws_col, wd_col, conc_col, ws_max, ws_bins, wd_bins, min_count,
            n_splines, render, resolution, uncertainty, None, exclude_missing,
            exclude_distance,
        )
        surfaces.append((grid_u, grid_v, Z))

    (grid_u, grid_v, Z_before), (_, _, Z_after) = surfaces
    # NaN propagates, so this is the intersection of the two coverage masks.
    difference = Z_after - Z_before

    if np.all(np.isnan(difference)):
        raise ValueError(
            'The two periods have no wind sectors in common that both support '
            'a fit. Try exclude_missing=False or a lower min_count.'
        )

    if limit is None:
        # Floored, so that two periods which happen to be identical render as
        # a flat "no change" surface rather than collapsing the colour scale
        # to a single point. Comparing a period against itself is a reasonable
        # thing to do when checking a pipeline.
        limit = max(float(np.nanmax(np.abs(difference))), 1e-9)

    label = quick_text(conc_col)
    heading = title if title is not None else f'Change in {label}'
    bar = dict(title=f'Δ {label}', thickness=20, len=0.75, y=0.5)
    hover = ('wind from %{customdata:.0f}°<br>wind speed %{text:.1f}<br>'
             'change %{z:.2f}<extra></extra>')
    # Hover needs polar coordinates; the grid is Cartesian.
    speed = np.sqrt(grid_u ** 2 + grid_v ** 2)
    direction = (np.degrees(np.arctan2(grid_u, grid_v))) % 360

    fig = go.Figure()
    common = dict(x=grid_u[0, :], y=grid_v[:, 0], z=difference,
                  colorscale=color_palette, zmin=-limit, zmax=limit,
                  connectgaps=False, colorbar=bar, text=speed,
                  customdata=direction, hovertemplate=hover)
    if render == 'contour':
        fig.add_trace(go.Contour(ncontours=n_contours,
                                 contours=dict(coloring='fill', showlines=False),
                                 **common))
    else:
        # 'tile' predicts onto a polar grid, but the difference is still a
        # rectangular array of z values, so a heatmap renders both.
        fig.add_trace(go.Heatmap(zsmooth='best' if render == 'raster' else False,
                                 **common))

    outer = _add_polar_axes(fig, ws_max)
    extent = outer * 1.16
    fig.update_layout(
        title=heading, width=fig_width, height=fig_height,
        template='plotly_white',
        xaxis=dict(visible=False, scaleanchor='y', scaleratio=1,
                   range=[-extent, extent], constrain='domain'),
        yaxis=dict(visible=False, range=[-extent, extent], constrain='domain'),
        plot_bgcolor='white',
    )
    return fig
