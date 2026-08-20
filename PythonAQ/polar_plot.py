"""Bivariate polar plots. Port of openair's ``polarPlot``.

Concentrations are smoothed over wind speed and direction with a tensor-product
GAM, then rendered as a continuous surface.

The surface is predicted onto a regular Cartesian grid in (u, v) wind-component
space and drawn as a single raster, which is what openair does via lattice's
levelplot. Drawing one flat-filled polygon per bin instead makes the result look
blocky no matter how smooth the underlying fit is, because there is no
interpolation between neighbouring cells; that approach is still available as
``render='tile'``.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pygam import LinearGAM, te
from scipy.spatial import cKDTree

from .text import quick_text
from .faceting import conditionable

__all__ = ['polar_plot']


def _to_components(ws, wd):
    """Convert wind speed and direction to (u, v) plotting components.

    The x axis is u and the y axis is v, so a wind from bearing `wd` is drawn
    at that compass bearing with north at the top and angles increasing
    clockwise.
    """
    radians = np.deg2rad(wd)
    return ws * np.sin(radians), ws * np.cos(radians)


def _prepare_polar_data(df, ws_col, wd_col, conc_col, ws_limit):
    """Clean the inputs and resolve the radial extent.

    Split out from `polar_plot` so that `polar_diff` can resolve one extent
    across both of its datasets: two surfaces drawn to different radii cannot
    meaningfully be subtracted.
    """
    data = df[[ws_col, wd_col, conc_col]].dropna().copy()
    data = data[data[ws_col] >= 0]
    if data.empty:
        raise ValueError('No complete wind speed / direction / concentration rows.')
    data[wd_col] = data[wd_col] % 360

    if ws_limit == 'auto':
        ws_max = float(data[ws_col].quantile(0.99))
    elif ws_limit == 'max':
        ws_max = float(data[ws_col].max())
    else:
        ws_max = float(ws_limit)
    if ws_max <= 0:
        raise ValueError('ws_limit must resolve to a positive wind speed.')
    return data, ws_max


def _polar_surface(data, ws_col, wd_col, conc_col, ws_max, ws_bins, wd_bins,
                   min_count, n_splines, render, resolution, uncertainty,
                   upper_limit, exclude_missing, exclude_distance):
    """Fit the smooth and predict it onto a grid.

    Returns (grid_u, grid_v, Z, ws_bins_array, wd_bins_array). Z carries NaN
    wherever the fit is not supported by data.
    """
    ws_bins_array = np.linspace(float(data[ws_col].min()), ws_max, ws_bins + 1)
    wd_bins_array = np.linspace(0, 360, wd_bins + 1)
    data['ws_bin'] = pd.cut(data[ws_col], bins=ws_bins_array, labels=False,
                            include_lowest=True)
    data['wd_bin'] = pd.cut(data[wd_col], bins=wd_bins_array, labels=False,
                            include_lowest=True)

    # Aggregate before fitting: the GAM only needs the conditional mean surface,
    # and fitting on bin means is far cheaper than on every observation.
    binned = data.groupby(['ws_bin', 'wd_bin'], observed=True).agg(
        ws_mean=(ws_col, 'mean'),
        wd_mean=(wd_col, 'mean'),
        conc_mean=(conc_col, 'mean'),
        count=(conc_col, 'count'),
    ).reset_index()
    binned = binned[binned['count'] >= min_count]
    if len(binned) < 10:
        raise ValueError(
            f'Only {len(binned)} bins have at least {min_count} observations; '
            f'too few to fit a surface. Lower min_count or widen the bins.'
        )

    u_obs, v_obs = _to_components(binned['ws_mean'], binned['wd_mean'])
    X = np.column_stack([u_obs, v_obs])
    y = binned['conc_mean'].to_numpy()

    gam = LinearGAM(te(0, 1, n_splines=(n_splines, n_splines)))
    gam.gridsearch(X, y, progress=False)

    if render == 'tile':
        grid_u, grid_v, Z = _predict_polar_grid(
            gam, ws_bins_array, wd_bins_array,
        )
    else:
        grid_u, grid_v, Z = _predict_cartesian_grid(gam, ws_max, resolution)

    # Blank regions the fit cannot support.
    if uncertainty is not None:
        points = np.column_stack([grid_u.ravel(), grid_v.ravel()])
        interval = gam.prediction_intervals(points, width=uncertainty)
        half_width = ((interval[:, 1] - interval[:, 0]) / 2).reshape(Z.shape)
        Z = np.where(half_width > np.abs(Z), np.nan, Z)
    if upper_limit is not None:
        Z = np.where(Z > upper_limit, np.nan, Z)

    if exclude_missing:
        if exclude_distance is None:
            exclude_distance = 0.06 * ws_max
        # Blank grid cells with no observation nearby, so that empty sectors are
        # left out rather than filled by extrapolation. Doing this on the fine
        # grid gives a boundary that follows the data, rather than the sawtooth
        # edge produced by dropping whole bins.
        #
        # The query runs against every raw observation, not the bin means: bin
        # means thin out towards the rim, which would carve the surface into
        # detached islands wherever a single bin happened to be empty.
        #
        # The test is on the distance to the `min_count`-th nearest observation,
        # not the first. Proximity to a single stray point is not evidence that
        # the surface is supported there, and using the nearest neighbour alone
        # lets the GAM's edge extrapolation survive in near-empty sectors as
        # spurious hot and cold spots.
        raw_u, raw_v = _to_components(data[ws_col], data[wd_col])
        tree = cKDTree(np.column_stack([raw_u, raw_v]))
        neighbours = max(int(min_count), 1)
        distance, _ = tree.query(
            np.column_stack([grid_u.ravel(), grid_v.ravel()]), k=neighbours,
        )
        if neighbours > 1:
            distance = distance[:, -1]
        keep = distance.reshape(Z.shape) <= exclude_distance
        keep = _tidy_mask(keep, wrap_axis=0 if render == 'tile' else None)
        Z = np.where(keep, Z, np.nan)

    if np.all(np.isnan(Z)):
        raise ValueError(
            'Every grid cell was excluded. Try exclude_missing=False, a larger '
            'exclude_distance, or a less strict uncertainty.'
        )

    return grid_u, grid_v, Z, ws_bins_array, wd_bins_array


@conditionable
def polar_plot(
    df,
    ws_col='ws',
    wd_col='wd',
    conc_col='NO2',
    ws_bins=30,
    wd_bins=72,
    color_palette='Spectral_r',
    title='Polar Plot of Concentration',
    vmin=None,
    vmax=None,
    fig_width=800,
    fig_height=800,
    min_count=3,
    n_splines=10,
    uncertainty=None,
    render='raster',
    resolution=300,
    n_contours=14,
    exclude_missing=True,
    exclude_distance=None,
    upper_limit=None,
    ws_limit='auto',
):
    """Create a bivariate polar plot of concentration by wind speed and direction.

    Parameters:
    - df (pd.DataFrame): DataFrame containing wind data and concentrations.
    - ws_col (str): Column name for wind speed.
    - wd_col (str): Column name for wind direction, in degrees.
    - conc_col (str): Column name for concentration.
    - ws_bins (int): Wind speed bins used when aggregating before the fit.
    - wd_bins (int): Wind direction bins used when aggregating before the fit.
    - color_palette (str or list): Plotly colour scale.
    - title (str): Title of the plot.
    - vmin, vmax (float or None): Colour scale limits.
    - fig_width, fig_height (int): Figure size in pixels.
    - min_count (int): Minimum observations per bin for it to inform the fit.
    - n_splines (int): Splines per dimension of the tensor-product smooth.
    - uncertainty (float or None): Confidence width for the prediction interval
      used to blank untrustworthy regions, e.g. 0.95. None disables the check.
    - render (str): 'raster' for a continuous surface (default), 'contour' for
      filled contour bands, or 'tile' for the original one-polygon-per-bin
      rendering.
    - resolution (int): Grid points per axis for the predicted surface. Higher
      is smoother and slower; only used by 'raster' and 'contour'.
    - n_contours (int): Number of bands when render='contour'.
    - exclude_missing (bool): Blank areas of the grid that sit too far from any
      observation, so the surface is not extrapolated into empty sectors.
    - exclude_distance (float or None): How far from an observation counts as
      too far, in wind speed units. Defaults to 4% of the maximum wind speed.
    - upper_limit (float or None): Blank predictions above this concentration.
      None keeps them all.
    - ws_limit (str or float): Radial extent. 'auto' (default) uses the 99th
      percentile of wind speed, since the rare highest speeds are too sparse to
      smooth and would otherwise leave most of the circle blank; 'max' uses the
      full observed range; a number sets it explicitly.

    Returns:
    - fig (plotly.graph_objects.Figure): The resulting polar plot.
    """
    for col in [ws_col, wd_col, conc_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    if render not in ('raster', 'contour', 'tile'):
        raise ValueError("render must be 'raster', 'contour' or 'tile'.")
    if resolution < 20:
        raise ValueError('resolution must be at least 20.')

    data, ws_max = _prepare_polar_data(df, ws_col, wd_col, conc_col, ws_limit)
    grid_u, grid_v, Z, ws_bins_array, wd_bins_array = _polar_surface(
        data, ws_col, wd_col, conc_col, ws_max, ws_bins, wd_bins, min_count,
        n_splines, render, resolution, uncertainty, upper_limit,
        exclude_missing, exclude_distance,
    )

    if vmin is None:
        vmin = min(0.0, float(np.nanmin(Z)))
    if vmax is None:
        vmax = float(np.nanmax(Z))

    label = quick_text(conc_col)
    fig = go.Figure()
    if render == 'contour':
        fig.add_trace(go.Contour(
            x=grid_u[0, :], y=grid_v[:, 0], z=Z,
            colorscale=color_palette, zmin=vmin, zmax=vmax,
            ncontours=n_contours, contours=dict(coloring='fill', showlines=False),
            connectgaps=False,
            colorbar=dict(title=label, thickness=20, len=0.75, y=0.5),
            hovertemplate=_hover_template(label),
        ))
    elif render == 'raster':
        fig.add_trace(go.Heatmap(
            x=grid_u[0, :], y=grid_v[:, 0], z=Z,
            colorscale=color_palette, zmin=vmin, zmax=vmax,
            zsmooth='best', connectgaps=False,
            colorbar=dict(title=label, thickness=20, len=0.75, y=0.5),
            hovertemplate=_hover_template(label),
        ))
    else:
        _add_tiles(fig, ws_bins_array, wd_bins_array, Z, color_palette, vmin, vmax)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers', showlegend=False,
            marker=dict(colorscale=color_palette, cmin=vmin, cmax=vmax, color=[],
                        showscale=True,
                        colorbar=dict(title=label, thickness=20, len=0.75, y=0.5)),
        ))

    # The axis range must clear the outermost ring and its compass labels,
    # which sit beyond ws_max; sizing the range from ws_max alone clips them.
    outer = _add_polar_axes(fig, ws_max)
    limit = outer * 1.16
    fig.update_layout(
        title=title, width=fig_width, height=fig_height, template='plotly_white',
        xaxis=dict(visible=False, scaleanchor='y', scaleratio=1,
                   range=[-limit, limit], constrain='domain'),
        yaxis=dict(visible=False, range=[-limit, limit], constrain='domain'),
        plot_bgcolor='white',
    )
    return fig


def _tidy_mask(keep, radius=2, wrap_axis=None):
    """Remove speckle from the coverage mask and close pinholes in it.

    Thresholding a distance field cell by cell leaves isolated specks just
    inside the cut-off and single-cell holes just outside, which read as noise
    along the boundary. A morphological opening then closing removes both while
    leaving the overall shape of the covered region alone.

    `wrap_axis` names an axis that is circular, which for a mask indexed by
    wind direction is the direction axis. Without it the morphology treats
    0 and 360 degrees as opposite ends of a rectangle and erodes the join,
    punching a wedge out of the north of every plot. Wind speed is not
    circular, so its axis is deliberately left to erode at the rim.
    """
    try:
        from scipy.ndimage import binary_closing, binary_opening
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        return keep

    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disc = (x ** 2 + y ** 2) <= radius ** 2 + 1e-9
    assert disc.shape == (size, size)

    padded = keep
    if wrap_axis is not None:
        widths = [(0, 0)] * keep.ndim
        widths[wrap_axis] = (radius, radius)
        padded = np.pad(keep, widths, mode='wrap')

    opened = binary_opening(padded, structure=disc)
    # Opening can erase a genuinely small covered region entirely; if so, keep
    # the original rather than silently dropping data.
    if not opened.any():
        return keep
    tidied = binary_closing(opened, structure=disc)

    if wrap_axis is not None:
        trim = [slice(None)] * keep.ndim
        trim[wrap_axis] = slice(radius, -radius)
        tidied = tidied[tuple(trim)]
    return tidied


def _hover_template(conc_col):
    """Hover text. x and y are the wind components, so report both those and
    the speed/bearing they correspond to."""
    return (
        f'{conc_col}: %{{z:.1f}}<br>'
        'u: %{x:.1f}  v: %{y:.1f}'
        '<extra></extra>'
    )


def _predict_cartesian_grid(gam, ws_max, resolution):
    """Predict the surface on a regular Cartesian grid over the wind components.

    A Cartesian grid is what makes the result render as a smooth image: cells
    are uniform squares that Plotly can interpolate between, rather than
    wedges that grow towards the rim.
    """
    axis = np.linspace(-ws_max, ws_max, resolution)
    grid_u, grid_v = np.meshgrid(axis, axis)
    Z = gam.predict(np.column_stack([grid_u.ravel(), grid_v.ravel()]))
    Z = Z.reshape(grid_u.shape)
    # Keep the plot circular: outside the maximum observed wind speed there is
    # nothing to say.
    Z = np.where(np.hypot(grid_u, grid_v) > ws_max, np.nan, Z)
    return grid_u, grid_v, Z


def _predict_polar_grid(gam, ws_bins_array, wd_bins_array):
    """Predict at polar bin centres, for the legacy tile rendering."""
    ws_centers = (ws_bins_array[:-1] + ws_bins_array[1:]) / 2
    wd_centers = (wd_bins_array[:-1] + wd_bins_array[1:]) / 2
    ws_grid, wd_grid = np.meshgrid(ws_centers, wd_centers)
    grid_u, grid_v = _to_components(ws_grid, wd_grid)
    Z = gam.predict(np.column_stack([grid_u.ravel(), grid_v.ravel()]))
    return grid_u, grid_v, Z.reshape(ws_grid.shape)


def _add_tiles(fig, ws_bins_array, wd_bins_array, Z, color_palette, vmin, vmax):
    """Draw one filled polygon per polar bin (the original rendering)."""
    from plotly.colors import get_colorscale, sample_colorscale

    colorscale = (get_colorscale(color_palette)
                  if isinstance(color_palette, str) else color_palette)
    span = (vmax - vmin) or 1.0
    shapes = []
    for i in range(len(ws_bins_array) - 1):
        for j in range(len(wd_bins_array) - 1):
            value = Z[j, i]
            if np.isnan(value):
                continue
            r0, r1 = ws_bins_array[i], ws_bins_array[i + 1]
            t0 = np.deg2rad((-wd_bins_array[j] + 90) % 360)
            t1 = np.deg2rad((-wd_bins_array[j + 1] + 90) % 360)
            corners = [
                (r0 * np.cos(t0), r0 * np.sin(t0)), (r0 * np.cos(t1), r0 * np.sin(t1)),
                (r1 * np.cos(t1), r1 * np.sin(t1)), (r1 * np.cos(t0), r1 * np.sin(t0)),
            ]
            path = 'M ' + ' L '.join(f'{x},{y}' for x, y in corners) + ' Z'
            colour = sample_colorscale(
                colorscale, [float(np.clip((value - vmin) / span, 0, 1))]
            )[0]
            shapes.append(dict(type='path', path=path, fillcolor=colour,
                               line=dict(width=0.5, color=colour),
                               xref='x', yref='y', layer='below'))
    fig.update_layout(shapes=shapes)


def _add_polar_axes(fig, ws_max):
    """Overlay compass labels and radial rings. Returns the outermost radius."""
    step = 5 if ws_max > 10 else 2
    rounded = (int(ws_max // step) + 1) * step

    for radius in np.arange(step, rounded + step, step):
        fig.add_shape(type='circle', xref='x', yref='y',
                      x0=-radius, y0=-radius, x1=radius, y1=radius,
                      line=dict(color='rgba(120,120,120,0.35)', width=1),
                      layer='above')
        fig.add_annotation(x=radius * np.cos(np.deg2rad(45)),
                           y=radius * np.sin(np.deg2rad(45)),
                           text=f'{radius:g}', showarrow=False,
                           font=dict(size=10, color='rgba(60,60,60,0.8)'),
                           bgcolor='rgba(255,255,255,0.6)', xref='x', yref='y')

    for bearing, label in zip([0, 90, 180, 270], ['N', 'E', 'S', 'W']):
        angle = np.deg2rad((-bearing + 90) % 360)
        fig.add_shape(type='line', xref='x', yref='y', x0=0, y0=0,
                      x1=rounded * np.cos(angle), y1=rounded * np.sin(angle),
                      line=dict(color='rgba(120,120,120,0.35)', width=1),
                      layer='above')
        fig.add_annotation(
            x=rounded * 1.08 * np.cos(angle), y=rounded * 1.08 * np.sin(angle),
            text=f'<b>{label}</b>', showarrow=False,
            font=dict(size=14, color='#333'), xref='x', yref='y',
        )
    return rounded
