"""Directional analysis plots placed on a map. Port of ``openairmaps``.

A polar plot says which direction a source lies in. A map says what is in that
direction. Neither answers the question alone, and reading one against the
other by eye is exactly the step where mistakes happen - especially with
several sites, where the interesting result is usually that two sites disagree
about where the pollution comes from.

Each site's plot is drawn in geographic coordinates rather than pasted on as a
fixed-size image, so the markers keep their bearings against the streets
underneath at every zoom level. The radial axis is not a distance, though: a
marker's size on the ground is a drawing choice set by `radius_km`, and a
feature reaching the rim says nothing about how far away the source is.

The default basemap is Carto Positron, the muted OpenStreetMap style, with
Carto Dark Matter as the dark counterpart. Both are deliberately low-contrast:
a full-colour basemap competes with the data drawn on top of it.
"""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import get_colorscale, sample_colorscale

from .polar_plot import _polar_surface, _prepare_polar_data
from .text import quick_text

__all__ = ['annulus_map', 'freq_map', 'percentile_rose_map', 'polar_map',
           'pollutant_rose_map', 'wind_rose_map']

_KM_PER_DEGREE = 111.32
# Marker diameter as a fraction of the mapped extent, below which the
# automatic radius is reported as cramped. Measured against real networks:
# three West Yorkshire sites one per town give 0.57 and read comfortably,
# while adding Leeds Headingley 2.7 km from Leeds Centre drops every marker
# to 0.13 and they stop being legible. A fifth sits clear of both rather than
# being fitted to either.
_CRAMPED = 0.20
_LATITUDE_NAMES = ('latitude', 'lat', 'Latitude', 'LATITUDE')
_LONGITUDE_NAMES = ('longitude', 'lon', 'lng', 'long', 'Longitude', 'LONGITUDE')
_SITE_NAMES = ('site_id', 'site', 'site_name', 'code', 'station')


def _resolve_column(df, given, candidates, what):
    """Find a column by name, or by the usual spellings of it."""
    if given is not None:
        if given not in df.columns:
            raise ValueError(f"Column '{given}' not found in the DataFrame.")
        return given
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(
        f'No {what} column found. Looked for {list(candidates)}; pass one '
        f'explicitly.')


def _ground_offset(lat0, lon0, north_km, east_km):
    """Shift a position by a distance north and east, in degrees.

    A local flat-earth approximation. Over the few kilometres a marker spans
    the error is metres, and the alternative - a proper geodesic - would place
    the same polygon to a precision the radial axis does not have, since it is
    a wind speed rather than a distance.
    """
    latitude = lat0 + north_km / _KM_PER_DEGREE
    # Longitude degrees shorten towards the poles. Clamped so that a site at
    # the pole scales by a large factor rather than dividing by zero.
    shrink = max(np.cos(np.deg2rad(lat0)), 1e-6)
    longitude = lon0 + east_km / (_KM_PER_DEGREE * shrink)
    return latitude, longitude


def _sector(lat0, lon0, r0_km, r1_km, wd0, wd1, arc=4):
    """One annular sector as a closed ring of (lat, lon) points.

    `wd0` and `wd1` are wind directions in degrees, which are bearings: zero
    is north and they increase clockwise, so the sector lands on the map in
    the direction the wind came from.
    """
    bearings = np.linspace(wd0, wd1, arc + 1)
    radians = np.deg2rad(bearings)
    north, east = np.cos(radians), np.sin(radians)

    lats, lons = [], []
    for radius, order in ((r0_km, slice(None)), (r1_km, slice(None, None, -1))):
        for n, e in zip(north[order], east[order]):
            lat, lon = _ground_offset(lat0, lon0, radius * n, radius * e)
            lats.append(lat)
            lons.append(lon)
    lats.append(lats[0])
    lons.append(lons[0])
    return lats, lons


def _auto_radius(positions):
    """Choose a marker radius that keeps neighbouring markers apart.

    Overlapping markers are worse than small ones: two overlapping polar plots
    cannot be read at all, whereas a small one can be zoomed into. The radius
    is therefore set by the closest pair of sites, not by the size of the map.

    The cost is that one close pair shrinks every marker on the map, including
    markers nowhere near it. That is inherent to drawing on the ground rather
    than pinning fixed-size icons, so the default is kept and `radius_km`
    named in a warning when it starts to matter, rather than being something
    to discover after squinting at the result.
    """
    if len(positions) < 2:
        return 2.0
    coordinates = np.asarray(positions, dtype=float)
    mean_latitude = float(np.mean(coordinates[:, 0]))
    shrink = max(np.cos(np.deg2rad(mean_latitude)), 1e-6)
    north = coordinates[:, 0] * _KM_PER_DEGREE
    east = coordinates[:, 1] * _KM_PER_DEGREE * shrink

    separations = []
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            separations.append(np.hypot(north[i] - north[j], east[i] - east[j]))
    closest, furthest = min(separations), max(separations)
    if closest <= 0:
        return 2.0
    radius = float(np.clip(0.35 * closest, 0.3, 25.0))

    # A marker spanning a small fraction of the mapped area cannot be read at
    # the zoom that shows every site, however correct it is.
    if furthest > 0 and 2 * radius / furthest < _CRAMPED:
        warnings.warn(
            f'The closest two sites are {closest:.1f} km apart but the map '
            f'spans {furthest:.0f} km, so markers are drawn at {radius:.1f} km '
            'to stop them overlapping and will be small. Pass radius_km= to '
            'draw them larger and accept the overlap, or map a subset of the '
            'sites.',
            UserWarning, stacklevel=3,
        )
    return radius


def _band_colours(palette, n_levels):
    """`n_levels` colours sampled evenly along a colour scale."""
    scale = get_colorscale(palette) if isinstance(palette, str) else palette
    positions = (np.linspace(0, 1, n_levels) if n_levels > 1 else [0.5])
    return sample_colorscale(scale, list(positions))


def _add_bands(fig, cells, vmin, vmax, palette, n_levels, opacity, label):
    """Draw cells grouped into colour bands, one trace per band.

    One trace per *polygon* would be thousands of traces and a browser that
    cannot pan. Grouping by colour band gives a fixed, small number of traces
    however many sites and sectors there are, at the cost of quantising a
    continuous surface - which is what a filled contour plot does anyway.
    """
    colours = _band_colours(palette, n_levels)
    span = (vmax - vmin) or 1.0
    edges = np.linspace(vmin, vmax, n_levels + 1)

    grouped = [([], []) for _ in range(n_levels)]
    for lats, lons, value in cells:
        if not np.isfinite(value):
            continue
        index = int(np.clip((value - vmin) / span * n_levels, 0, n_levels - 1))
        lat_list, lon_list = grouped[index]
        lat_list.extend([*lats, None])
        lon_list.extend([*lons, None])

    for index, (lat_list, lon_list) in enumerate(grouped):
        if not lat_list:
            continue
        fig.add_trace(go.Scattermap(
            lat=lat_list, lon=lon_list, mode='lines', fill='toself',
            fillcolor=colours[index], line=dict(width=0, color=colours[index]),
            opacity=opacity, hoverinfo='skip', showlegend=False,
            name=f'{edges[index]:.3g} - {edges[index + 1]:.3g}',
        ))

    # A Scattermap cannot carry a colourbar on its own, so an empty scatter
    # trace is used to hang one off, exactly as polar_plot does for tiles.
    fig.add_trace(go.Scattermap(
        lat=[None], lon=[None], mode='markers', showlegend=False,
        hoverinfo='skip',
        marker=dict(size=0, color=[], colorscale=palette, cmin=vmin, cmax=vmax,
                    showscale=True,
                    colorbar=dict(title=label, thickness=18, len=0.7, y=0.5)),
    ))


def _site_frames(df, site_col, latitude_col, longitude_col):
    """Split into one frame per site, with its position. Sites without
    coordinates are dropped rather than silently drawn at (0, 0)."""
    frames = []
    for name, block in df.groupby(site_col, observed=True, sort=True):
        lat = pd.to_numeric(block[latitude_col], errors='coerce').dropna()
        lon = pd.to_numeric(block[longitude_col], errors='coerce').dropna()
        if lat.empty or lon.empty:
            continue
        frames.append((str(name), float(lat.iloc[0]), float(lon.iloc[0]), block))
    if not frames:
        raise ValueError(
            'No site had usable coordinates. Check the latitude and longitude '
            'columns, and that they are numeric.')
    return frames


def _finalise(fig, frames, radius_km, map_style, title, zoom, width, height,
              show_sites, site_hovers):
    """Add the site markers and set the map view."""
    if show_sites:
        fig.add_trace(go.Scattermap(
            lat=[f[1] for f in frames], lon=[f[2] for f in frames],
            mode='markers',
            marker=dict(size=7, color='#111111'),
            text=[f[0] for f in frames], customdata=site_hovers,
            hovertemplate='<b>%{text}</b><br>%{customdata}<extra></extra>',
            showlegend=False, name='sites',
        ))

    latitudes = [f[1] for f in frames]
    longitudes = [f[2] for f in frames]
    centre = dict(lat=float(np.mean(latitudes)), lon=float(np.mean(longitudes)))
    if zoom is None:
        # Fit the markers and the sites they surround, padded by a marker
        # radius on each side so the outermost plot is not clipped.
        shrink = max(np.cos(np.deg2rad(centre['lat'])), 1e-6)
        span_lat = (max(latitudes) - min(latitudes)
                    + 2.4 * radius_km / _KM_PER_DEGREE)
        span_lon = (max(longitudes) - min(longitudes)
                    + 2.4 * radius_km / (_KM_PER_DEGREE * shrink))

        # A web map shows a span of longitude across its *width*. Fitting a
        # latitude span therefore needs that converted through the aspect
        # ratio: on a landscape figure there is less vertical room than
        # horizontal, and comparing the two spans directly silently clips the
        # northernmost and southernmost markers.
        aspect = max(width, 1) / max(height, 1)
        needed = max(span_lon, span_lat * aspect / shrink, 1e-6)
        zoom = float(np.clip(np.log2(360.0 / needed), 1.0, 16.0))

    fig.update_layout(
        map=dict(style=map_style, center=centre, zoom=zoom),
        title=title, width=width, height=height,
        margin=dict(l=10, r=10, t=50 if title else 10, b=10),
        showlegend=False,
    )
    return fig


def polar_map(df, pollutant='NO2', ws_col='ws', wd_col='wd', site=None,
              latitude=None, longitude=None, radius_km=None,
              map_style='carto-positron', colour_palette='Spectral_r',
              n_levels=12, limits='fixed', vmin=None, vmax=None,
              ws_bins=20, wd_bins=48, min_count=3, n_splines=10,
              ws_limit='auto', exclude_missing=True, opacity=0.85,
              title=None, zoom=None, width=1000, height=760, show_sites=True):
    """Draw a bivariate polar plot at each site's position on a map.

    Parameters:
    - df (pd.DataFrame): Wind speed, wind direction, concentration, and a
      position for each site. If positions are not in `df`, merge them on from
      `import_aq_meta` first.
    - pollutant (str): Concentration column.
    - ws_col, wd_col (str): Wind speed and wind direction columns.
    - site (str or None): Column identifying the site. Found automatically
      among 'site_id', 'site', 'site_name', 'code', 'station' when None.
    - latitude, longitude (str or None): Position columns, likewise found
      automatically among the usual spellings.
    - radius_km (float or None): How far each plot reaches on the ground.
      None picks 35% of the distance between the two closest sites, so
      markers do not overlap; one close pair therefore shrinks every marker
      on the map, and a warning names this argument when that leaves them
      under a fifth of the width of the mapped area. Setting it explicitly
      accepts any overlap and silences the warning.
    - map_style (str): Basemap. 'carto-positron' (default) and
      'carto-darkmatter' are the muted OpenStreetMap styles, light and dark.
    - colour_palette (str or list): Plotly colour scale.
    - n_levels (int): Colour bands. The surface is quantised into this many,
      as a filled contour plot would be.
    - limits (str): 'fixed' draws every site on one colour scale, so they can
      be compared; 'free' gives each site its own, which shows the shape of a
      weak site but makes the colours mean different things in each marker.
    - vmin, vmax (float or None): Colour scale limits, overriding `limits`.
    - ws_bins, wd_bins (int): Polar grid resolution per marker.
    - min_count (int): Minimum observations per bin for it to inform a fit.
    - n_splines (int): Splines per dimension of each smooth.
    - ws_limit (str or float): Radial extent, resolved once across all sites
      so that a given radius means the same wind speed in every marker.
    - exclude_missing (bool): Blank sectors with no nearby observations.
    - opacity (float): Marker transparency, so the basemap shows through.
    - title (str or None): Plot title.
    - zoom (float or None): Map zoom. None fits the markers.
    - width, height (int): Figure size in pixels.
    - show_sites (bool): Draw a dot at each site position, with a hover label.

    Returns:
    - fig (go.Figure): The map.

    Notes:
    - The radial axis is wind speed, not distance. A marker's extent on the
      ground is a drawing choice; a feature at the rim of a marker says the
      wind was fast, not that the source is `radius_km` away.
    - `limits='fixed'` is the default because the reason to put several sites
      on one map is to compare them, and a per-site scale silently defeats
      that. Switch to 'free' only when one site's range would flatten the
      others to a single colour.
    """
    return _directional_map(
        df, _polar_cells, dict(
            pollutant=pollutant, ws_col=ws_col, wd_col=wd_col, ws_bins=ws_bins,
            wd_bins=wd_bins, min_count=min_count, n_splines=n_splines,
            ws_limit=ws_limit, exclude_missing=exclude_missing),
        site=site, latitude=latitude, longitude=longitude,
        radius_km=radius_km, map_style=map_style,
        colour_palette=colour_palette, n_levels=n_levels, limits=limits,
        vmin=vmin, vmax=vmax, opacity=opacity,
        title=title if title is not None else f'{quick_text(pollutant)} by wind',
        label=quick_text(pollutant), zoom=zoom, width=width, height=height,
        show_sites=show_sites)


def freq_map(df, ws_col='ws', wd_col='wd', site=None, latitude=None,
             longitude=None, radius_km=None, map_style='carto-positron',
             colour_palette='Blues', n_levels=10, limits='fixed', vmin=None,
             vmax=None, ws_bins=12, wd_bins=36, ws_limit='auto', opacity=0.85,
             title=None, zoom=None, width=1000, height=760, show_sites=True):
    """Draw how often each wind speed and direction occurs, at each site.

    Parameters:
    - As for `polar_map`, without the pollutant: the value drawn is a count.

    Returns:
    - fig (go.Figure): The map.

    Notes:
    - Worth putting beside `polar_map` on the same sites. A striking feature
      in a sector the wind rarely comes from is not a finding, and this is
      what shows that.
    """
    return _directional_map(
        df, _frequency_cells, dict(ws_col=ws_col, wd_col=wd_col,
                                   ws_bins=ws_bins, wd_bins=wd_bins,
                                   ws_limit=ws_limit),
        site=site, latitude=latitude, longitude=longitude,
        radius_km=radius_km, map_style=map_style,
        colour_palette=colour_palette, n_levels=n_levels, limits=limits,
        vmin=vmin, vmax=vmax, opacity=opacity,
        title=title if title is not None else 'Wind frequency',
        label='hours', zoom=zoom, width=width, height=height,
        show_sites=show_sites)


def wind_rose_map(df, ws_col='ws', wd_col='wd', site=None, latitude=None,
                  longitude=None, radius_km=None, map_style='carto-positron',
                  colour_palette='Viridis', n_levels=6, wd_bins=16,
                  ws_limit='auto', opacity=0.9, title=None, zoom=None,
                  width=1000, height=760, show_sites=True):
    """Draw a wind rose at each site's position.

    Parameters:
    - As for `polar_map`. `n_levels` sets the wind speed bands, `wd_bins` the
      number of compass sectors.

    Returns:
    - fig (go.Figure): The map.

    Notes:
    - Petal length is the proportion of time the wind blew from that sector,
      scaled so the longest petal at any site reaches `radius_km`. The scale
      is shared, so a site with one dominant direction has a visibly longer
      petal than a site with an even distribution.
    """
    return _rose_map(
        df, value_col=ws_col, ws_col=ws_col, wd_col=wd_col, site=site,
        latitude=latitude, longitude=longitude, radius_km=radius_km,
        map_style=map_style, colour_palette=colour_palette, n_levels=n_levels,
        wd_bins=wd_bins, value_limit=ws_limit, opacity=opacity,
        title=title if title is not None else 'Wind rose',
        label='wind speed', zoom=zoom, width=width, height=height,
        show_sites=show_sites)


def pollutant_rose_map(df, pollutant='NO2', ws_col='ws', wd_col='wd',
                       site=None, latitude=None, longitude=None,
                       radius_km=None, map_style='carto-positron',
                       colour_palette='Spectral_r', n_levels=6, wd_bins=16,
                       opacity=0.9, title=None, zoom=None, width=1000,
                       height=760, show_sites=True):
    """Draw a pollution rose at each site's position.

    Parameters:
    - As for `polar_map`. The petals are split by concentration band rather
      than by wind speed.

    Returns:
    - fig (go.Figure): The map.
    """
    return _rose_map(
        df, value_col=pollutant, ws_col=ws_col, wd_col=wd_col, site=site,
        latitude=latitude, longitude=longitude, radius_km=radius_km,
        map_style=map_style, colour_palette=colour_palette, n_levels=n_levels,
        wd_bins=wd_bins, value_limit='auto', opacity=opacity,
        title=(title if title is not None
               else f'{quick_text(pollutant)} rose'),
        label=quick_text(pollutant), zoom=zoom, width=width, height=height,
        show_sites=show_sites)


def percentile_rose_map(df, pollutant='NO2', wd_col='wd', percentile=95,
                        site=None, latitude=None, longitude=None,
                        radius_km=None, map_style='carto-positron',
                        colour_palette='Spectral_r', n_levels=10,
                        limits='fixed', vmin=None, vmax=None, wd_bins=36,
                        opacity=0.85, title=None, zoom=None, width=1000,
                        height=760, show_sites=True):
    """Draw a percentile of concentration by wind direction, at each site.

    Parameters:
    - percentile (float): Percentile to draw, between 0 and 100.
    - Otherwise as for `polar_map`. There is no wind speed axis: the ring is
      filled uniformly and only the colour varies with direction.

    Returns:
    - fig (go.Figure): The map.

    Notes:
    - A high percentile answers a different question from a mean. A sector
      whose 95th percentile is high but whose mean is not is a sector with
      occasional episodes rather than a steady contribution.
    """
    if not 0 <= percentile <= 100:
        raise ValueError('percentile must be between 0 and 100.')
    return _directional_map(
        df, _percentile_cells, dict(pollutant=pollutant, wd_col=wd_col,
                                    percentile=percentile, wd_bins=wd_bins),
        site=site, latitude=latitude, longitude=longitude,
        radius_km=radius_km, map_style=map_style,
        colour_palette=colour_palette, n_levels=n_levels, limits=limits,
        vmin=vmin, vmax=vmax, opacity=opacity,
        title=(title if title is not None else
               f'{quick_text(pollutant)}, {percentile:g}th percentile by wind '
               'direction'),
        label=quick_text(pollutant), zoom=zoom, width=width, height=height,
        show_sites=show_sites)


def annulus_map(df, pollutant='NO2', wd_col='wd', period='hour',
                date_col='date_time', site=None, latitude=None, longitude=None,
                radius_km=None, map_style='carto-positron',
                colour_palette='Spectral_r', n_levels=12, limits='fixed',
                vmin=None, vmax=None, wd_bins=36, opacity=0.85, title=None,
                zoom=None, width=1000, height=760, show_sites=True):
    """Draw a polar annulus at each site: wind direction around, time through.

    Parameters:
    - period (str): 'hour', 'weekday', 'month' or 'season' - the variable
      running from the centre of the ring outwards.
    - Otherwise as for `polar_map`.

    Returns:
    - fig (go.Figure): The map.

    Notes:
    - For sources that only appear at certain hours or in certain months,
      which a plain polar plot averages away. Reading it takes a moment: the
      inner edge is the first period and the outer edge the last, so a source
      active only at night shows as a bright band at both edges of an 'hour'
      annulus rather than one arc.
    """
    return _directional_map(
        df, _annulus_cells, dict(pollutant=pollutant, wd_col=wd_col,
                                 period=period, date_col=date_col,
                                 wd_bins=wd_bins),
        site=site, latitude=latitude, longitude=longitude,
        radius_km=radius_km, map_style=map_style,
        colour_palette=colour_palette, n_levels=n_levels, limits=limits,
        vmin=vmin, vmax=vmax, opacity=opacity,
        title=(title if title is not None
               else f'{quick_text(pollutant)} by wind direction and {period}'),
        label=quick_text(pollutant), zoom=zoom, width=width, height=height,
        show_sites=show_sites)


# --------------------------------------------------------------- builders ---
# Each returns (cells, radial_extent), where a cell is
# (r0, r1, wd0, wd1, value) in the builder's own radial units. The driver
# rescales the radius onto the ground, so the builders never see kilometres.

def _polar_cells(block, pollutant, ws_col, wd_col, ws_bins, wd_bins,
                 min_count, n_splines, ws_limit, exclude_missing,
                 shared_extent=None, validate_only=False):
    """A GAM-smoothed concentration surface, as polar cells."""
    for col in (ws_col, wd_col, pollutant):
        if col not in block.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
    if validate_only:
        return
    data, ws_max = _prepare_polar_data(block, ws_col, wd_col, pollutant,
                                       ws_limit)
    if shared_extent is not None:
        ws_max = shared_extent
    _, _, Z, ws_edges, wd_edges = _polar_surface(
        data, ws_col, wd_col, pollutant, ws_max, ws_bins, wd_bins, min_count,
        n_splines, 'tile', 300, None, None, exclude_missing, None,
    )
    cells = []
    for i in range(len(ws_edges) - 1):
        for j in range(len(wd_edges) - 1):
            cells.append((ws_edges[i], ws_edges[i + 1],
                          wd_edges[j], wd_edges[j + 1], Z[j, i]))
    return cells, ws_max


def _frequency_cells(block, ws_col, wd_col, ws_bins, wd_bins, ws_limit,
                     shared_extent=None, validate_only=False):
    """Counts per wind speed and direction bin."""
    for col in (ws_col, wd_col):
        if col not in block.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
    if validate_only:
        return
    data = block[[ws_col, wd_col]].dropna()
    data = data[data[ws_col] >= 0]
    if data.empty:
        return [], 1.0
    ws_max = (shared_extent if shared_extent is not None else
              (float(data[ws_col].quantile(0.99)) if ws_limit == 'auto'
               else float(data[ws_col].max()) if ws_limit == 'max'
               else float(ws_limit)))
    ws_edges = np.linspace(0.0, ws_max, ws_bins + 1)
    wd_edges = np.linspace(0.0, 360.0, wd_bins + 1)
    counts, _, _ = np.histogram2d(
        data[ws_col].clip(upper=ws_max), data[wd_col] % 360,
        bins=[ws_edges, wd_edges])
    cells = []
    for i in range(ws_bins):
        for j in range(wd_bins):
            value = counts[i, j]
            cells.append((ws_edges[i], ws_edges[i + 1],
                          wd_edges[j], wd_edges[j + 1],
                          value if value > 0 else np.nan))
    return cells, ws_max


def _percentile_cells(block, pollutant, wd_col, percentile, wd_bins,
                      shared_extent=None, validate_only=False):
    """One ring, coloured by a percentile of concentration per sector."""
    for col in (wd_col, pollutant):
        if col not in block.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
    if validate_only:
        return
    data = block[[wd_col, pollutant]].dropna()
    if data.empty:
        return [], 1.0
    wd_edges = np.linspace(0.0, 360.0, wd_bins + 1)
    sector = np.clip(np.digitize(data[wd_col] % 360, wd_edges) - 1,
                     0, wd_bins - 1)
    cells = []
    for j in range(wd_bins):
        values = data[pollutant].to_numpy()[sector == j]
        cells.append((0.0, 1.0, wd_edges[j], wd_edges[j + 1],
                      float(np.percentile(values, percentile))
                      if len(values) else np.nan))
    return cells, 1.0


_ANNULUS_PERIODS = {'hour': 24, 'weekday': 7, 'month': 12, 'season': 4}


def _annulus_cells(block, pollutant, wd_col, period, date_col, wd_bins,
                   shared_extent=None, validate_only=False):
    """Wind direction around the ring, a temporal variable through it."""
    for col in (wd_col, pollutant):
        if col not in block.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
    if date_col not in block.columns:
        raise ValueError(f"Date column '{date_col}' not found in the DataFrame.")
    if period not in _ANNULUS_PERIODS:
        raise ValueError(f'period must be one of {sorted(_ANNULUS_PERIODS)}.')
    if validate_only:
        return

    stamps = pd.to_datetime(block[date_col])
    periods = {'hour': stamps.dt.hour, 'weekday': stamps.dt.dayofweek,
               'month': stamps.dt.month - 1,
               'season': (stamps.dt.month % 12) // 3}
    steps = _ANNULUS_PERIODS[period]

    data = pd.DataFrame({'wd': block[wd_col] % 360, 'value': block[pollutant],
                         'step': periods[period]}).dropna()
    if data.empty:
        return [], float(steps)

    wd_edges = np.linspace(0.0, 360.0, wd_bins + 1)
    sector = np.clip(np.digitize(data['wd'], wd_edges) - 1, 0, wd_bins - 1)
    means = (data.assign(sector=sector)
             .groupby(['step', 'sector'], observed=True)['value'].mean())

    cells = []
    for step in range(steps):
        for j in range(wd_bins):
            cells.append((float(step), float(step + 1),
                          wd_edges[j], wd_edges[j + 1],
                          float(means.get((step, j), np.nan))))
    return cells, float(steps)


# ---------------------------------------------------------------- drivers ---

def _shared_extent(builder, frames, builder_kwargs):
    """One radial extent across every site.

    Resolved before any fitting, so that a given radius means the same wind
    speed in every marker. Without it each site would set its own and the
    markers could not be compared, which is the only reason to draw them
    together.
    """
    extents = []
    for _, _, _, block in frames:
        try:
            _, extent = builder(block, shared_extent=None, **builder_kwargs)
        except ValueError:
            continue
        extents.append(extent)
    if not extents:
        raise ValueError('No site had enough data to draw.')
    return max(extents)


def _build_sites(builder, frames, builder_kwargs, shared_extent):
    """Build every site's cells, skipping the ones that cannot be fitted."""
    built = []
    for name, lat, lon, block in frames:
        try:
            cells, _ = builder(block, shared_extent=shared_extent,
                               **builder_kwargs)
        except ValueError:
            # A site too sparse to fit is skipped rather than failing the map,
            # which would let one bad site hide every good one. Configuration
            # errors are caught before this loop, so only sparsity lands here.
            continue
        if cells:
            built.append((name, lat, lon, cells))
    if not built:
        raise ValueError(
            'No site had enough data to draw. Lower min_count, or check that '
            'the wind and concentration columns are populated.')
    return built


def _site_geometry(lat, lon, cells, extent, radius_km, low, high, scale_min,
                   scale_max):
    """Project one site's polar cells onto the ground."""
    span = (high - low) or 1.0
    width = (scale_max - scale_min) or 1.0
    out = []
    for r0, r1, wd0, wd1, value in cells:
        if not np.isfinite(value):
            continue
        lats, lons = _sector(lat, lon, r0 / extent * radius_km,
                             r1 / extent * radius_km, wd0, wd1)
        # Rescaled onto the shared scale, so limits='free' still uses one set
        # of bands and one colourbar while each site spans all of them.
        out.append((lats, lons, scale_min + (value - low) / span * width))
    return out


def _directional_map(df, builder, builder_kwargs, site, latitude, longitude,
                     radius_km, map_style, colour_palette, n_levels, limits,
                     vmin, vmax, opacity, title, label, zoom, width, height,
                     show_sites):
    """Build one surface per site and draw them all on one map."""
    if limits not in ('fixed', 'free'):
        raise ValueError("limits must be 'fixed' or 'free'.")
    if n_levels < 2:
        raise ValueError('n_levels must be at least 2.')

    site_col = _resolve_column(df, site, _SITE_NAMES, 'site')
    lat_col = _resolve_column(df, latitude, _LATITUDE_NAMES, 'latitude')
    lon_col = _resolve_column(df, longitude, _LONGITUDE_NAMES, 'longitude')
    frames = _site_frames(df, site_col, lat_col, lon_col)

    # Checked once against the whole frame, before any fitting. The per-site
    # loop below deliberately swallows ValueError so that one unusable site
    # cannot hide every good one, and without this a mistyped column name
    # would come back as "no site had enough data" rather than naming it.
    builder(df, validate_only=True, **builder_kwargs)

    shared_extent = _shared_extent(builder, frames, builder_kwargs)
    built = _build_sites(builder, frames, builder_kwargs, shared_extent)

    values = np.array([c[4] for _, _, _, cells in built for c in cells],
                      dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError('Every cell was empty or excluded.')
    scale_min = float(np.nanmin(finite)) if vmin is None else vmin
    scale_max = float(np.nanmax(finite)) if vmax is None else vmax

    positions = [(lat, lon) for _, lat, lon, _ in built]
    if radius_km is None:
        radius_km = _auto_radius(positions)
    if radius_km <= 0:
        raise ValueError('radius_km must be positive.')

    fig = go.Figure()
    geo_cells, hovers = [], []
    per_site = (limits == 'free' and vmin is None and vmax is None)
    for name, lat, lon, cells in built:
        finite_here = np.array([c[4] for c in cells], dtype=float)
        finite_here = finite_here[np.isfinite(finite_here)]
        low, high = ((float(finite_here.min()), float(finite_here.max()))
                     if per_site and finite_here.size
                     else (scale_min, scale_max))
        geo_cells.extend(_site_geometry(lat, lon, cells, shared_extent,
                                        radius_km, low, high, scale_min,
                                        scale_max))
        hovers.append(f'{label}: {finite_here.mean():.3g} mean over '
                      f'{len(finite_here)} cells'
                      if finite_here.size else 'no data')

    _add_bands(fig, geo_cells, scale_min, scale_max, colour_palette, n_levels,
               opacity, label)
    return _finalise(fig, [(n, la, lo, None) for n, la, lo, _ in built],
                     radius_km, map_style, title, zoom, width, height,
                     show_sites, hovers)


def _rose_counts(frames, wd_col, value_col, wd_edges, edges, wd_bins,
                 n_levels):
    """Proportion of observations per (sector, value band), for each site."""
    built = []
    for name, lat, lon, block in frames:
        data = block[[wd_col, value_col]].dropna()
        if data.empty:
            continue
        sector = np.clip(np.digitize(data[wd_col] % 360, wd_edges) - 1,
                         0, wd_bins - 1)
        band = np.clip(np.digitize(data[value_col], edges) - 1, 0, n_levels - 1)
        counts = np.zeros((wd_bins, n_levels))
        np.add.at(counts, (sector, band), 1.0)
        built.append((name, lat, lon, counts / max(len(data), 1)))
    return built


def _stack_petals(grouped, lat, lon, counts, wd_edges, longest, radius_km):
    """Stack one site's petals outwards, band by band.

    Each band starts where the previous one ended, so petal length is the
    total frequency for the sector and the colours divide it.
    """
    wd_bins, n_levels = counts.shape
    for j in range(wd_bins):
        base = 0.0
        for b in range(n_levels):
            fraction = counts[j, b] / longest
            if fraction <= 0:
                continue
            lats, lons = _sector(lat, lon, base * radius_km,
                                 (base + fraction) * radius_km,
                                 wd_edges[j], wd_edges[j + 1])
            grouped[b][0].extend([*lats, None])
            grouped[b][1].extend([*lons, None])
            base += fraction


def _rose_map(df, value_col, ws_col, wd_col, site, latitude, longitude,
              radius_km, map_style, colour_palette, n_levels, wd_bins,
              value_limit, opacity, title, label, zoom, width, height,
              show_sites):
    """Stacked petals: length is frequency, colour is a band of `value_col`."""
    if n_levels < 2:
        raise ValueError('n_levels must be at least 2.')
    if wd_bins < 4:
        raise ValueError('wd_bins must be at least 4.')

    site_col = _resolve_column(df, site, _SITE_NAMES, 'site')
    lat_col = _resolve_column(df, latitude, _LATITUDE_NAMES, 'latitude')
    lon_col = _resolve_column(df, longitude, _LONGITUDE_NAMES, 'longitude')
    for col in (wd_col, value_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
    frames = _site_frames(df, site_col, lat_col, lon_col)

    pooled = pd.to_numeric(df[value_col], errors='coerce').dropna()
    if pooled.empty:
        raise ValueError(f"Column '{value_col}' has no usable values.")
    top = (float(pooled.quantile(0.99)) if value_limit == 'auto'
           else float(pooled.max()) if value_limit == 'max'
           else float(value_limit))
    # Bands shared across sites: the same colour must mean the same value in
    # every petal, or the map cannot be read across its markers.
    edges = np.linspace(float(max(pooled.min(), 0.0)), top, n_levels + 1)
    wd_edges = np.linspace(0.0, 360.0, wd_bins + 1)

    built = _rose_counts(frames, wd_col, value_col, wd_edges, edges, wd_bins,
                         n_levels)
    if not built:
        raise ValueError('No site had usable wind direction and value data.')

    longest = max(float(counts.sum(axis=1).max()) for _, _, _, counts in built)
    if longest <= 0:
        raise ValueError('No site had any complete observations to count.')

    positions = [(lat, lon) for _, lat, lon, _ in built]
    if radius_km is None:
        radius_km = _auto_radius(positions)
    if radius_km <= 0:
        raise ValueError('radius_km must be positive.')

    colours = _band_colours(colour_palette, n_levels)
    fig = go.Figure()
    grouped = [([], []) for _ in range(n_levels)]
    hovers = []
    for name, lat, lon, counts in built:
        _stack_petals(grouped, lat, lon, counts, wd_edges, longest, radius_km)
        dominant = wd_edges[int(np.argmax(counts.sum(axis=1)))]
        hovers.append(f'most frequent wind from {dominant:.0f}°')

    for b, (lat_list, lon_list) in enumerate(grouped):
        if not lat_list:
            continue
        fig.add_trace(go.Scattermap(
            lat=lat_list, lon=lon_list, mode='lines', fill='toself',
            fillcolor=colours[b], line=dict(width=0, color=colours[b]),
            opacity=opacity, hoverinfo='skip', showlegend=False,
            name=f'{edges[b]:.3g} - {edges[b + 1]:.3g}',
        ))
    fig.add_trace(go.Scattermap(
        lat=[None], lon=[None], mode='markers', showlegend=False,
        hoverinfo='skip',
        marker=dict(size=0, color=[], colorscale=colour_palette,
                    cmin=float(edges[0]), cmax=float(edges[-1]), showscale=True,
                    colorbar=dict(title=label, thickness=18, len=0.7, y=0.5)),
    ))
    return _finalise(fig, [(n, la, lo, None) for n, la, lo, _ in built],
                     radius_km, map_style, title, zoom, width, height,
                     show_sites, hovers)
