"""Conditioning support: openair's ``type`` argument.

openair introduces ``type`` in chapter 2, before any plotting chapter, because
nearly every function takes it. ``type='season'`` draws the same plot once per
season, ``type='weekday'`` once per day of the week, and so on.

Rather than teach each plotting function to build its own grid of panels, this
splits the data, calls the function once per level, and stitches the resulting
figures into a single subplot grid. Plot functions therefore stay simple and
gain conditioning for free.
"""

import copy
from math import ceil

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_utils import cut_data

__all__ = ['facet_by_type']

_POLAR_TRACES = {'scatterpolar', 'barpolar', 'scatterpolargl'}
_MAP_TRACES = {'scattermapbox', 'scattermap', 'densitymapbox', 'densitymap'}

# Axis properties safe to carry across. Anything positional (domain, anchor)
# belongs to the subplot grid and must not be copied from the source figure.
_AXIS_PROPERTIES = ('visible', 'range', 'title', 'type', 'tickmode', 'tickvals',
                    'ticktext', 'dtick', 'tickangle', 'autorange', 'showgrid',
                    'zeroline', 'tickformat')


def _subplot_type(fig):
    """Which subplot kind the traces in `fig` need."""
    for trace in fig.data:
        if trace.type in _POLAR_TRACES:
            return 'polar'
        if trace.type in _MAP_TRACES:
            return 'mapbox'
    return 'xy'


def _levels_of(series):
    """Ordered, present levels of a conditioning column."""
    present = set(series.dropna().unique())
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [c for c in series.cat.categories if c in present]
    return sorted(present)


def _axis_names(index):
    """Plotly's axis keys for the index-th subplot cell."""
    suffix = '' if index == 0 else str(index + 1)
    return f'xaxis{suffix}', f'yaxis{suffix}', f'x{suffix}', f'y{suffix}'


def _harmonise_colour_scales(fig):
    """Give every colour-mapped trace one shared scale and one colourbar.

    Panels are only comparable if they share limits, and N colourbars for N
    panels is noise. Applied after combining, so plot functions do not need to
    know they are being faceted.
    """
    scaled = [t for t in fig.data if hasattr(t, 'zmin') and getattr(t, 'z', None) is not None]
    if not scaled:
        return

    values = np.concatenate([
        np.asarray(t.z, dtype=float).ravel() for t in scaled
    ])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    low, high = float(values.min()), float(values.max())
    for position, trace in enumerate(scaled):
        trace.update(zmin=low, zmax=high, showscale=(position == 0))


def facet_by_type(plot_fn, df, type, date_col='date_time', hemisphere='northern',
                  n_levels=4, ncols=3, width=None, height=None,
                  panel_height=380, latitude=None, longitude=None, **kwargs):
    """Draw `plot_fn` once per level of `type` and combine into one figure.

    Parameters:
    - plot_fn (callable): A plotting function taking a DataFrame first. It may
      return a figure, or a (figure, DataFrame) pair.
    - df (pd.DataFrame): Input data.
    - type (str): Conditioning variable, as accepted by `cut_data`: 'year',
      'month', 'season', 'weekday', 'weekend', 'hour', 'daylight', 'wd', or a
      numeric column name split into quantiles.
    - date_col (str), hemisphere (str), n_levels (int): Passed to `cut_data`.
    - latitude, longitude (float): Passed to `cut_data`; required by
      type='daylight' for a real sunrise/sunset split.
    - ncols (int): Panels per row.
    - width, height (int or None): Figure size; height defaults to
      `panel_height` per row.
    - **kwargs: Forwarded to `plot_fn`.

    Returns:
    - fig (go.Figure): The faceted figure.
    - summary (pd.DataFrame or None): The per-level summaries concatenated with
      a column named after `type`, or None if `plot_fn` returns only a figure.
    """
    tagged = cut_data(df, type=type, date_col=date_col, hemisphere=hemisphere,
                      n_levels=n_levels, latitude=latitude, longitude=longitude)
    levels = _levels_of(tagged[type])
    if not levels:
        raise ValueError(f"Conditioning on '{type}' produced no levels.")

    figures, summaries = [], []
    returned_tuple = False
    for level in levels:
        subset = tagged[tagged[type] == level]
        # The conditioning column is an artefact of splitting; passing it on
        # would show up in functions that plot every numeric column.
        result = plot_fn(subset.drop(columns=[type], errors='ignore'), **kwargs)
        if isinstance(result, tuple):
            returned_tuple = True
            figure, summary = result[0], result[1]
        else:
            figure, summary = result, None
        figures.append(figure)
        if isinstance(summary, pd.DataFrame):
            summaries.append(summary.assign(**{type: level}))

    kind = _subplot_type(figures[0])
    ncols = _choose_columns(len(levels), ncols)
    nrows = ceil(len(levels) / ncols)

    combined = make_subplots(
        rows=nrows, cols=ncols,
        specs=[[{'type': kind} for _ in range(ncols)] for _ in range(nrows)],
        subplot_titles=[str(level) for level in levels],
        horizontal_spacing=0.06, vertical_spacing=0.10,
    )

    for index, source in enumerate(figures):
        row, col = index // ncols + 1, index % ncols + 1
        _transfer(source, combined, index, row, col, kind)

    _harmonise_colour_scales(combined)
    _harmonise_ranges(combined, len(levels), kind)

    combined.update_layout(
        template=figures[0].layout.template,
        title=figures[0].layout.title.text,
        width=width, height=height or panel_height * nrows,
        showlegend=any(t.showlegend for t in combined.data if t.showlegend),
    )

    # Match what the wrapped function returns. Conditioning must not change a
    # function's return shape, or callers cannot write uniform code: polar_plot
    # returns a figure, so polar_plot(type='season') must too.
    if not returned_tuple:
        return combined
    summary = pd.concat(summaries, ignore_index=True) if summaries else None
    return combined, summary


def _choose_columns(n_levels, requested):
    """Pick a column count that leaves as few empty cells as possible.

    Four seasons across three columns leaves two blanks; two columns gives a
    tidy 2x2. Only a small reduction is considered, so seven weekdays are not
    forced into a single tall column.
    """
    requested = max(1, min(requested, n_levels))
    for candidate in (requested, requested - 1):
        if candidate >= 1 and n_levels % candidate == 0:
            return candidate
    return requested


def _harmonise_ranges(fig, n_levels, kind):
    """Give every panel the same axes, so they can actually be compared.

    Each panel is computed from its own subset, so a data-driven limit — the
    99th percentile of wind speed, say — differs between them. Left alone, a
    ring at the same radius would mean a different wind speed in each panel.
    """
    if kind == 'polar':
        keys = ['polar' if i == 0 else f'polar{i + 1}' for i in range(n_levels)]
        ranges = [fig.layout[k].radialaxis.range for k in keys
                  if fig.layout[k] is not None and fig.layout[k].radialaxis.range]
        if ranges:
            low = min(r[0] for r in ranges)
            high = max(r[1] for r in ranges)
            for key in keys:
                if fig.layout[key] is not None:
                    fig.layout[key].radialaxis.update(range=[low, high])
        return

    for axis in ('xaxis', 'yaxis'):
        keys = [axis if i == 0 else f'{axis}{i + 1}' for i in range(n_levels)]
        ranges = [fig.layout[k].range for k in keys
                  if fig.layout[k] is not None and fig.layout[k].range is not None]
        if len(ranges) < 2:
            continue
        low = min(r[0] for r in ranges)
        high = max(r[1] for r in ranges)
        for key in keys:
            if fig.layout[key] is not None:
                fig.layout[key].update(range=[low, high])


def _transfer(source, combined, index, row, col, kind):
    """Move one panel's traces, shapes, annotations and axis setup across."""
    for trace in source.data:
        moved = copy.deepcopy(trace)
        # Only the first panel carries the legend, otherwise every series is
        # repeated once per level.
        if index > 0 and moved.showlegend is not False:
            moved.showlegend = False
        combined.add_trace(moved, row=row, col=col)

    # Shapes and annotations tied to data coordinates belong to their panel.
    # Paper-referenced ones were positioned against the whole original figure
    # and would land somewhere arbitrary here, so they are dropped.
    for shape in source.layout.shapes or ():
        if shape.xref in (None, 'x') and shape.yref in (None, 'y'):
            combined.add_shape(copy.deepcopy(shape), row=row, col=col)
    for annotation in source.layout.annotations or ():
        if annotation.xref in (None, 'x') and annotation.yref in (None, 'y'):
            combined.add_annotation(copy.deepcopy(annotation), row=row, col=col)

    if kind == 'polar':
        key = 'polar' if index == 0 else f'polar{index + 1}'
        if source.layout.polar is not None:
            settings = source.layout.polar.to_plotly_json()
            settings.pop('domain', None)
            combined.layout[key].update(settings)
        return

    x_key, y_key, x_ref, y_ref = _axis_names(index)
    for source_axis, key in ((source.layout.xaxis, x_key),
                             (source.layout.yaxis, y_key)):
        settings = {
            name: source_axis[name] for name in _AXIS_PROPERTIES
            if source_axis[name] is not None
        }
        combined.layout[key].update(settings)

    # Equal aspect must be re-anchored to this cell's own y axis; carrying the
    # original 'y' across would tie every panel to the first one.
    if source.layout.xaxis.scaleanchor is not None:
        combined.layout[x_key].update(
            scaleanchor=y_ref, scaleratio=source.layout.xaxis.scaleratio or 1,
        )


def conditionable(fn):
    """Give a plotting function openair's ``type`` argument.

    Without ``type`` the call goes straight through, so there is no cost to
    decorating a function. With it, the data is split and the function is run
    once per level, the panels combined into one figure.

    The wrapper also adds ``ncols``, ``hemisphere``, ``n_levels`` and
    ``panel_height``, which only apply when conditioning.
    """
    import functools
    import inspect

    try:
        own = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        own = set()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        condition = kwargs.pop('type', None)
        # Claimed only if the wrapped function does not define the name
        # itself, so decorating never silently swallows its arguments.
        facet_kwargs = {
            name: kwargs.pop(name)
            for name in ('ncols', 'hemisphere', 'n_levels', 'panel_height',
                         'latitude', 'longitude')
            if name in kwargs and name not in own
        }
        if condition is None:
            return fn(*args, **kwargs)
        if not args:
            raise TypeError(f'{fn.__name__}() requires a DataFrame.')

        df, rest = args[0], args[1:]
        return facet_by_type(
            lambda subset: fn(subset, *rest, **kwargs),
            df, condition,
            date_col=kwargs.get('date_col', 'date_time'),
            **facet_kwargs,
        )

    # functools.wraps copies __wrapped__, which makes inspect.signature report
    # the undecorated signature and hide the arguments added here. State the
    # combined signature explicitly so help() and introspection stay honest.
    try:
        original = inspect.signature(fn)
        extra = [
            inspect.Parameter('type', inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter('ncols', inspect.Parameter.KEYWORD_ONLY, default=3),
            inspect.Parameter('hemisphere', inspect.Parameter.KEYWORD_ONLY,
                              default='northern'),
            inspect.Parameter('n_levels', inspect.Parameter.KEYWORD_ONLY, default=4),
            inspect.Parameter('panel_height', inspect.Parameter.KEYWORD_ONLY,
                              default=380),
            inspect.Parameter('latitude', inspect.Parameter.KEYWORD_ONLY,
                              default=None),
            inspect.Parameter('longitude', inspect.Parameter.KEYWORD_ONLY,
                              default=None),
        ]
        existing = set(original.parameters)
        keep = [p for p in original.parameters.values()
                if p.kind is not inspect.Parameter.VAR_KEYWORD]
        wrapper.__signature__ = original.replace(
            parameters=keep + [p for p in extra if p.name not in existing]
        )
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        pass

    wrapper.__doc__ = (fn.__doc__ or '') + _TYPE_DOC
    return wrapper


_TYPE_DOC = """

    Conditioning (added by the ``type`` decorator):
    - type (str or None): Split the data and draw one panel per level, using
      any value `cut_data` accepts: 'year', 'month', 'season', 'weekday',
      'weekend', 'hour', 'daylight', 'wd', or a numeric column name. None
      (default) draws a single plot.
    - ncols (int): Panels per row when conditioning.
    - hemisphere (str): 'northern' or 'southern', for season definitions.
    - n_levels (int): Quantile count when `type` names a numeric column.
    - panel_height (int): Height in pixels of each panel row.
    - latitude, longitude (float): Site position, in degrees north and east.
      Only used by type='daylight', which needs them to know when the sun
      actually rose; without them it falls back to a fixed window and warns.

    When conditioning, panels share one colour scale and one set of axis
    limits so they can be compared. Any other keyword is passed to every
    panel, so a data-driven limit can be fixed across them by giving it
    explicitly, for example ``ws_limit=12``.
    """


def _is_figure(value):
    return isinstance(value, go.Figure)
