"""Heat maps of atmospheric composition. Port of openair's ``trendLevel``."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_utils import cut_data
from .text import quick_text

__all__ = ['trend_level']

_VALID_AXES = ['hour', 'weekday', 'month', 'year', 'season', 'monthyear', 'wd']


def _axis_values(data, name, date_col, hemisphere):
    """Return a labelled, correctly ordered categorical for one axis."""
    if name in data.columns and name not in _VALID_AXES:
        return data[name], sorted(data[name].dropna().unique())
    cut = cut_data(data, type=name, date_col=date_col, hemisphere=hemisphere)
    values = cut[name]
    if isinstance(values.dtype, pd.CategoricalDtype):
        order = [c for c in values.cat.categories if c in set(values.dropna())]
    else:
        order = sorted(values.dropna().unique())
    return values, order


def trend_level(df, pollutant, x='month', y='hour', type='year',
                statistic='mean', date_col='date_time', colorscale='Spectral_r',
                hemisphere='northern', title=None, subplot_cols=3,
                fig_width=1100, fig_height=700, zmin=None, zmax=None):
    """Plot a pollutant as a heat map over two time dimensions.

    Parameters:
    - df (pd.DataFrame): Input data.
    - pollutant (str): Column to aggregate.
    - x, y (str): Axis variables, from 'hour', 'weekday', 'month', 'year',
      'season', 'monthyear', 'wd', or any existing column.
    - type (str or None): Variable to split into separate panels; None for one.
    - statistic (str): 'mean', 'median', 'max', 'min' or 'frequency'.
    - date_col (str): Name of the datetime column.
    - colorscale (str): Plotly colour scale name.
    - hemisphere (str): 'northern' or 'southern', for season definitions.
    - title (str or None): Plot title.
    - subplot_cols (int): Panels per row when `type` is set.
    - fig_width, fig_height (int): Figure size in pixels.
    - zmin, zmax (float or None): Fix the colour scale limits across panels.

    Returns:
    - fig (go.Figure): The heat map.
    - summary (pd.DataFrame): Long-format aggregated values.
    """
    if pollutant not in df.columns:
        raise ValueError(f"Column '{pollutant}' not found in the DataFrame.")
    aggregators = {'mean': 'mean', 'median': 'median', 'max': 'max',
                   'min': 'min', 'frequency': 'count'}
    if statistic not in aggregators:
        raise ValueError(
            f"Unknown statistic '{statistic}'. Choose from {sorted(aggregators)}."
        )

    data = df.copy()
    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col])

    data['_x'], x_order = _axis_values(data, x, date_col, hemisphere)
    data['_y'], y_order = _axis_values(data, y, date_col, hemisphere)
    if type is not None:
        data['_panel'], panel_order = _axis_values(data, type, date_col, hemisphere)
    else:
        data['_panel'], panel_order = 'all', ['all']

    summary = (
        data.dropna(subset=[pollutant])
        .groupby(['_panel', '_x', '_y'], observed=True)[pollutant]
        .agg(aggregators[statistic])
        .reset_index()
        .rename(columns={'_panel': type or 'panel', '_x': x, '_y': y,
                         pollutant: statistic})
    )

    if zmin is None or zmax is None:
        finite = summary[statistic].replace([np.inf, -np.inf], np.nan).dropna()
        zmin = finite.min() if zmin is None and len(finite) else zmin
        zmax = finite.max() if zmax is None and len(finite) else zmax

    n_panels = len(panel_order)
    ncols = min(subplot_cols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[str(p) for p in panel_order],
        shared_xaxes=False, shared_yaxes=True,
        horizontal_spacing=0.06, vertical_spacing=0.10,
    )

    for index, panel in enumerate(panel_order):
        subset = data[data['_panel'] == panel]
        grid = (
            subset.dropna(subset=[pollutant])
            .pivot_table(index='_y', columns='_x', values=pollutant,
                         aggfunc=aggregators[statistic], observed=True)
            .reindex(index=y_order, columns=x_order)
        )
        row, col = index // ncols + 1, index % ncols + 1
        fig.add_trace(go.Heatmap(
            z=grid.values,
            x=[str(v) for v in grid.columns],
            y=[str(v) for v in grid.index],
            colorscale=colorscale, zmin=zmin, zmax=zmax,
            showscale=(index == 0),
            colorbar=dict(title=f'{quick_text(pollutant)}<br>({statistic})') if index == 0 else None,
            hovertemplate=f'{x}: %{{x}}<br>{y}: %{{y}}<br>{statistic}: %{{z:.1f}}<extra></extra>',
        ), row=row, col=col)
        fig.update_xaxes(title_text=x, row=row, col=col)
        fig.update_yaxes(title_text=y, row=row, col=1)

    fig.update_layout(
        title=title or f'Trend level: {quick_text(pollutant)}',
        template='plotly_white', width=fig_width, height=fig_height,
    )
    return fig, summary
