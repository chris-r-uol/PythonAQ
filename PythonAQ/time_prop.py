"""Time proportion plots. Port of openair's ``timeProp``.

A stacked bar time series where each bar is split by the contribution of some
category — wind sector, season, a source label. The bar heights give the total,
the segments give who is responsible for it, which is the usual way source
apportionment output gets read.
"""

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go

from .data_utils import _parse_avg_time, cut_data
from .text import quick_text

__all__ = ['time_prop']


def time_prop(df, pollutant, proportion, avg_time='month', date_col='date_time',
              statistic='mean', normalise=False, n_levels=4,
              hemisphere='northern', colours=None, title=None, ylab=None,
              width=1000, height=560):
    """Stacked bars over time, split by the contribution of a category.

    Parameters:
    - df (pd.DataFrame): Input data.
    - pollutant (str): Column giving the bar heights.
    - proportion (str): Column splitting each bar. May be an existing
      categorical column, a numeric column (split into `n_levels` quantiles),
      or anything `cut_data` understands such as 'season' or 'wd'.
    - avg_time (str): Bar width in time, e.g. 'month' (default), 'week', 'year'.
    - date_col (str): Name of the datetime column.
    - statistic (str): 'mean' or 'sum'. 'mean' splits the period mean by each
      category's share; 'sum' stacks raw totals.
    - normalise (bool): Scale every bar to 100%, to compare composition rather
      than magnitude.
    - n_levels (int): Quantile count when `proportion` is numeric.
    - hemisphere (str): Passed to `cut_data` for season definitions.
    - colours (list or None): Segment colours.
    - title, ylab (str or None): Labels; generated when None.
    - width, height (int): Figure size in pixels.

    Returns:
    - fig (go.Figure): The stacked bar series.
    - summary (pd.DataFrame): Value per period and category.
    """
    for column in (pollutant, date_col):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
    if statistic not in ('mean', 'sum'):
        raise ValueError("statistic must be 'mean' or 'sum'.")

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.dropna(subset=[date_col, pollutant])
    if data.empty:
        raise ValueError('No rows with both a valid date and a value.')

    # Resolve the splitting column. An existing categorical is used as-is;
    # anything else goes through cut_data, which also handles 'season', 'wd'
    # and numeric quantiles.
    if (proportion in data.columns
            and not pd.api.types.is_numeric_dtype(data[proportion])):
        split = data[proportion].astype('category')
    else:
        data = cut_data(data, type=proportion, date_col=date_col,
                        hemisphere=hemisphere, n_levels=n_levels)
        split = data[proportion]
    data['_split'] = split

    levels = (list(split.cat.categories)
              if isinstance(split.dtype, pd.CategoricalDtype)
              else sorted(split.dropna().unique()))
    levels = [lv for lv in levels if lv in set(split.dropna())]

    freq = _parse_avg_time(avg_time)
    period = data[date_col].dt.to_period(_period_alias(freq))
    data['_period'] = period.dt.start_time

    # Each segment is that category's share of the period, so the bars still
    # total the period statistic rather than summing category means.
    totals = data.groupby('_period', observed=True)[pollutant].agg(statistic)
    counts = data.groupby(['_period', '_split'], observed=True)[pollutant].agg(
        ['sum', 'count']
    ).reset_index()
    period_sums = counts.groupby('_period', observed=True)['sum'].transform('sum')

    with np.errstate(invalid='ignore', divide='ignore'):
        share = counts['sum'] / period_sums.replace(0, np.nan)
    counts['share'] = share
    counts['value'] = counts['_period'].map(totals) * counts['share']

    summary = counts.rename(columns={'_period': date_col, '_split': proportion})
    summary = summary[[date_col, proportion, 'sum', 'count', 'share', 'value']]

    if normalise:
        summary['value'] = summary['share'] * 100.0

    colours = colours or pcolors.qualitative.Safe
    fig = go.Figure()
    for index, level in enumerate(levels):
        rows = summary[summary[proportion] == level]
        if rows.empty:
            continue
        fig.add_trace(go.Bar(
            x=rows[date_col], y=rows['value'], name=str(level),
            marker_color=colours[index % len(colours)],
            hovertemplate=(f'{level}<br>%{{x|%b %Y}}<br>%{{y:.2f}}'
                           f'<br>%{{customdata}} observations<extra></extra>'),
            customdata=rows['count'],
        ))

    default_ylab = ('share of total (%)' if normalise
                    else f'{quick_text(pollutant)} ({statistic})')
    fig.update_layout(
        barmode='stack',
        title=title or f'{quick_text(pollutant)} by {proportion}',
        xaxis_title='date', yaxis_title=ylab or default_ylab,
        template='plotly_white', width=width, height=height,
        legend=dict(title=proportion, orientation='h', yanchor='bottom',
                    y=1.02, xanchor='center', x=0.5),
        bargap=0.05,
    )
    if normalise:
        fig.update_yaxes(range=[0, 100])
    return fig, summary


def _period_alias(freq):
    """Map a resampling offset onto the period alias to_period wants."""
    head = ''.join(ch for ch in freq if ch.isalpha()).upper()
    mapping = {'H': 'h', 'D': 'D', 'W': 'W', 'MS': 'M', 'ME': 'M', 'M': 'M',
               'QS': 'Q', 'QE': 'Q', 'Q': 'Q', 'YS': 'Y', 'YE': 'Y', 'Y': 'Y',
               'QSDEC': 'Q'}
    return mapping.get(head, head or 'M')
