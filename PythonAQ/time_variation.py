"""Temporal variation plots. Port of openair's ``timeVariation``.

Produces the familiar four-panel summary of how concentrations vary by hour of
day, hour split by weekday, month of year and day of week, with bootstrap
confidence intervals in the mean.
"""

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .faceting import conditionable

__all__ = ['time_variation']

_WEEKDAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                  'Saturday', 'Sunday']
_MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def _bootstrap_interval(values, statistic='mean', conf_int=0.95, n_boot=100, rng=None):
    """Bootstrap a confidence interval for the mean (or median) of `values`.

    Returns (centre, lower, upper). Resampling is vectorised: all `n_boot`
    replicates are drawn as a single (n_boot, n) index matrix.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan

    centre = np.median(values) if statistic == 'median' else np.mean(values)
    if values.size == 1:
        return centre, centre, centre

    rng = rng if rng is not None else np.random.default_rng()
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    replicates = values[idx]
    stats = (np.median(replicates, axis=1) if statistic == 'median'
             else np.mean(replicates, axis=1))

    tail = (1.0 - conf_int) / 2.0
    lower, upper = np.quantile(stats, [tail, 1.0 - tail])
    return centre, lower, upper


def _summarise(df, group_col, value_col, statistic, conf_int, n_boot, rng):
    """Aggregate `value_col` by `group_col` with bootstrap intervals."""
    rows = []
    for key, group in df.groupby(group_col, observed=True, sort=True):
        centre, lower, upper = _bootstrap_interval(
            group[value_col], statistic, conf_int, n_boot, rng
        )
        rows.append({group_col: key, 'value': centre,
                     'lower': lower, 'upper': upper})
    return pd.DataFrame(rows)


def _add_series(fig, x, summary, name, colour, row, col, show_ci, showlegend):
    """Draw one line with a shaded confidence band."""
    rgb = pcolors.unlabel_rgb(colour) if colour.startswith('rgb') else pcolors.hex_to_rgb(colour)
    fill = f'rgba({rgb[0]},{rgb[1]},{rgb[2]},0.2)'

    if show_ci and summary['lower'].notna().any():
        fig.add_trace(go.Scatter(
            x=list(x) + list(x)[::-1],
            y=list(summary['upper']) + list(summary['lower'])[::-1],
            fill='toself', fillcolor=fill, line=dict(width=0),
            hoverinfo='skip', showlegend=False, name=name,
        ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=x, y=summary['value'], mode='lines+markers', name=name,
        line=dict(color=colour, width=2), marker=dict(size=5, color=colour),
        legendgroup=name, showlegend=showlegend,
    ), row=row, col=col)


@conditionable
def time_variation(df, pollutant, date_col='date_time', statistic='mean',
                   conf_int=0.95, n_boot=100, normalise=False, ci=True,
                   title='Time Variation', colours=None, width=1100,
                   height=750, random_state=None):
    """Plot how concentrations vary by hour, weekday and month.

    Port of openair's ``timeVariation``. The four panels are: hour of day split
    by weekday (top, full width), then day of week, hour of day and month.

    Parameters:
    - df (pd.DataFrame): Input data.
    - pollutant (str or list): One or more columns to plot.
    - date_col (str): Name of the datetime column.
    - statistic (str): 'mean' or 'median'.
    - conf_int (float): Confidence level for the bootstrap interval, e.g. 0.95.
    - n_boot (int): Number of bootstrap replicates (openair's `B`, default 100).
    - normalise (bool): Divide each series by its own mean, so that variables on
      different scales can be compared by shape.
    - ci (bool): Whether to draw the confidence bands.
    - title (str): Plot title.
    - colours (list or None): Line colours; defaults to a qualitative palette.
    - width, height (int): Figure size in pixels.
    - random_state (int or None): Seed, for reproducible bootstrap intervals.

    Returns:
    - fig (go.Figure): The four-panel figure.
    - summary (pd.DataFrame): Long-format table of every plotted statistic.
    """
    pollutants = [pollutant] if isinstance(pollutant, str) else list(pollutant)
    missing = [p for p in pollutants if p not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in the DataFrame: {missing}")
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in the DataFrame.")
    if statistic not in ('mean', 'median'):
        raise ValueError("statistic must be 'mean' or 'median'.")
    if not 0 < conf_int < 1:
        raise ValueError('conf_int must be strictly between 0 and 1.')

    data = df[[date_col] + pollutants].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.dropna(subset=[date_col])
    if data.empty:
        raise ValueError('No valid dates in the DataFrame.')

    if normalise:
        for name in pollutants:
            mean = data[name].mean()
            if mean and np.isfinite(mean):
                data[name] = data[name] / mean

    dates = data[date_col].dt
    data['hour'] = dates.hour
    data['weekday'] = pd.Categorical(dates.day_name(),
                                     categories=_WEEKDAY_ORDER, ordered=True)
    data['month'] = dates.month

    rng = np.random.default_rng(random_state)
    colours = colours or pcolors.qualitative.Plotly
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{'colspan': 2}, None], [{}, {}], [{}, {}]],
        subplot_titles=('Hour of day, by weekday', 'Day of week',
                        'Hour of day', 'Month of year', ''),
        vertical_spacing=0.14, horizontal_spacing=0.08,
    )

    records = []
    for i, name in enumerate(pollutants):
        colour = colours[i % len(colours)]
        subset = data.dropna(subset=[name])

        # Panel 1: hour of day within each weekday, laid out as a single axis
        # running 0-167, so Monday 00:00 is 0 and Sunday 23:00 is 167.
        #
        # cat.codes is int8, and Sunday's code of 6 times 24 is 144, past the
        # 127 that int8 holds. Left as int8 it wraps to -112, putting Sunday's
        # block off the left-hand end of the axis. Widen before multiplying.
        weekday_index = subset['weekday'].cat.codes.astype('int64')
        by_day_hour = _summarise(
            subset.assign(_key=weekday_index * 24 + subset['hour']),
            '_key', name, statistic, conf_int, n_boot, rng,
        )
        _add_series(fig, by_day_hour['_key'], by_day_hour, name, colour,
                    1, 1, ci, showlegend=True)
        records.append(by_day_hour.assign(pollutant=name, panel='weekday.hour')
                       .rename(columns={'_key': 'x'}))

        # Panel 2: day of week. Abbreviated on the axis to match panel 1 above
        # and to stop full names being rotated into the panel below; the
        # returned summary keeps the full names.
        by_day = _summarise(subset, 'weekday', name, statistic, conf_int, n_boot, rng)
        _add_series(fig, [str(d)[:3] for d in by_day['weekday']], by_day, name,
                    colour, 2, 1, ci, showlegend=False)
        records.append(by_day.assign(pollutant=name, panel='weekday')
                       .rename(columns={'weekday': 'x'}))

        # Panel 3: hour of day
        by_hour = _summarise(subset, 'hour', name, statistic, conf_int, n_boot, rng)
        _add_series(fig, by_hour['hour'], by_hour, name, colour,
                    2, 2, ci, showlegend=False)
        records.append(by_hour.assign(pollutant=name, panel='hour')
                       .rename(columns={'hour': 'x'}))

        # Panel 4: month of year
        by_month = _summarise(subset, 'month', name, statistic, conf_int, n_boot, rng)
        _add_series(fig, [_MONTH_ABBR[m - 1] for m in by_month['month']], by_month,
                    name, colour, 3, 1, ci, showlegend=False)
        records.append(by_month.assign(pollutant=name, panel='month')
                       .rename(columns={'month': 'x'}))

    # Label panel 1 with weekday names at the midpoint of each 24-hour block.
    # The range is pinned to the full week so the labels stay aligned with the
    # data even if a weekday is entirely missing from the input.
    fig.update_xaxes(
        tickmode='array',
        tickvals=[day * 24 + 12 for day in range(7)],
        ticktext=[d[:3] for d in _WEEKDAY_ORDER],
        range=[-1, 168],
        row=1, col=1,
    )
    for day in range(1, 7):
        fig.add_vline(x=day * 24 - 0.5, line=dict(color='lightgrey', width=1),
                      row=1, col=1)

    fig.update_xaxes(title_text='weekday', row=2, col=1)
    fig.update_xaxes(title_text='hour', row=2, col=2, dtick=6)
    fig.update_xaxes(title_text='month', row=3, col=1)
    axis_title = 'normalised level' if normalise else 'concentration'
    for row, col in [(1, 1), (2, 1), (2, 2), (3, 1)]:
        fig.update_yaxes(title_text=axis_title, row=row, col=col)

    interval_note = (f'{statistic}, {conf_int:.0%} CI' if ci else statistic)
    fig.update_layout(
        title=f'{title} ({interval_note})',
        template='plotly_white', width=width, height=height,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.04,
                    xanchor='center', x=0.5),
    )

    summary = pd.concat(records, ignore_index=True)
    summary = summary[['pollutant', 'panel', 'x', 'value', 'lower', 'upper']]
    return fig, summary
