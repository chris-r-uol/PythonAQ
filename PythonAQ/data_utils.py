"""Data manipulation utilities ported from openair.

These underpin most of the plotting functions: they handle time averaging with
a data-capture threshold, date selection, rolling means and the conditioning
splits (season, weekday, ...) that openair calls ``type``.
"""

import warnings

import numpy as np
import pandas as pd

from .solar import is_daylight

__all__ = [
    'bin_data',
    'calc_percentile',
    'cut_data',
    'date_pad',
    'rolling_mean',
    'select_by_date',
    'select_running',
    'split_by_date',
    'time_average',
]

# openair-style period names mapped to pandas offset aliases.
_AVG_TIME_ALIASES = {
    'sec': 's', 'second': 's',
    'min': 'min', 'minute': 'min',
    'hour': 'h',
    'day': 'D',
    'week': 'W',
    'month': 'MS',
    'quarter': 'QS',
    'season': 'QS-DEC',
    'year': 'YS',
}

_SEASONS_NORTH = {
    12: 'winter (DJF)', 1: 'winter (DJF)', 2: 'winter (DJF)',
    3: 'spring (MAM)', 4: 'spring (MAM)', 5: 'spring (MAM)',
    6: 'summer (JJA)', 7: 'summer (JJA)', 8: 'summer (JJA)',
    9: 'autumn (SON)', 10: 'autumn (SON)', 11: 'autumn (SON)',
}
_SEASON_ORDER_NORTH = ['spring (MAM)', 'summer (JJA)', 'autumn (SON)', 'winter (DJF)']

# In the southern hemisphere the labels shift by six months.
_SEASONS_SOUTH = {
    month: _SEASONS_NORTH[(month + 5) % 12 + 1] for month in range(1, 13)
}
_SEASON_ORDER_SOUTH = _SEASON_ORDER_NORTH

_WEEKDAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                  'Saturday', 'Sunday']
_MONTH_ORDER = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                'August', 'September', 'October', 'November', 'December']


def _parse_avg_time(avg_time):
    """Translate an openair ``avg.time`` string into a pandas offset alias.

    Accepts openair spellings ('day', '3 day', 'month') and passes anything
    else through untouched so native pandas aliases ('D', 'ME', '7D') work too.
    """
    if avg_time is None:
        raise ValueError('avg_time must be provided.')

    text = str(avg_time).strip()
    parts = text.split()
    if len(parts) == 2:
        multiplier, unit = parts
    else:
        multiplier, unit = '', text

    unit_key = unit.lower().rstrip('s')  # tolerate 'days', 'hours'
    if unit_key in _AVG_TIME_ALIASES:
        return f'{multiplier}{_AVG_TIME_ALIASES[unit_key]}'
    return text  # already a pandas alias


def _infer_base_freq(index):
    """Infer the native sampling interval of a DatetimeIndex.

    Falls back to the modal gap between observations, which is robust to the
    gaps that real monitoring data always contains.
    """
    if len(index) < 2:
        return None
    inferred = pd.infer_freq(index)
    if inferred is not None:
        return pd.tseries.frequencies.to_offset(inferred)
    gaps = pd.Series(index).diff().dropna()
    if gaps.empty:
        return None
    modal = gaps.mode()
    if modal.empty:
        return None
    return pd.tseries.frequencies.to_offset(modal.iloc[0])


def time_average(df, avg_time='day', data_thresh=0, statistic='mean',
                 percentile=None, date_col='date_time', vector_ws=False,
                 interval=None):
    """Average a time series over a period, honouring a data-capture threshold.

    Port of openair's ``timeAverage``. Wind direction is averaged as a vector
    rather than a scalar, so that northerly winds either side of 360 degrees do
    not average to south.

    Parameters:
    - df (pd.DataFrame): Input data.
    - avg_time (str): Averaging period, e.g. 'hour', 'day', '3 day', 'month',
      'year', or any pandas offset alias.
    - data_thresh (float): Minimum percentage of data required in a period for
      it to be retained; periods below this become NaN.
    - statistic (str): 'mean', 'median', 'max', 'min', 'sum', 'sd', 'frequency'
      (count of valid values), 'data.cap' (percentage captured) or 'percentile'.
    - percentile (float): Percentile to use when statistic='percentile'.
    - date_col (str): Name of the datetime column.
    - vector_ws (bool): If True, also report the vector (rather than scalar)
      mean wind speed.
    - interval (str or None): The series' true sampling interval, e.g. 'hour'.
      Only used when `data_thresh` is set. Inferred from the data when None.

    Returns:
    - pd.DataFrame: Averaged data with `date_col` as a column.

    Notes:
    - Give `interval` when rows may be absent rather than present-and-NaN.
      Capture is measured against a regular time base, and inferring that base
      from the data uses the most common gap between observations - which is
      wrong precisely when data is missing. An hourly series with every other
      row absent looks like a complete two-hourly series, and reports 100%
      capture on 50% of the data. Stating the interval removes the guess.
      `date_pad` does the same job as a separate step.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in the DataFrame.")
    if not 0 <= data_thresh <= 100:
        raise ValueError('data_thresh must be a percentage between 0 and 100.')
    if statistic == 'percentile' and percentile is None:
        raise ValueError("percentile must be given when statistic='percentile'.")

    freq = _parse_avg_time(avg_time)
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.dropna(subset=[date_col]).sort_values(date_col)
    if data.empty:
        return data.reset_index(drop=True)

    # Resolve wind direction into vector components before averaging.
    has_wd = 'wd' in data.columns
    has_ws = 'ws' in data.columns
    if has_wd:
        radians = np.deg2rad(data['wd'])
        magnitude = data['ws'] if has_ws else 1.0
        data['_u'] = magnitude * np.sin(radians)
        data['_v'] = magnitude * np.cos(radians)

    data = data.set_index(date_col)
    numeric = data.select_dtypes(include=[np.number])

    # Pad onto a regular time base so that absent rows count against capture,
    # exactly as openair's datePad does. `interval` states that base explicitly;
    # inferring it is a guess that fails in the case it most needs to handle.
    base = (pd.tseries.frequencies.to_offset(_parse_avg_time(interval))
            if interval is not None else _infer_base_freq(data.index))
    if base is not None and data_thresh > 0:
        full_index = pd.date_range(data.index.min(), data.index.max(), freq=base)
        numeric = numeric.reindex(numeric.index.union(full_index))

    grouped = numeric.resample(freq)
    aggregators = {
        'mean': lambda g: g.mean(),
        'median': lambda g: g.median(),
        'max': lambda g: g.max(),
        'min': lambda g: g.min(),
        'sum': lambda g: g.sum(min_count=1),
        'sd': lambda g: g.std(),
        'frequency': lambda g: g.count(),
        'data.cap': lambda g: 100.0 * g.count() / g.size().replace(0, np.nan).values[:, None],
        'percentile': lambda g: g.quantile(percentile / 100.0),
    }
    if statistic not in aggregators:
        raise ValueError(
            f"Unknown statistic '{statistic}'. Choose from {sorted(aggregators)}."
        )
    result = aggregators[statistic](grouped)

    # Apply the data-capture threshold.
    if data_thresh > 0 and statistic not in ('frequency', 'data.cap'):
        valid = grouped.count()
        slots = grouped.size()
        capture = 100.0 * valid.div(slots.replace(0, np.nan), axis=0)
        result = result.where(capture >= data_thresh)

    # Recover wind direction (and optionally speed) from the averaged vectors.
    if has_wd and '_u' in result.columns:
        u, v = result['_u'], result['_v']
        result['wd'] = (np.degrees(np.arctan2(u, v)) + 360) % 360
        if has_ws and vector_ws:
            result['ws'] = np.hypot(u, v)
        result = result.drop(columns=['_u', '_v'])

    # Carry through columns that are constant and non-numeric (site, code).
    for column in df.columns:
        if column in result.columns or column == date_col:
            continue
        uniques = df[column].dropna().unique()
        if len(uniques) == 1:
            result[column] = uniques[0]

    return result.reset_index(names=date_col)


def select_by_date(df, start=None, end=None, year=None, month=None, day=None,
                   hour=None, season=None, date_col='date_time',
                   hemisphere='northern'):
    """Subset a DataFrame by date components. Port of openair's ``selectByDate``.

    All supplied criteria are combined with AND. `month` accepts numbers or
    names ('January', 'jan'); `day` accepts weekday names or month-day numbers.

    Parameters:
    - df (pd.DataFrame): Input data.
    - start, end (str or datetime): Inclusive date range bounds.
    - year, month, day, hour (int, str or sequence): Components to keep.
    - season (str or sequence): e.g. 'summer (JJA)', or just 'summer'.
    - date_col (str): Name of the datetime column.
    - hemisphere (str): 'northern' or 'southern', for season definitions.

    Returns:
    - pd.DataFrame: The matching subset.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in the DataFrame.")

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    dates = data[date_col]
    mask = pd.Series(True, index=data.index)

    if start is not None:
        mask &= dates >= pd.to_datetime(start)
    if end is not None:
        # A bare date means the whole of that day.
        end_ts = pd.to_datetime(end)
        if end_ts == end_ts.normalize():
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        mask &= dates <= end_ts

    def _as_sequence(value):
        if value is None:
            return None
        if isinstance(value, (str, int, np.integer)):
            return [value]
        return list(value)

    if (years := _as_sequence(year)) is not None:
        mask &= dates.dt.year.isin([int(y) for y in years])

    if (months := _as_sequence(month)) is not None:
        wanted = set()
        for item in months:
            if isinstance(item, str):
                matches = [
                    i + 1 for i, name in enumerate(_MONTH_ORDER)
                    if name.lower().startswith(item.lower()[:3])
                ]
                if not matches:
                    raise ValueError(f"Unrecognised month '{item}'.")
                wanted.update(matches)
            else:
                wanted.add(int(item))
        mask &= dates.dt.month.isin(wanted)

    if (days := _as_sequence(day)) is not None:
        weekdays, monthdays = set(), set()
        for item in days:
            if isinstance(item, str):
                matches = [
                    name for name in _WEEKDAY_ORDER
                    if name.lower().startswith(item.lower()[:3])
                ]
                if not matches:
                    raise ValueError(f"Unrecognised day '{item}'.")
                weekdays.update(matches)
            else:
                monthdays.add(int(item))
        if weekdays:
            mask &= dates.dt.day_name().isin(weekdays)
        if monthdays:
            mask &= dates.dt.day.isin(monthdays)

    if (hours := _as_sequence(hour)) is not None:
        mask &= dates.dt.hour.isin([int(h) for h in hours])

    if (seasons := _as_sequence(season)) is not None:
        lookup = _SEASONS_SOUTH if hemisphere == 'southern' else _SEASONS_NORTH
        labels = dates.dt.month.map(lookup)
        wanted = set()
        for item in seasons:
            stem = str(item).split(' ')[0].lower()
            wanted.update(
                label for label in lookup.values() if label.startswith(stem)
            )
        if not wanted:
            raise ValueError(f"Unrecognised season {season!r}.")
        mask &= labels.isin(wanted)

    return data[mask].reset_index(drop=True)


def rolling_mean(df, pollutant, width=8, new_name=None, data_thresh=75,
                 align='centre', date_col='date_time'):
    """Rolling mean with a data-capture threshold. Port of ``rollingMean``.

    The default of an 8-point window matches the running 8-hour mean used for
    ozone in most air quality standards.

    Parameters:
    - df (pd.DataFrame): Input data.
    - pollutant (str): Column to smooth.
    - width (int): Window width in observations.
    - new_name (str or None): Output column name; defaults to
      'rolling{width}_{pollutant}'.
    - data_thresh (float): Minimum percentage of valid values in a window.
    - align (str): 'centre'/'center', 'left' or 'right'.
    - date_col (str): Name of the datetime column, used to order the data.

    Returns:
    - pd.DataFrame: Input data with the rolling mean column appended.
    """
    if pollutant not in df.columns:
        raise ValueError(f"Column '{pollutant}' not found in the DataFrame.")
    if align not in ('centre', 'center', 'left', 'right'):
        raise ValueError("align must be one of 'centre', 'left' or 'right'.")
    if not 0 <= data_thresh <= 100:
        raise ValueError('data_thresh must be a percentage between 0 and 100.')

    data = df.copy()
    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.sort_values(date_col).reset_index(drop=True)

    new_name = new_name or f'rolling{width}_{pollutant}'
    min_periods = int(np.ceil(width * data_thresh / 100.0)) or 1

    series = data[pollutant]
    if align in ('centre', 'center'):
        rolled = series.rolling(window=width, center=True, min_periods=min_periods).mean()
    elif align == 'right':
        rolled = series.rolling(window=width, min_periods=min_periods).mean()
    else:  # 'left' - the window looks forward
        rolled = (
            series[::-1].rolling(window=width, min_periods=min_periods).mean()[::-1]
        )

    data[new_name] = rolled
    return data


def cut_data(df, type='season', date_col='date_time', hemisphere='northern',
             n_levels=4, latitude=None, longitude=None):
    """Add a conditioning column for splitting data. Port of ``cutData``.

    Parameters:
    - df (pd.DataFrame): Input data.
    - type (str): One of 'year', 'month', 'monthyear', 'season', 'seasonyear',
      'weekday', 'weekend', 'hour', 'daylight', 'wd', or the name of a numeric
      column (which is split into `n_levels` quantiles).
    - date_col (str): Name of the datetime column.
    - hemisphere (str): 'northern' or 'southern', for season definitions.
    - n_levels (int): Number of quantile levels when splitting a numeric column.
    - latitude, longitude (float): Site position in degrees north and east,
      used by type='daylight' to work out whether the sun was actually up.
      Without them 'daylight' falls back to a fixed 07:00-19:00 window and
      warns, because that window is wrong by hours outside the tropics.

    Returns:
    - pd.DataFrame: Input data with a categorical column named `type` added.
    """
    data = df.copy()

    # 'wd' is checked before the generic numeric branch below: wind direction is
    # numeric, but splitting it into quantiles rather than compass sectors would
    # be meaningless.
    if type == 'wd':
        if 'wd' not in data.columns:
            raise ValueError("A 'wd' column is required for type='wd'.")
        sectors = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        # Shift by half a sector so that north spans 337.5-22.5 degrees.
        shifted = (data['wd'] + 22.5) % 360
        data['wd'] = pd.Categorical(
            pd.cut(shifted, bins=np.arange(0, 361, 45), labels=sectors,
                   include_lowest=True, right=False),
            categories=sectors, ordered=True,
        )
        return data

    # Any other numeric column is split into quantiles rather than by date.
    if type in data.columns and pd.api.types.is_numeric_dtype(data[type]):
        labels = [f'{type} level {i + 1}' for i in range(n_levels)]
        data[type] = pd.qcut(data[type], n_levels, labels=labels, duplicates='drop')
        return data

    if date_col not in data.columns:
        raise ValueError(f"Date column '{date_col}' not found in the DataFrame.")
    dates = pd.to_datetime(data[date_col])

    if type == 'year':
        data[type] = dates.dt.year
    elif type == 'month':
        data[type] = pd.Categorical(dates.dt.month_name(),
                                    categories=_MONTH_ORDER, ordered=True)
    elif type == 'monthyear':
        data[type] = dates.dt.to_period('M').astype(str)
    elif type in ('season', 'seasonyear'):
        lookup = _SEASONS_SOUTH if hemisphere == 'southern' else _SEASONS_NORTH
        order = _SEASON_ORDER_SOUTH if hemisphere == 'southern' else _SEASON_ORDER_NORTH
        labels = dates.dt.month.map(lookup)
        if type == 'seasonyear':
            data[type] = labels + ' ' + dates.dt.year.astype(str)
        else:
            data[type] = pd.Categorical(labels, categories=order, ordered=True)
    elif type == 'weekday':
        data[type] = pd.Categorical(dates.dt.day_name(),
                                    categories=_WEEKDAY_ORDER, ordered=True)
    elif type == 'weekend':
        data[type] = pd.Categorical(
            np.where(dates.dt.dayofweek >= 5, 'weekend', 'weekday'),
            categories=['weekday', 'weekend'], ordered=True,
        )
    elif type == 'hour':
        data[type] = dates.dt.hour
    elif type == 'daylight':
        if latitude is None or longitude is None:
            # A fixed window is not merely approximate away from the equator:
            # at 54 degrees north it mislabels five hours a day in midsummer,
            # and in the opposite direction in midwinter. Anyone comparing
            # daytime with nighttime across seasons would be reading that
            # error rather than their data, so say so rather than imply a
            # precision this branch does not have.
            warnings.warn(
                "cut_data(type='daylight') without latitude and longitude "
                'falls back to a fixed 07:00-19:00 window, which is wrong by '
                'up to several hours a day away from the equator. Pass '
                'latitude= and longitude= for a real sunrise/sunset split.',
                UserWarning, stacklevel=2,
            )
            daylight = ((dates.dt.hour >= 7) & (dates.dt.hour < 19)).to_numpy()
        else:
            daylight = is_daylight(dates, latitude, longitude)
        data[type] = pd.Categorical(
            np.where(daylight, 'daylight', 'nighttime'),
            categories=['daylight', 'nighttime'], ordered=True,
        )
    else:
        raise ValueError(
            f"Unknown type '{type}'. Use a numeric column name or one of: "
            "'year', 'month', 'monthyear', 'season', 'seasonyear', 'weekday', "
            "'weekend', 'hour', 'daylight', 'wd'."
        )
    return data


def calc_percentile(df, pollutant, percentile=(25, 50, 75, 95),
                    avg_time='month', data_thresh=0, date_col='date_time'):
    """Calculate percentiles of a pollutant over a period. Port of ``calcPercentile``.

    Parameters:
    - df (pd.DataFrame): Input data.
    - pollutant (str): Column to summarise.
    - percentile (sequence of float): Percentiles to compute.
    - avg_time (str): Averaging period, as for `time_average`.
    - data_thresh (float): Minimum data capture percentage per period.
    - date_col (str): Name of the datetime column.

    Returns:
    - pd.DataFrame: One column per requested percentile, named 'percentile.N'.
    """
    if pollutant not in df.columns:
        raise ValueError(f"Column '{pollutant}' not found in the DataFrame.")

    percentiles = [percentile] if np.isscalar(percentile) else list(percentile)
    frames = []
    for value in percentiles:
        averaged = time_average(
            df[[date_col, pollutant]], avg_time=avg_time,
            data_thresh=data_thresh, statistic='percentile',
            percentile=value, date_col=date_col,
        )
        frames.append(averaged.set_index(date_col)[pollutant]
                      .rename(f'percentile.{value:g}'))
    return pd.concat(frames, axis=1).reset_index(names=date_col)


def date_pad(df, date_col='date_time', interval=None, type=None,
             hemisphere='northern'):
    """Pad a time series onto a complete, regular time base.

    Port of openair's ``datePad``. Real monitoring data has gaps where rows are
    simply absent rather than present-but-NaN, which quietly distorts anything
    that counts observations per period: a day holding six rows looks fully
    captured if you only count what is there.

    Parameters:
    - df (pd.DataFrame): Input data.
    - date_col (str): Name of the datetime column.
    - interval (str or None): Spacing to pad to, as a pandas offset alias or an
      openair period name. Inferred from the data when None.
    - type (str or None): Column identifying separate series, typically 'site'.
      Each is padded over its own span rather than the whole frame's, so a site
      that started reporting late does not gain years of empty rows.
    - hemisphere (str): Unused; accepted so callers can pass it through.

    Returns:
    - pd.DataFrame: The input with rows inserted for every missing timestamp,
      sorted by date. Inserted rows are NaN except for `date_col` and `type`.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in the DataFrame.")

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.dropna(subset=[date_col]).sort_values(date_col)
    if data.empty:
        return data.reset_index(drop=True)

    if type is not None:
        if type not in data.columns:
            raise ValueError(f"Column '{type}' not found in the DataFrame.")
        padded = [
            date_pad(group, date_col=date_col, interval=interval).assign(**{type: key})
            for key, group in data.groupby(type, observed=True, sort=True)
        ]
        return pd.concat(padded, ignore_index=True).sort_values(
            [type, date_col]
        ).reset_index(drop=True)

    if interval is None:
        offset = _infer_base_freq(pd.DatetimeIndex(data[date_col]))
        if offset is None:
            return data.reset_index(drop=True)
    else:
        offset = pd.tseries.frequencies.to_offset(_parse_avg_time(interval))

    full = pd.date_range(data[date_col].min(), data[date_col].max(), freq=offset)
    return (
        data.set_index(date_col)
        .reindex(data.set_index(date_col).index.union(full))
        .rename_axis(date_col)
        .reset_index()
    )


def split_by_date(df, dates, labels=None, date_col='date_time', name='split_by'):
    """Split a series at given dates. Port of openair's ``splitByDate``.

    Parameters:
    - df (pd.DataFrame): Input data.
    - dates (str or sequence): One or more cut points. N cut points give N+1
      periods.
    - labels (sequence or None): Names for the periods. Generated from the cut
      points when None.
    - date_col (str): Name of the datetime column.
    - name (str): Name of the column added.

    Returns:
    - pd.DataFrame: The input with an ordered categorical column added.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in the DataFrame.")

    cuts = [pd.to_datetime(dates)] if isinstance(dates, str) else \
        [pd.to_datetime(d) for d in dates]
    cuts = sorted(cuts)
    if not cuts:
        raise ValueError('At least one split date is required.')

    if labels is None:
        labels = [f'before {cuts[0]:%d %b %Y}']
        labels += [f'{a:%d %b %Y} to {b:%d %b %Y}'
                   for a, b in zip(cuts[:-1], cuts[1:])]
        labels.append(f'after {cuts[-1]:%d %b %Y}')
    elif len(labels) != len(cuts) + 1:
        raise ValueError(
            f'{len(cuts)} split date(s) need {len(cuts) + 1} labels, '
            f'got {len(labels)}.'
        )

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    edges = [pd.Timestamp.min] + cuts + [pd.Timestamp.max]
    data[name] = pd.cut(data[date_col], bins=edges, labels=labels,
                        right=False, ordered=True)
    return data


def select_running(df, pollutant, run_length=5, threshold=None,
                   date_col='date_time', name='criterion', mode='flag'):
    """Find runs of consecutive values at or above a threshold.

    Port of openair's ``selectRunning``. Useful for isolating pollution
    episodes, which are defined by persistence rather than by any single high
    hour: one spike is not an episode, ten consecutive hours is.

    Parameters:
    - df (pd.DataFrame): Input data.
    - pollutant (str): Column to test.
    - run_length (int): Minimum number of consecutive observations.
    - threshold (float or None): Value to test against. Defaults to the
      pollutant's own 95th percentile.
    - date_col (str): Datetime column, used to order the data.
    - name (str): Name of the flag column added.
    - mode (str): 'flag' adds a yes/no column; 'filter' returns only the rows
      belonging to a qualifying run.

    Returns:
    - pd.DataFrame: Flagged or filtered data.
    """
    if pollutant not in df.columns:
        raise ValueError(f"Column '{pollutant}' not found in the DataFrame.")
    if mode not in ('flag', 'filter'):
        raise ValueError("mode must be 'flag' or 'filter'.")
    if run_length < 1:
        raise ValueError('run_length must be at least 1.')

    data = df.copy()
    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.sort_values(date_col).reset_index(drop=True)

    if threshold is None:
        threshold = float(data[pollutant].quantile(0.95))

    above = (data[pollutant] >= threshold).fillna(False)
    # Number each unbroken stretch, then keep only those long enough. NaN counts
    # as below, so a gap ends a run rather than silently bridging it.
    block = (above != above.shift()).cumsum()
    lengths = above.groupby(block).transform('size')
    qualifies = above & (lengths >= run_length)

    data[name] = np.where(qualifies, 'yes', 'no')
    if mode == 'filter':
        return data[qualifies].reset_index(drop=True)
    return data


def bin_data(df, x, y, bins=20, statistic='mean', conf_int=0.95, n_boot=200,
             random_state=None):
    """Bin one variable against another, with bootstrap intervals on each bin.

    Port of openair's ``binData``. Answers "how does y behave across the range
    of x?" with an honest uncertainty on each bin, rather than a single
    regression line that hides where the data actually is.

    Parameters:
    - df (pd.DataFrame): Input data.
    - x (str): Column to bin along.
    - y (str): Column to summarise within each bin.
    - bins (int or sequence): Number of equal-width bins, or explicit edges.
    - statistic (str): 'mean' or 'median'.
    - conf_int (float): Confidence level for the interval.
    - n_boot (int): Bootstrap replicates.
    - random_state (int or None): Seed, for reproducible intervals.

    Returns:
    - pd.DataFrame: One row per bin with the bin centre, count, the statistic
      and its lower and upper bounds.
    """
    for column in (x, y):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
    if statistic not in ('mean', 'median'):
        raise ValueError("statistic must be 'mean' or 'median'.")

    data = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        raise ValueError(f"No complete '{x}'/'{y}' pairs to bin.")

    edges = (np.linspace(data[x].min(), data[x].max(), int(bins) + 1)
             if np.isscalar(bins) else np.asarray(bins, dtype=float))
    assigned = pd.cut(data[x], bins=edges, include_lowest=True, labels=False)

    rng = np.random.default_rng(random_state)
    tail = (1.0 - conf_int) / 2.0
    rows = []
    for index in range(len(edges) - 1):
        values = data.loc[assigned == index, y].to_numpy(dtype=float)
        centre = (edges[index] + edges[index + 1]) / 2.0
        if values.size == 0:
            rows.append({x: centre, 'n': 0, statistic: np.nan,
                         'lower': np.nan, 'upper': np.nan})
            continue

        estimate = (np.median(values) if statistic == 'median'
                    else np.mean(values))
        if values.size == 1:
            lower = upper = estimate
        else:
            draws = values[rng.integers(0, values.size, (n_boot, values.size))]
            replicates = (np.median(draws, axis=1) if statistic == 'median'
                          else np.mean(draws, axis=1))
            lower, upper = np.quantile(replicates, [tail, 1.0 - tail])
        rows.append({x: centre, 'n': int(values.size), statistic: estimate,
                     'lower': float(lower), 'upper': float(upper)})

    return pd.DataFrame(rows)
