"""Data manipulation utilities ported from openair.

These underpin most of the plotting functions: they handle time averaging with
a data-capture threshold, date selection, rolling means and the conditioning
splits (season, weekday, ...) that openair calls ``type``.
"""

import numpy as np
import pandas as pd

__all__ = [
    'calc_percentile',
    'cut_data',
    'rolling_mean',
    'select_by_date',
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
                 percentile=None, date_col='date_time', vector_ws=False):
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

    Returns:
    - pd.DataFrame: Averaged data with `date_col` as a column.
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
    # exactly as openair's datePad does.
    base = _infer_base_freq(data.index)
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
    - latitude, longitude (float): Reserved for a future solar-position based
      'daylight' split; currently a fixed ?? hour approximation is used.

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
        # Approximation: openair uses solar elevation. Without a location we
        # fall back to a fixed window, which is adequate for coarse splits.
        hours = dates.dt.hour
        data[type] = pd.Categorical(
            np.where((hours >= 7) & (hours < 19), 'daylight', 'nighttime'),
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
