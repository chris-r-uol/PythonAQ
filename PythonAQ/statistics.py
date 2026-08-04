"""Model evaluation and summary statistics.

Ports of openair's ``modStats`` and ``aqStats``. The modStats formulas follow
the openair R source exactly, including the piecewise definition of IOA
(Willmott et al., 2011) and the use of sums rather than means for the
normalised statistics.
"""

import numpy as np
import pandas as pd
from scipy import stats as _stats

from .data_utils import rolling_mean, time_average

__all__ = ['mod_stats', 'aq_stats']


def mod_stats(df, mod='mod', obs='obs', group_by=None):
    """Common model evaluation statistics. Port of openair's ``modStats``.

    Parameters:
    - df (pd.DataFrame): Data containing the modelled and observed columns.
    - mod (str): Modelled values column.
    - obs (str): Observed values column.
    - group_by (str or list or None): Column(s) to compute statistics within.

    Returns:
    - pd.DataFrame: One row per group with columns n, FAC2, MB, MGE, NMB,
      NMGE, RMSE, r, P, COE and IOA.

    Notes:
    - FAC2 is the fraction of predictions within a factor of two of the
      observations, i.e. 0.5 <= mod/obs <= 2.
    - COE is 1 when the model is perfect, 0 when it is no better than the
      observed mean, and negative when it is worse.
    - IOA spans -1 to +1 and is piecewise, per Willmott et al. (2011).
    """
    for column in (mod, obs):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

    if group_by is not None:
        keys = [group_by] if isinstance(group_by, str) else list(group_by)
        missing = [k for k in keys if k not in df.columns]
        if missing:
            raise ValueError(f"Grouping column(s) not found: {missing}")
        frames = []
        for key, group in df.groupby(keys, observed=True, sort=True):
            row = mod_stats(group, mod=mod, obs=obs)
            for name, value in zip(keys, key if isinstance(key, tuple) else (key,)):
                row.insert(0, name, value)
            frames.append(row)
        return pd.concat(frames, ignore_index=True)

    pair = df[[mod, obs]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(pair)
    if n == 0:
        return pd.DataFrame([{
            'n': 0, 'FAC2': np.nan, 'MB': np.nan, 'MGE': np.nan, 'NMB': np.nan,
            'NMGE': np.nan, 'RMSE': np.nan, 'r': np.nan, 'P': np.nan,
            'COE': np.nan, 'IOA': np.nan,
        }])

    m = pair[mod].to_numpy(dtype=float)
    o = pair[obs].to_numpy(dtype=float)
    residual = m - o
    abs_residual = np.abs(residual)

    # FAC2: ratios are undefined where obs == 0, matching R's na.omit of Inf/NaN.
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = m / o
    ratio = ratio[np.isfinite(ratio)]
    fac2 = float(np.mean((ratio >= 0.5) & (ratio <= 2))) if ratio.size else np.nan

    sum_obs = o.sum()
    obs_deviation = np.abs(o - o.mean()).sum()

    if n > 1 and np.std(m) > 0 and np.std(o) > 0:
        r_value, p_value = _stats.pearsonr(m, o)
    else:
        r_value, p_value = np.nan, np.nan

    # IOA is piecewise about the point where the error equals twice the
    # observed variability.
    lhs = abs_residual.sum()
    rhs = 2.0 * obs_deviation
    if rhs == 0:
        ioa = np.nan
    elif lhs <= rhs:
        ioa = 1.0 - lhs / rhs
    else:
        ioa = rhs / lhs - 1.0

    return pd.DataFrame([{
        'n': n,
        'FAC2': fac2,
        'MB': float(residual.mean()),
        'MGE': float(abs_residual.mean()),
        'NMB': float(residual.sum() / sum_obs) if sum_obs else np.nan,
        'NMGE': float(lhs / sum_obs) if sum_obs else np.nan,
        'RMSE': float(np.sqrt(np.mean(residual ** 2))),
        'r': float(r_value) if np.isfinite(r_value) else np.nan,
        'P': float(p_value) if np.isfinite(p_value) else np.nan,
        'COE': float(1.0 - lhs / obs_deviation) if obs_deviation else np.nan,
        'IOA': float(ioa) if np.isfinite(ioa) else np.nan,
    }])


def aq_stats(df, pollutant, date_col='date_time', data_thresh=0,
             percentile=(95,), transpose=False):
    """Annual air quality summary statistics. Port of openair's ``aqStats``.

    Parameters:
    - df (pd.DataFrame): Input data, at hourly resolution.
    - pollutant (str): Column to summarise.
    - date_col (str): Name of the datetime column.
    - data_thresh (float): Minimum data capture percentage per year.
    - percentile (sequence of float): Additional percentiles to report.
    - transpose (bool): Return statistics as rows rather than columns.

    Returns:
    - pd.DataFrame: One row per year with the data capture percentage, mean,
      minimum, maximum, median, requested percentiles, the maximum daily and
      rolling 8-hour means, and exceedance counts against the UK objectives
      for the pollutant where one is defined.
    """
    if pollutant not in df.columns:
        raise ValueError(f"Column '{pollutant}' not found in the DataFrame.")
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in the DataFrame.")

    # UK air quality objective thresholds, in the usual reporting units.
    thresholds = {
        'o3': ('rolling_8_hour', 100.0),
        'no2': ('hourly', 200.0),
        'so2': ('daily', 125.0),
        'pm10': ('daily', 50.0),
        'co': ('rolling_8_hour', 10.0),
    }

    data = df[[date_col, pollutant]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.dropna(subset=[date_col]).sort_values(date_col)
    if data.empty:
        return pd.DataFrame()

    percentiles = list(percentile) if not np.isscalar(percentile) else [percentile]
    with_rolling = rolling_mean(data, pollutant, width=8, data_thresh=data_thresh or 75,
                                date_col=date_col)
    rolling_col = f'rolling8_{pollutant}'
    daily = time_average(data, avg_time='day', data_thresh=data_thresh,
                         date_col=date_col)

    rows = []
    for year, group in data.groupby(data[date_col].dt.year):
        values = group[pollutant]
        expected = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='h')
        capture = 100.0 * values.notna().sum() / len(expected)

        row = {
            'year': int(year),
            'data_capture': round(float(capture), 1),
            'mean': values.mean(),
            'minimum': values.min(),
            'maximum': values.max(),
            'median': values.median(),
        }
        for p in percentiles:
            row[f'percentile.{p:g}'] = values.quantile(p / 100.0)

        year_daily = daily[daily[date_col].dt.year == year][pollutant]
        year_rolling = with_rolling[with_rolling[date_col].dt.year == year][rolling_col]
        row['max_daily'] = year_daily.max() if len(year_daily) else np.nan
        row['max_rolling_8'] = year_rolling.max() if len(year_rolling) else np.nan

        entry = thresholds.get(pollutant.lower())
        if entry is not None:
            basis, limit = entry
            if basis == 'hourly':
                exceedances = int((values > limit).sum())
            elif basis == 'daily':
                exceedances = int((year_daily > limit).sum())
            else:
                exceedances = int((year_rolling > limit).sum())
            row[f'days_{basis}_gt_{limit:g}'] = exceedances

        if data_thresh and capture < data_thresh:
            for key in list(row):
                if key not in ('year', 'data_capture'):
                    row[key] = np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary.set_index('year').T if transpose else summary
