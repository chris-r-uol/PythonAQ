"""Smoothers ported from openair.

Four ways to separate signal from noise in a series, each with a different
notion of what counts as signal:

- `rolling_quantile` tracks a percentile rather than the mean, so a rising
  floor is distinguishable from a few large episodes.
- `kz_filter` is the iterated moving average used in trend work, whose
  repeated passes give it a far sharper frequency cutoff than one long window.
- `whittaker_smooth` balances fidelity against roughness globally, so it
  bridges gaps instead of leaving holes.
- `gaussian_smooth` is the plain weighted-kernel smoother, on a series or a
  grid.

All of them follow the same convention as `rolling_mean`: they take a
DataFrame and a column and return the DataFrame with a new column appended.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

__all__ = ['gaussian_smooth', 'kz_filter', 'rolling_quantile',
           'whittaker_smooth']


def _ordered(df, date_col):
    """Sort by time if there is a time column, as the smoothers assume order."""
    data = df.copy()
    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.sort_values(date_col).reset_index(drop=True)
    return data


def rolling_quantile(df, pollutant, width=8, quantile=0.5, new_name=None,
                     data_thresh=75, align='centre', date_col='date_time'):
    """Rolling quantile with a data-capture threshold. Port of ``rollingQuantile``.

    Parameters:
    - df (pd.DataFrame): Input data.
    - pollutant (str): Column to smooth.
    - width (int): Window width in observations.
    - quantile (float): Quantile to track, between 0 and 1. 0.5 is the median.
    - new_name (str or None): Output column; defaults to
      'rolling{width}_q{quantile}_{pollutant}'.
    - data_thresh (float): Minimum percentage of valid values in a window.
    - align (str): 'centre'/'center', 'left' or 'right'.
    - date_col (str): Datetime column, used to order the data.

    Returns:
    - pd.DataFrame: Input data with the rolling quantile appended.

    Notes:
    - A rolling median is the usual reason to reach for this: unlike a mean it
      is unmoved by a single extreme hour, so it separates a shift in the bulk
      of the distribution from a handful of episodes.
    """
    if pollutant not in df.columns:
        raise ValueError(f"Column '{pollutant}' not found in the DataFrame.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError('quantile must be between 0 and 1.')
    if align not in ('centre', 'center', 'left', 'right'):
        raise ValueError("align must be one of 'centre', 'left' or 'right'.")
    if not 0 <= data_thresh <= 100:
        raise ValueError('data_thresh must be a percentage between 0 and 100.')
    if width < 1:
        raise ValueError('width must be at least 1.')

    data = _ordered(df, date_col)
    new_name = new_name or f'rolling{width}_q{quantile:g}_{pollutant}'
    minimum = int(np.ceil(width * data_thresh / 100.0)) or 1

    series = data[pollutant]
    centre = align in ('centre', 'center')
    if align == 'left':
        # A left-aligned window looks forward, which pandas expresses by
        # reversing the series rather than with a parameter.
        rolled = (series[::-1].rolling(width, min_periods=minimum)
                  .quantile(quantile)[::-1])
    else:
        rolled = series.rolling(width, min_periods=minimum,
                                center=centre).quantile(quantile)

    data[new_name] = rolled
    return data


def kz_filter(df, pollutant, width=15, iterations=5, new_name=None,
              date_col='date_time'):
    """Kolmogorov-Zurbenko low-pass filter. Port of ``kzFilter``.

    A moving average applied `iterations` times over a window of `width`. Each
    pass is cheap, and repeating one sharpens the frequency cutoff far more
    than widening a single pass would: KZ(15, 5) suppresses high frequencies
    much more cleanly than one 75-point mean, while following a slow trend just
    as closely. This is why it is the usual first step in separating a
    long-term trend from weather-driven variation.

    Parameters:
    - df (pd.DataFrame): Input data.
    - pollutant (str): Column to filter.
    - width (int): Window width in observations, per pass.
    - iterations (int): Number of passes.
    - new_name (str or None): Output column; defaults to
      'kz{width}_{iterations}_{pollutant}'.
    - date_col (str): Datetime column, used to order the data.

    Returns:
    - pd.DataFrame: Input data with the filtered series appended.

    Notes:
    - The effective width after k passes is about `width * sqrt(k)`.
    - Gaps are bridged rather than propagated: each pass ignores missing
      values, so an outage narrows the window instead of erasing a window's
      worth of output. The filtered series therefore has no missing values
      even where the input did, and across a long gap it is an interpolation.
    - For the same reason the first and last points are real numbers rather
      than NaN, but they are computed from a partial window - roughly half of
      one - and are the least reliable points in the series. Trim them before
      reading a trend off the ends.
    """
    if pollutant not in df.columns:
        raise ValueError(f"Column '{pollutant}' not found in the DataFrame.")
    if width < 1:
        raise ValueError('width must be at least 1.')
    if iterations < 1:
        raise ValueError('iterations must be at least 1.')

    data = _ordered(df, date_col)
    new_name = new_name or f'kz{width}_{iterations}_{pollutant}'

    smoothed = data[pollutant].astype(float)
    for _ in range(iterations):
        smoothed = smoothed.rolling(width, min_periods=1, center=True).mean()
    data[new_name] = smoothed
    return data


def whittaker_smooth(df, pollutant, lam=1600.0, order=2, new_name=None,
                     date_col='date_time'):
    """Whittaker-Eilers penalised least squares smoother. Port of ``WhittakerSmooth``.

    Finds the series z that minimises the distance to the data plus `lam` times
    the roughness of z. Unlike a moving average this is a single global fit, so
    it does not shorten the series at the ends and it bridges gaps rather than
    leaving them.

    Parameters:
    - df (pd.DataFrame): Input data.
    - pollutant (str): Column to smooth.
    - lam (float): Smoothing weight. Larger is smoother; the scale is roughly
      logarithmic, so try factors of ten rather than small adjustments.
    - order (int): Order of the difference penalty. 2 penalises curvature and
      tends towards a straight line; 1 penalises slope and tends towards a
      constant.
    - new_name (str or None): Output column; defaults to
      'whittaker_{pollutant}'.
    - date_col (str): Datetime column, used to order the data.

    Returns:
    - pd.DataFrame: Input data with the smoothed series appended.

    Notes:
    - Missing values are given zero weight rather than being dropped, so the
      smoother interpolates across them under the same roughness penalty as
      everywhere else. Across a long gap that interpolation is an extrapolation
      of the penalty, not evidence, and should be treated as such.
    - The series is assumed to be evenly spaced. It is ordered by `date_col`
      if present, but a gap in the timestamps is not the same as a gap in the
      rows: pad the series first if that distinction matters.
    """
    if pollutant not in df.columns:
        raise ValueError(f"Column '{pollutant}' not found in the DataFrame.")
    if lam <= 0:
        raise ValueError('lam must be positive.')
    if order < 1:
        raise ValueError('order must be at least 1.')

    data = _ordered(df, date_col)
    new_name = new_name or f'whittaker_{pollutant}'

    values = data[pollutant].to_numpy(dtype=float)
    n = len(values)
    if n <= order:
        raise ValueError(
            f'Need more than {order} rows to apply an order-{order} penalty.')

    observed = np.isfinite(values)
    if not observed.any():
        data[new_name] = np.nan
        return data

    weights = observed.astype(float)
    y = np.where(observed, values, 0.0)

    # Sparse difference operator of the requested order.
    differences = sparse.eye(n, format='csc')
    for _ in range(order):
        differences = differences[1:] - differences[:-1]

    lhs = sparse.diags(weights) + lam * (differences.T @ differences)
    data[new_name] = spsolve(sparse.csc_matrix(lhs), weights * y)
    return data


def gaussian_smooth(df, pollutant, sigma=2.0, truncate=4.0, new_name=None,
                    date_col='date_time'):
    """Gaussian kernel smoother. Port of ``GaussianSmooth``.

    Parameters:
    - df (pd.DataFrame): Input data.
    - pollutant (str): Column to smooth.
    - sigma (float): Kernel standard deviation, in observations.
    - truncate (float): Kernel half-width in standard deviations.
    - new_name (str or None): Output column; defaults to 'gaussian_{pollutant}'.
    - date_col (str): Datetime column, used to order the data.

    Returns:
    - pd.DataFrame: Input data with the smoothed series appended.

    Notes:
    - Missing values are excluded from each kernel and the weights
      renormalised, rather than being treated as zero. Treating them as zero
      would pull the smoothed series towards zero near every gap, which looks
      like a real dip.
    """
    if pollutant not in df.columns:
        raise ValueError(f"Column '{pollutant}' not found in the DataFrame.")
    if sigma <= 0:
        raise ValueError('sigma must be positive.')
    if truncate <= 0:
        raise ValueError('truncate must be positive.')

    data = _ordered(df, date_col)
    new_name = new_name or f'gaussian_{pollutant}'

    values = data[pollutant].to_numpy(dtype=float)
    observed = np.isfinite(values)

    radius = int(truncate * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()

    filled = np.where(observed, values, 0.0)
    # Convolving the mask alongside the data gives the weight actually
    # available at each point, which is what the result must be divided by.
    numerator = np.convolve(filled, kernel, mode='same')
    denominator = np.convolve(observed.astype(float), kernel, mode='same')

    with np.errstate(invalid='ignore', divide='ignore'):
        smoothed = np.where(denominator > 0, numerator / denominator, np.nan)
    data[new_name] = np.where(observed, smoothed, np.nan)
    return data
