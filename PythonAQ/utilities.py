import numpy as np
from pandas.tseries.frequencies import to_offset

_DAYS_PER_YEAR = 365.25

# Number of periods in one year, keyed by the *base* pandas offset name with
# any anchor suffix stripped. Covers both the pre- and post-pandas-2.2 spellings.
_PERIODS_PER_YEAR = {
    'A': 1, 'AS': 1, 'Y': 1, 'YS': 1, 'YE': 1,          # yearly
    'Q': 4, 'QS': 4, 'QE': 4,                            # quarterly
    'M': 12, 'MS': 12, 'ME': 12,                         # monthly
    'W': _DAYS_PER_YEAR / 7,                             # weekly
    'D': _DAYS_PER_YEAR,                                 # daily
    'H': _DAYS_PER_YEAR * 24, 'h': _DAYS_PER_YEAR * 24,  # hourly
    'T': _DAYS_PER_YEAR * 24 * 60,                       # minutely
    'min': _DAYS_PER_YEAR * 24 * 60,
    'S': _DAYS_PER_YEAR * 24 * 3600,                     # secondly
    's': _DAYS_PER_YEAR * 24 * 3600,
}

def e_sat(T_obs):
    """
    Calculate the saturation vapor pressure (e_sat) in millibars for a given temperature
    using the Magnus formula.

    Parameters:
    ----------
    T_obs : float or array-like
        Temperature in degrees Celsius. Valid for -45°C ≤ T_obs ≤ 60°C.

    Returns:
    -------
    e_sat : float or ndarray
        Saturation vapor pressure in millibars.
    """
    T_obs = np.asarray(T_obs, dtype=np.float64)
    # Handle invalid temperatures
    with np.errstate(invalid='ignore'):
        invalid_temp = (T_obs < -45) | (T_obs > 60) | np.isnan(T_obs)
        # exponent = (17.67 * T_obs) / (T_obs + 243.5) # original values, updating to new ones
        exponent = (17.625 * T_obs) / (T_obs + 243.04) # constants from https://www.omnicalculator.com/physics/relative-humidity
        e_sat_values = 6.112 * np.exp(exponent)
    # np.where keeps this working for scalar (0-d) input, which fancy-index
    # assignment does not support.
    return np.where(invalid_temp, np.nan, e_sat_values)

def rh(T, T_d):
    """
    Calculate the relative humidity based on air temperature and dew point.

    Parameters:
    ----------
    T : float or array-like
        Air temperature in degrees Celsius.
    T_d : float or array-like
        Dew point temperature in degrees Celsius.

    Returns:
    -------
    rh : float or ndarray
        Relative humidity as a percentage, clipped to [0, 100].
    """
    T = np.asarray(T, dtype=np.float64)
    T_d = np.asarray(T_d, dtype=np.float64)
    # Ensure T and T_d are broadcast-compatible
    if T.shape != T_d.shape:
        raise ValueError("Air temperature and dew point must have the same shape.")
    e_t = e_sat(T)
    e_d = e_sat(T_d)
    with np.errstate(divide='ignore', invalid='ignore'):
        rh_values = (e_d / e_t) * 100
        rh_values = np.where(np.isfinite(rh_values), rh_values, np.nan)
        # Dew point cannot exceed air temperature; small measurement errors
        # can push the ratio marginally above 1.
        rh_values = np.clip(rh_values, 0.0, 100.0)
    return rh_values

def get_period(interval):
    """
    Calculate the period for seasonal decomposition based on the resampling interval.

    Parameters:
    - interval (str): Resampling interval string (e.g., '7D', 'M', 'Q').

    Returns:
    - int: Number of periods in one seasonal cycle (e.g., a year).
    """
    if interval is None or interval.strip() == '':
        # Default period for daily data
        return 365

    offset = to_offset(interval)
    n = offset.n
    # Offset names carry an anchor suffix (e.g. 'W-SUN', 'QE-DEC') and differ
    # between pandas versions ('M'/'ME', 'H'/'h', 'T'/'min', 'A'/'Y'/'YE').
    base_freq = offset.name.split('-')[0]

    if base_freq in _PERIODS_PER_YEAR:
        periods_per_year = _PERIODS_PER_YEAR[base_freq] / n
    else:
        raise ValueError(
            f"Unsupported interval '{interval}' (base frequency '{base_freq}')."
        )

    period = int(round(periods_per_year))
    if period < 1:
        raise ValueError(
            f"Interval '{interval}' is longer than one year, so it has no "
            f"seasonal cycle to decompose."
        )
    return period





