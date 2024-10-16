import numpy as np
import logging

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
    try:
        T_obs = np.asarray(T_obs, dtype=np.float64)
        # Handle invalid temperatures
        invalid_temp = (T_obs < -45) | (T_obs > 60) | np.isnan(T_obs)
        # exponent = (17.67 * T_obs) / (T_obs + 243.5) # original values, updating to new ones
        exponent = (17.625 * T_obs) / (T_obs + 243.04) # constants from https://www.omnicalculator.com/physics/relative-humidity
        e_sat_values = 6.112 * np.exp(exponent)
        # Set e_sat to NaN for invalid temperatures
        e_sat_values[invalid_temp] = np.nan
        return e_sat_values
    except Exception as e:
        logging.error(f"Error in e_sat calculation: {e}")
        return np.full_like(T_obs, np.nan)

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
        Relative humidity as a percentage.
    """
    try:
        T = np.asarray(T, dtype=np.float64)
        T_d = np.asarray(T_d, dtype=np.float64) / 100
        # Ensure T and T_d have the same shape
        if T.shape != T_d.shape:
            raise ValueError("Air temperature and dew point must have the same shape.")
        e_t = e_sat(T)
        e_d = e_sat(T_d)
        with np.errstate(divide='ignore', invalid='ignore'):
            rh_values = (e_d / e_t) * 100
            rh_values = np.where(np.isfinite(rh_values), rh_values, np.nan)
        return rh_values
    except Exception as e:
        logging.error(f"Error in RH calculation: {e}")
        return np.full_like(T, np.nan)
