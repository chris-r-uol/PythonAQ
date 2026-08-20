"""Solar position, for splitting data by daylight.

openair's ``cutData(type = "daylight")`` asks whether the sun is above the
horizon, which depends on where and when you are. A fixed clock window is not a
usable substitute outside the tropics: at 54 degrees north the sun rises before
04:00 in June and after 08:00 in December, so a 07:00-19:00 rule mislabels
several hours a day in opposite directions through the year. Anything comparing
daytime with nighttime chemistry across seasons would be comparing the error.

The NOAA solar calculator equations are used. Their geometric elevation
matches ``astral`` exactly (to five decimal places, checked from Tromso to
Ushuaia); the cheaper truncated series often quoted for this job drifts by up
to a degree, which is wider than the sun itself and enough to put roughly fifty
boundary hours a year on the wrong side of sunrise.
"""

import numpy as np
import pandas as pd

__all__ = ['solar_elevation', 'is_daylight']


def solar_elevation(dates, latitude, longitude):
    """Solar elevation above the horizon, in degrees.

    Parameters:
    - dates (pd.Series or DatetimeIndex): Timestamps. Timezone-aware input is
      converted to UTC; naive input is *assumed* to be UTC, which is what the
      UK networks and NOAA ISD both publish.
    - latitude (float): Degrees north, negative for the southern hemisphere.
    - longitude (float): Degrees east, negative for the western hemisphere.

    Returns:
    - np.ndarray: Elevation in degrees. Negative means the sun is below the
      horizon. Refraction is not applied, so values near zero are the
      geometric position rather than the visible one: published sunrise times
      correspond to a geometric elevation of about -0.833 degrees, because the
      atmosphere lifts the image of the sun and the upper limb clears the
      horizon before the centre does.

    Notes:
    - Feeding local clock time to this function silently shifts the result by
      the UTC offset, which in British Summer Time is a whole hour and enough
      to move the boundary across a data point.
    """
    stamps = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).to_numpy()))
    if stamps.tz is not None:
        stamps = stamps.tz_convert('UTC').tz_localize(None)

    if not np.isfinite(latitude) or not np.isfinite(longitude):
        raise ValueError('latitude and longitude must be finite numbers.')
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f'latitude {latitude} is outside -90 to 90 degrees.')
    if not -180.0 <= longitude <= 360.0:
        raise ValueError(f'longitude {longitude} is outside -180 to 360 degrees.')

    # Julian day, then Julian century from the J2000.0 epoch. Working from an
    # absolute day number rather than day-of-year avoids the leap-year fudge
    # the truncated series needs.
    #
    # Divided as a Timedelta rather than off the integer representation: the
    # underlying unit is nanoseconds on pandas 2 and microseconds on pandas 3,
    # and reading it directly is silently wrong by a factor of a thousand on
    # one of them.
    days_since_epoch = ((stamps - pd.Timestamp('1970-01-01'))
                        / pd.Timedelta(days=1)).to_numpy(dtype=float)
    julian_day = days_since_epoch + 2_440_587.5
    century = (julian_day - 2_451_545.0) / 36_525.0

    # Geometric mean longitude and anomaly of the sun, degrees.
    mean_long = (280.46646 + century * (36000.76983 + century * 0.0003032)) % 360.0
    mean_anom = 357.52911 + century * (35999.05029 - 0.0001537 * century)
    eccentricity = 0.016708634 - century * (0.000042037 + 0.0000001267 * century)

    # Equation of the centre: the correction from the fictitious mean sun to
    # the real one, which is what makes the orbit elliptical rather than round.
    anom_rad = np.radians(mean_anom)
    centre = (np.sin(anom_rad) * (1.914602 - century * (0.004817 + 0.000014 * century))
              + np.sin(2 * anom_rad) * (0.019993 - 0.000101 * century)
              + np.sin(3 * anom_rad) * 0.000289)

    # Apparent longitude, corrected for nutation and aberration.
    omega = np.radians(125.04 - 1934.136 * century)
    apparent_long = np.radians(mean_long + centre - 0.00569 - 0.00478 * np.sin(omega))

    # Obliquity of the ecliptic: the axial tilt, which is what gives seasons.
    mean_obliquity = (23.0 + (26.0 + (21.448 - century * (
        46.815 + century * (0.00059 - century * 0.001813))) / 60.0) / 60.0)
    obliquity = np.radians(mean_obliquity + 0.00256 * np.cos(omega))

    declination = np.arcsin(np.sin(obliquity) * np.sin(apparent_long))

    # Equation of time, minutes.
    vary = np.tan(obliquity / 2.0) ** 2
    long_rad = np.radians(mean_long)
    eq_time = 4.0 * np.degrees(
        vary * np.sin(2 * long_rad)
        - 2.0 * eccentricity * np.sin(anom_rad)
        + 4.0 * eccentricity * vary * np.sin(anom_rad) * np.cos(2 * long_rad)
        - 0.5 * vary * vary * np.sin(4 * long_rad)
        - 1.25 * eccentricity * eccentricity * np.sin(2 * anom_rad)
    )

    hour = (stamps.hour.to_numpy(dtype=float)
            + stamps.minute.to_numpy(dtype=float) / 60.0
            + stamps.second.to_numpy(dtype=float) / 3600.0)

    # True solar time, minutes since local solar midnight. Four minutes per
    # degree of longitude is the Earth turning.
    true_solar = (hour * 60.0 + eq_time + 4.0 * longitude) % 1440.0
    hour_angle = np.radians(true_solar / 4.0 - 180.0)

    lat = np.radians(latitude)
    cos_zenith = (np.sin(lat) * np.sin(declination)
                  + np.cos(lat) * np.cos(declination) * np.cos(hour_angle))
    return 90.0 - np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))


def is_daylight(dates, latitude, longitude, threshold=0.0):
    """True where the sun is above `threshold` degrees of elevation.

    Parameters:
    - dates, latitude, longitude: As for `solar_elevation`.
    - threshold (float): Elevation counted as daylight. Zero is the geometric
      horizon. Use -6 for civil twilight.

    Returns:
    - np.ndarray of bool.
    """
    return solar_elevation(dates, latitude, longitude) > threshold
