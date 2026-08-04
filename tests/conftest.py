"""Shared fixtures providing synthetic data, so tests never touch the network."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(20240101)


@pytest.fixture
def aq_df(rng):
    """Three years of hourly air quality + meteorology data.

    Contains a genuine downward trend and a seasonal cycle so that the trend
    functions have something to find, plus a realistic sprinkling of NaNs.
    """
    dates = pd.date_range('2020-01-01', '2022-12-31 23:00', freq='h')
    n = len(dates)
    t = np.arange(n)

    seasonal = 10 * np.sin(2 * np.pi * t / (365.25 * 24))
    diurnal = 5 * np.sin(2 * np.pi * t / 24)
    trend = -3.0 * t / (365.25 * 24)  # -3 units per year

    no2 = 40 + seasonal + diurnal + trend + rng.normal(0, 5, n)
    pm10 = 20 + 0.5 * seasonal + trend / 2 + rng.normal(0, 4, n)

    df = pd.DataFrame({
        'date': dates,
        'date_time': dates,
        'site': 'Test Site',
        'code': 'TEST',
        'NO2': no2.clip(min=0),
        'PM10': pm10.clip(min=0),
        'ws': rng.gamma(2.0, 2.0, n),
        'wd': rng.uniform(0, 360, n),
    })

    # Introduce missing values, as real monitoring data always has them.
    missing = rng.choice(n, size=n // 20, replace=False)
    df.loc[missing, 'NO2'] = np.nan
    return df


@pytest.fixture
def noaa_raw():
    """A raw NOAA ISD 'global-hourly' frame, including missing-value sentinels.

    ISD packs several sub-fields per column and scales them by 10; the final
    row uses the documented sentinels for every field.
    """
    return pd.DataFrame({
        'STATION': ['03377099999'] * 4,
        'DATE': [
            '2020-01-01T00:00:00', '2020-01-01T01:00:00',
            '2020-01-01T02:00:00', '2020-01-01T03:00:00',
        ],
        # temperature: +15.0, +10.5, -5.0 degC, then missing
        'TMP': ['+0150,1', '+0105,1', '-0050,1', '+9999,9'],
        # dew point: +10.0, +5.0, -10.0 degC, then missing
        'DEW': ['+0100,1', '+0050,1', '-0100,1', '+9999,9'],
        # sea level pressure: 1013.2, 1009.8, 1020.0 hPa, then missing
        'SLP': ['10132,1', '10098,1', '10200,1', '99999,9'],
        # wind: direction deg, quality, type, speed (tenths m/s), quality
        'WND': [
            '180,1,N,0050,1',   # 180 deg, 5.0 m/s
            '270,1,N,0103,1',   # 270 deg, 10.3 m/s
            '090,1,N,0000,1',   # 90 deg, calm
            '999,9,C,9999,9',   # missing
        ],
    })
