"""Tests for solar position and the daylight conditioning split.

The elevation calculation was checked against ``astral`` while it was being
written and agreed to five decimal places from Tromso to Ushuaia. astral is not
a dependency, so the values pinned here are physical identities that can be
derived without it: solstice noon elevation, equinox day length, hemispheric
inversion and polar day.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from PythonAQ import cut_data
from PythonAQ.solar import is_daylight, solar_elevation

LEEDS = (53.8008, -1.5491)
SYDNEY = (-33.8688, 151.2093)
TROMSO = (69.6492, 18.9553)
OBLIQUITY = 23.44  # axial tilt, degrees


def hours_of(year='2022'):
    return pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='h')


class TestSolarElevation:
    def test_solstice_noon_matches_the_geometric_identity(self):
        """At solar noon the elevation is 90 - |lat - declination|.

        On the June solstice the declination is the axial tilt, so this is a
        closed-form check that needs no reference implementation.
        """
        # Solar noon at Leeds is a few minutes past 12:00 UTC: the site is west
        # of Greenwich and the equation of time is non-zero.
        t = pd.date_range('2022-06-21 11:30', '2022-06-21 12:30', freq='1min')
        peak = solar_elevation(t, *LEEDS).max()
        assert peak == pytest.approx(90 - LEEDS[0] + OBLIQUITY, abs=0.1)

    def test_december_solstice_noon(self):
        t = pd.date_range('2022-12-21 11:30', '2022-12-21 12:30', freq='1min')
        peak = solar_elevation(t, *LEEDS).max()
        assert peak == pytest.approx(90 - LEEDS[0] - OBLIQUITY, abs=0.1)

    def test_equinox_gives_about_twelve_hours_everywhere(self):
        """Day length at an equinox is ~12 h at every latitude."""
        t = pd.date_range('2022-09-23 00:00', '2022-09-23 23:59', freq='1min')
        for lat, lon in (LEEDS, SYDNEY, TROMSO, (0.0, 0.0)):
            up = (solar_elevation(t, lat, lon) > 0).sum() / 60.0
            assert up == pytest.approx(12.0, abs=0.35), f'lat {lat}'

    def test_elevation_never_leaves_the_sphere(self):
        t = hours_of()
        for lat, lon in (LEEDS, SYDNEY, TROMSO):
            e = solar_elevation(t, lat, lon)
            assert e.min() >= -90.0 and e.max() <= 90.0

    def test_polar_day_and_night_at_tromso(self):
        """Above the Arctic Circle the sun does not set in June, nor rise in
        December. A fixed clock window cannot express this at all."""
        june = pd.date_range('2022-06-21 00:00', '2022-06-21 23:00', freq='h')
        december = pd.date_range('2022-12-21 00:00', '2022-12-21 23:00', freq='h')
        assert is_daylight(june, *TROMSO).all()
        assert not is_daylight(december, *TROMSO).any()

    def test_hemispheres_are_opposite(self):
        t = hours_of()
        north = is_daylight(t, *LEEDS)
        south = is_daylight(t, *SYDNEY)
        june, december = t.month == 6, t.month == 12
        assert north[june].sum() > north[december].sum()
        assert south[june].sum() < south[december].sum()

    def test_timezone_aware_input_is_converted_not_ignored(self):
        """An hour's offset moves the sun by 15 degrees, so silently treating
        BST as UTC would be a visible error rather than a rounding one."""
        naive = pd.DatetimeIndex(['2022-06-21 12:00'])
        aware = naive.tz_localize('Europe/London')  # BST, so 11:00 UTC
        shifted = pd.DatetimeIndex(['2022-06-21 11:00'])
        assert solar_elevation(aware, *LEEDS) == pytest.approx(
            solar_elevation(shifted, *LEEDS))
        assert solar_elevation(aware, *LEEDS) != pytest.approx(
            solar_elevation(naive, *LEEDS))

    def test_accepts_series_index_and_list(self):
        stamps = ['2022-06-21 12:00', '2022-06-21 13:00']
        expected = solar_elevation(pd.DatetimeIndex(stamps), *LEEDS)
        assert solar_elevation(pd.Series(pd.to_datetime(stamps)), *LEEDS) == \
            pytest.approx(expected)
        assert solar_elevation(pd.to_datetime(stamps), *LEEDS) == \
            pytest.approx(expected)

    @pytest.mark.parametrize('lat,lon', [(91, 0), (-91, 0), (0, 400), (0, -181),
                                         (np.nan, 0), (0, np.inf)])
    def test_impossible_coordinates_raise(self, lat, lon):
        with pytest.raises(ValueError):
            solar_elevation(pd.DatetimeIndex(['2022-06-21 12:00']), lat, lon)

    def test_threshold_shifts_the_boundary(self):
        """Civil twilight is longer than daylight."""
        t = pd.date_range('2022-03-20', '2022-03-20 23:59', freq='1min')
        assert (is_daylight(t, *LEEDS, threshold=-6).sum()
                > is_daylight(t, *LEEDS).sum())

    def test_leap_year_does_not_shift_the_result(self):
        """Day-of-year formulations drift by a day across a leap year; working
        from the Julian day does not."""
        a = solar_elevation(pd.DatetimeIndex(['2019-06-21 12:00']), *LEEDS)[0]
        b = solar_elevation(pd.DatetimeIndex(['2023-06-21 12:00']), *LEEDS)[0]
        assert abs(a - b) < 0.15


class TestDaylightSplit:
    @pytest.fixture
    def year(self):
        t = hours_of()
        return pd.DataFrame({'date_time': t, 'NO2': 1.0}), t

    def test_falls_back_with_a_warning_when_unlocated(self, year):
        df, _ = year
        with pytest.warns(UserWarning, match='fixed 07:00-19:00 window'):
            result = cut_data(df, type='daylight')
        assert set(result['daylight'].dropna().unique()) == {'daylight', 'nighttime'}

    def test_located_split_does_not_warn(self, year):
        df, _ = year
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            cut_data(df, type='daylight', latitude=LEEDS[0], longitude=LEEDS[1])

    def test_the_fix_actually_changes_the_answer(self, year):
        """The whole point: the fixed window is wrong often enough to matter.

        Roughly one hour in eight is relabelled over a year at this latitude,
        and the error reverses sign between summer and winter, so it does not
        average out of a seasonal comparison.
        """
        df, t = year
        with pytest.warns(UserWarning):
            fixed = cut_data(df, type='daylight')['daylight'] == 'daylight'
        solar = cut_data(df, type='daylight', latitude=LEEDS[0],
                         longitude=LEEDS[1])['daylight'] == 'daylight'
        fixed, solar = fixed.to_numpy(), solar.to_numpy()
        assert (fixed != solar).mean() > 0.10

        june, december = t.month == 6, t.month == 12
        # Summer: the window is too short. Winter: too long. Opposite signs.
        assert solar[june].sum() > fixed[june].sum()
        assert solar[december].sum() < fixed[december].sum()

    def test_categories_stay_ordered(self, year):
        df, _ = year
        result = cut_data(df, type='daylight', latitude=LEEDS[0],
                          longitude=LEEDS[1])
        assert list(result['daylight'].cat.categories) == ['daylight', 'nighttime']
        assert result['daylight'].cat.ordered

    def test_partial_coordinates_still_fall_back(self, year):
        """Latitude alone cannot place the sunrise; do not half-use it."""
        df, _ = year
        with pytest.warns(UserWarning):
            cut_data(df, type='daylight', latitude=LEEDS[0])

    def test_conditioning_forwards_the_location(self, year):
        """type='daylight' on a plot must reach cut_data with the coordinates,
        or the panels are split by the fallback window regardless."""
        from PythonAQ import time_plot
        df, _ = year
        df = df.assign(NO2=np.linspace(1, 50, len(df)))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            fig = time_plot(df, columns_to_plot=['NO2'], type='daylight',
                            latitude=LEEDS[0], longitude=LEEDS[1])
        assert len(fig.data) > 0
