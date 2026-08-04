"""Tests for the meteorological helpers and frequency utilities."""

import numpy as np
import pytest

from PythonAQ import e_sat, get_period, rh


class TestESat:
    def test_known_value_at_20c(self):
        # Magnus formula at 20 degC is ~23.4 hPa
        assert e_sat(20.0) == pytest.approx(23.4, abs=0.2)

    def test_accepts_scalar_input(self):
        # Regression: fancy-index assignment used to raise on 0-d arrays,
        # so any scalar call failed.
        result = e_sat(25.0)
        assert np.isscalar(result) or result.ndim == 0
        assert np.isfinite(result)

    def test_monotonically_increasing(self):
        temps = np.array([-20.0, 0.0, 10.0, 20.0, 30.0, 40.0])
        assert np.all(np.diff(e_sat(temps)) > 0)

    @pytest.mark.parametrize('temp', [-50.0, 70.0, np.nan])
    def test_out_of_range_gives_nan(self, temp):
        assert np.isnan(e_sat(temp))


class TestRelativeHumidity:
    def test_saturation_when_dewpoint_equals_temperature(self):
        assert rh(15.0, 15.0) == pytest.approx(100.0)

    def test_known_value(self):
        # 20 degC air, 10 degC dew point is ~52% RH
        assert rh(20.0, 10.0) == pytest.approx(52.0, abs=1.5)

    def test_dewpoint_is_not_rescaled(self):
        """Regression: rh() used to divide the dew point by 100.

        That turned a 15 degC dew point into 0.15 degC and produced a wildly
        low humidity. With equal inputs the answer must be saturation.
        """
        assert rh(15.0, 15.0) > 99.0

    def test_lower_dewpoint_gives_lower_humidity(self):
        assert rh(20.0, 15.0) > rh(20.0, 5.0)

    def test_result_is_bounded(self):
        temps = np.array([5.0, 15.0, 25.0, 35.0])
        dews = np.array([5.0, 14.0, 20.0, 35.0])
        result = rh(temps, dews)
        assert np.all((result >= 0) & (result <= 100))

    def test_mismatched_shapes_raise(self):
        # Regression: the ValueError was raised inside a bare `except Exception`
        # that swallowed it, so the validation never actually fired.
        with pytest.raises(ValueError):
            rh(np.array([1.0, 2.0]), np.array([1.0]))


class TestGetPeriod:
    @pytest.mark.parametrize('interval,expected', [
        ('D', 365),
        ('7D', 52),
        ('h', 8766),
        ('ME', 12),
        ('QE', 4),
    ])
    def test_known_intervals(self, interval, expected):
        assert get_period(interval) == expected

    def test_blank_defaults_to_daily(self):
        assert get_period(None) == 365
        assert get_period('') == 365

    def test_multiplier_is_applied(self):
        # 365.25 / 2 = 182.625, which rounds to 183
        assert get_period('2D') == 183
        assert get_period('2ME') == 6

    def test_unsupported_interval_raises(self):
        with pytest.raises(ValueError):
            get_period('3B')  # business-day offset

    def test_interval_longer_than_a_year_raises(self):
        # Would previously round down to period=0 and break decomposition.
        with pytest.raises(ValueError, match='longer than one year'):
            get_period('5YE')
