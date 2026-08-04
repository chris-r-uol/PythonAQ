"""Tests for NOAA/AURN data handling. No network access."""

import numpy as np
import pytest

from PythonAQ import import_aq_meta, parse_noaa_data


class TestParseNoaaData:
    @pytest.fixture
    def parsed(self, noaa_raw):
        return parse_noaa_data(noaa_raw)

    def test_returns_expected_columns(self, parsed):
        assert not parsed.empty
        for col in ['air_temp', 'dew_point', 'atmospheric_pressure', 'ws', 'wd',
                    'site', 'relative_humidity']:
            assert col in parsed.columns

    def test_temperature_is_scaled(self, parsed):
        # '+0150' is tenths of degC, so 15.0 degC
        assert parsed['air_temp'].iloc[0] == pytest.approx(15.0)
        assert parsed['air_temp'].iloc[1] == pytest.approx(10.5)
        assert parsed['air_temp'].iloc[2] == pytest.approx(-5.0)

    def test_dew_point_is_scaled(self, parsed):
        """Regression: dew point was never divided by 10.

        It was left in tenths of a degree and then divided by 100 inside rh(),
        so every derived humidity was wrong.
        """
        assert parsed['dew_point'].iloc[0] == pytest.approx(10.0)
        assert parsed['dew_point'].iloc[1] == pytest.approx(5.0)
        assert parsed['dew_point'].iloc[2] == pytest.approx(-10.0)

    def test_pressure_is_scaled(self, parsed):
        """Regression: SLP was left in tenths of hPa."""
        assert parsed['atmospheric_pressure'].iloc[0] == pytest.approx(1013.2)
        assert parsed['atmospheric_pressure'].iloc[1] == pytest.approx(1009.8)

    def test_wind_is_parsed(self, parsed):
        assert parsed['wd'].iloc[0] == pytest.approx(180.0)
        assert parsed['ws'].iloc[0] == pytest.approx(5.0)
        assert parsed['ws'].iloc[1] == pytest.approx(10.3)

    def test_sentinels_become_nan(self, parsed):
        """The final fixture row uses the documented ISD missing values."""
        last = parsed.iloc[-1]
        for col in ['air_temp', 'dew_point', 'atmospheric_pressure', 'ws', 'wd']:
            assert np.isnan(last[col]), f'{col} sentinel was not converted to NaN'

    def test_relative_humidity_is_plausible(self, parsed):
        # 15 degC air with a 10 degC dew point is roughly 72% RH
        assert parsed['relative_humidity'].iloc[0] == pytest.approx(72.0, abs=2.0)
        valid = parsed['relative_humidity'].dropna()
        assert ((valid >= 0) & (valid <= 100)).all()

    def test_missing_optional_column_is_tolerated(self, noaa_raw):
        """A station that does not report SLP must still parse.

        The hard-coded resample column list previously raised a KeyError that
        was swallowed, silently returning an empty DataFrame.
        """
        result = parse_noaa_data(noaa_raw.drop(columns=['SLP']))
        assert not result.empty
        assert 'air_temp' in result.columns
        assert 'atmospheric_pressure' not in result.columns

    def test_empty_input_returns_empty_frame(self, noaa_raw):
        assert parse_noaa_data(noaa_raw.iloc[0:0]).empty

    def test_index_is_hourly(self, parsed):
        assert parsed.index.name == 'date'
        assert len(parsed) == 4


class TestImportAqMeta:
    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match='Invalid source'):
            import_aq_meta('not_a_real_source')
