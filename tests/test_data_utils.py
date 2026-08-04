"""Tests for the openair data utility ports."""

import numpy as np
import pandas as pd
import pytest

from PythonAQ import (
    calc_percentile,
    cut_data,
    rolling_mean,
    select_by_date,
    time_average,
)


class TestTimeAverage:
    def test_daily_mean_matches_manual_resample(self, aq_df):
        result = time_average(aq_df[['date_time', 'NO2']], 'day')
        expected = (aq_df.set_index('date_time')['NO2']
                    .resample('D').mean().reset_index(drop=True))
        pd.testing.assert_series_equal(
            result['NO2'], expected, check_names=False
        )

    @pytest.mark.parametrize('avg_time,expected_rows', [
        ('hour', 24 * 366), ('day', 366), ('month', 12), ('year', 1),
    ])
    def test_period_lengths(self, avg_time, expected_rows):
        dates = pd.date_range('2020-01-01', '2020-12-31 23:00', freq='h')
        df = pd.DataFrame({'date_time': dates, 'x': 1.0})
        assert len(time_average(df, avg_time)) == expected_rows

    def test_openair_and_pandas_aliases_agree(self, aq_df):
        a = time_average(aq_df[['date_time', 'NO2']], 'day')
        b = time_average(aq_df[['date_time', 'NO2']], 'D')
        pd.testing.assert_frame_equal(a, b)

    def test_multiplied_period(self):
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({'date_time': dates, 'x': 1.0})
        assert len(time_average(df, '10 day')) == 10

    def test_wind_direction_is_vector_averaged(self):
        """Winds either side of north must not average to south."""
        df = pd.DataFrame({
            'date_time': pd.date_range('2020-01-01', periods=4, freq='h'),
            'wd': [350.0, 10.0, 355.0, 5.0],
            'ws': [5.0, 5.0, 5.0, 5.0],
        })
        result = time_average(df, 'day')
        assert result['wd'].iloc[0] == pytest.approx(0.0, abs=1.0)

    def test_scalar_wind_direction_mean_would_be_wrong(self):
        """Guards the above: the naive scalar mean really is 180."""
        assert np.mean([350.0, 10.0, 355.0, 5.0]) == pytest.approx(180.0)

    def test_data_threshold_blanks_sparse_periods(self):
        dates = pd.date_range('2020-01-01', periods=48, freq='h')
        values = np.full(48, np.nan)
        values[:24] = 1.0          # day 1 fully captured
        values[24:30] = 1.0        # day 2 only 25% captured
        df = pd.DataFrame({'date_time': dates, 'x': values})
        result = time_average(df, 'day', data_thresh=75)
        assert result['x'].iloc[0] == pytest.approx(1.0)
        assert np.isnan(result['x'].iloc[1])

    def test_no_threshold_keeps_sparse_periods(self):
        dates = pd.date_range('2020-01-01', periods=48, freq='h')
        values = np.full(48, np.nan)
        values[:24] = 1.0
        values[24:30] = 1.0
        df = pd.DataFrame({'date_time': dates, 'x': values})
        result = time_average(df, 'day', data_thresh=0)
        assert result['x'].notna().all()

    @pytest.mark.parametrize('statistic,expected', [
        ('mean', 2.0), ('median', 2.0), ('max', 3.0), ('min', 1.0),
        ('sum', 6.0), ('frequency', 3.0),
    ])
    def test_statistics(self, statistic, expected):
        df = pd.DataFrame({
            'date_time': pd.date_range('2020-01-01', periods=3, freq='h'),
            'x': [1.0, 2.0, 3.0],
        })
        assert time_average(df, 'day', statistic=statistic)['x'].iloc[0] == expected

    def test_percentile_statistic(self):
        df = pd.DataFrame({
            'date_time': pd.date_range('2020-01-01', periods=101, freq='h'),
            'x': np.arange(101.0),
        })
        result = time_average(df, 'year', statistic='percentile', percentile=50)
        assert result['x'].iloc[0] == pytest.approx(50.0)

    def test_percentile_requires_a_value(self, aq_df):
        with pytest.raises(ValueError, match='percentile must be given'):
            time_average(aq_df, 'day', statistic='percentile')

    def test_unknown_statistic_raises(self, aq_df):
        with pytest.raises(ValueError, match='Unknown statistic'):
            time_average(aq_df, 'day', statistic='nonsense')

    def test_invalid_threshold_raises(self, aq_df):
        with pytest.raises(ValueError, match='between 0 and 100'):
            time_average(aq_df, 'day', data_thresh=150)

    def test_constant_metadata_is_carried_through(self, aq_df):
        result = time_average(aq_df, 'day')
        assert (result['site'] == 'Test Site').all()

    def test_does_not_mutate_input(self, aq_df):
        before = aq_df.copy()
        time_average(aq_df, 'day')
        pd.testing.assert_frame_equal(aq_df, before)


class TestSelectByDate:
    def test_start_and_end(self, aq_df):
        result = select_by_date(aq_df, start='2021-01-01', end='2021-12-31')
        assert result['date_time'].dt.year.unique().tolist() == [2021]
        # A bare end date includes the whole of that day
        assert result['date_time'].max().hour == 23

    def test_year(self, aq_df):
        assert select_by_date(aq_df, year=2020)['date_time'].dt.year.unique().tolist() == [2020]

    def test_multiple_years(self, aq_df):
        years = select_by_date(aq_df, year=[2020, 2022])['date_time'].dt.year.unique()
        assert sorted(years) == [2020, 2022]

    @pytest.mark.parametrize('month', ['June', 'jun', 6])
    def test_month_by_name_or_number(self, aq_df, month):
        result = select_by_date(aq_df, month=month)
        assert result['date_time'].dt.month.unique().tolist() == [6]

    def test_weekday_names(self, aq_df):
        result = select_by_date(aq_df, day=['Saturday', 'Sunday'])
        assert set(result['date_time'].dt.day_name()) == {'Saturday', 'Sunday'}

    def test_hour(self, aq_df):
        result = select_by_date(aq_df, hour=[0, 12])
        assert sorted(result['date_time'].dt.hour.unique()) == [0, 12]

    def test_season(self, aq_df):
        result = select_by_date(aq_df, season='summer')
        assert sorted(result['date_time'].dt.month.unique()) == [6, 7, 8]

    def test_criteria_combine_with_and(self, aq_df):
        result = select_by_date(aq_df, year=2021, month='March', hour=9)
        assert (result['date_time'].dt.year == 2021).all()
        assert (result['date_time'].dt.month == 3).all()
        assert (result['date_time'].dt.hour == 9).all()

    def test_unrecognised_month_raises(self, aq_df):
        with pytest.raises(ValueError, match='Unrecognised month'):
            select_by_date(aq_df, month='Smarch')

    def test_does_not_mutate_input(self, aq_df):
        before = aq_df.copy()
        select_by_date(aq_df, year=2021)
        pd.testing.assert_frame_equal(aq_df, before)


class TestRollingMean:
    def test_default_column_name(self, aq_df):
        assert 'rolling8_NO2' in rolling_mean(aq_df, 'NO2').columns

    def test_custom_name(self, aq_df):
        assert 'o3_8hr' in rolling_mean(aq_df, 'NO2', new_name='o3_8hr').columns

    def test_known_values_right_aligned(self):
        df = pd.DataFrame({
            'date_time': pd.date_range('2020-01-01', periods=5, freq='h'),
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = rolling_mean(df, 'x', width=3, align='right', data_thresh=100)
        # First two windows are incomplete, then trailing means of 3
        assert np.isnan(result['rolling3_x'].iloc[1])
        assert result['rolling3_x'].iloc[2] == pytest.approx(2.0)
        assert result['rolling3_x'].iloc[4] == pytest.approx(4.0)

    def test_centred_alignment(self):
        df = pd.DataFrame({
            'date_time': pd.date_range('2020-01-01', periods=5, freq='h'),
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = rolling_mean(df, 'x', width=3, align='centre', data_thresh=100)
        assert result['rolling3_x'].iloc[2] == pytest.approx(3.0)

    def test_threshold_blanks_sparse_windows(self):
        df = pd.DataFrame({
            'date_time': pd.date_range('2020-01-01', periods=8, freq='h'),
            'x': [1.0] + [np.nan] * 7,
        })
        result = rolling_mean(df, 'x', width=8, data_thresh=75)
        assert result['rolling8_x'].isna().all()

    def test_invalid_align_raises(self, aq_df):
        with pytest.raises(ValueError, match='align must be'):
            rolling_mean(aq_df, 'NO2', align='sideways')

    def test_missing_column_raises(self, aq_df):
        with pytest.raises(ValueError, match='not found'):
            rolling_mean(aq_df, 'NOPE')


class TestCutData:
    @pytest.mark.parametrize('type_,expected', [
        ('season', {'winter (DJF)', 'spring (MAM)', 'summer (JJA)', 'autumn (SON)'}),
        ('weekend', {'weekday', 'weekend'}),
        ('daylight', {'daylight', 'nighttime'}),
    ])
    def test_category_sets(self, aq_df, type_, expected):
        assert set(cut_data(aq_df, type=type_)[type_].dropna().unique()) == expected

    def test_weekday_is_ordered_from_monday(self, aq_df):
        values = cut_data(aq_df, type='weekday')['weekday']
        assert list(values.cat.categories)[:2] == ['Monday', 'Tuesday']

    def test_season_is_hemisphere_aware(self):
        january = pd.DataFrame({'date_time': [pd.Timestamp('2020-01-15')]})
        north = cut_data(january, type='season')['season'].iloc[0]
        south = cut_data(january, type='season', hemisphere='southern')['season'].iloc[0]
        assert north == 'winter (DJF)'
        assert south == 'summer (JJA)'

    def test_wd_gives_compass_sectors_not_quantiles(self, aq_df):
        """Regression: 'wd' is numeric and was being quantile-split."""
        result = cut_data(aq_df, type='wd')
        assert set(result['wd'].dropna().unique()) <= {
            'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'
        }

    def test_wd_sectors_are_centred_on_north(self):
        df = pd.DataFrame({'wd': [0.0, 10.0, 350.0, 90.0, 180.0, 270.0]})
        sectors = cut_data(df, type='wd')['wd'].tolist()
        assert sectors == ['N', 'N', 'N', 'E', 'S', 'W']

    def test_numeric_column_is_split_into_quantiles(self, aq_df):
        result = cut_data(aq_df, type='NO2', n_levels=4)
        assert len(result['NO2'].cat.categories) == 4

    def test_hour_and_year(self, aq_df):
        assert set(cut_data(aq_df, type='hour')['hour'].unique()) == set(range(24))
        assert set(cut_data(aq_df, type='year')['year'].unique()) == {2020, 2021, 2022}

    def test_unknown_type_raises(self, aq_df):
        with pytest.raises(ValueError, match='Unknown type'):
            cut_data(aq_df, type='fortnight')


class TestCalcPercentile:
    def test_columns_are_named_per_percentile(self, aq_df):
        result = calc_percentile(aq_df, 'NO2', percentile=(25, 50, 75))
        assert list(result.columns) == [
            'date_time', 'percentile.25', 'percentile.50', 'percentile.75'
        ]

    def test_percentiles_are_ordered(self, aq_df):
        result = calc_percentile(aq_df, 'NO2', percentile=(25, 50, 75)).dropna()
        assert (result['percentile.25'] <= result['percentile.50']).all()
        assert (result['percentile.50'] <= result['percentile.75']).all()

    def test_missing_column_raises(self, aq_df):
        with pytest.raises(ValueError, match='not found'):
            calc_percentile(aq_df, 'NOPE')
