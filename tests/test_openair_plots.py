"""Tests for the newly ported openair plotting functions."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from PythonAQ import (
    corr_plot,
    percentile_rose,
    polar_annulus,
    scatter_plot,
    time_variation,
    trend_level,
)


def assert_is_populated_figure(fig):
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0, 'figure has no traces'


@pytest.fixture
def structured_df(rng):
    """Data with a known diurnal peak, weekend dip and seasonal cycle."""
    dates = pd.date_range('2020-01-01', '2021-12-31 23:00', freq='h')
    n = len(dates)
    hour = dates.hour.to_numpy()
    # A single clean morning peak at 08:00
    diurnal = 15 * np.exp(-((hour - 8) ** 2) / 6.0)
    weekend = np.where(dates.dayofweek.to_numpy() >= 5, -10.0, 0.0)
    seasonal = 8 * np.sin(2 * np.pi * np.arange(n) / (365.25 * 24))

    no2 = 40 + diurnal + weekend + seasonal + rng.normal(0, 2, n)
    return pd.DataFrame({
        'date_time': dates,
        'NO2': no2,
        'NOx': no2 * 1.8 + rng.normal(0, 2, n),
        'PM10': 20 + 0.3 * diurnal + rng.normal(0, 3, n),
        'ws': rng.gamma(2.0, 2.0, n),
        'wd': rng.uniform(0, 360, n),
    })


class TestTimeVariation:
    def test_returns_figure_and_summary(self, structured_df):
        fig, summary = time_variation(structured_df, 'NO2', random_state=0)
        assert_is_populated_figure(fig)
        assert set(summary['panel']) == {'weekday.hour', 'weekday', 'hour', 'month'}

    def test_recovers_the_diurnal_peak(self, structured_df):
        _, summary = time_variation(structured_df, 'NO2', random_state=0)
        hourly = summary[summary['panel'] == 'hour']
        assert int(hourly.loc[hourly['value'].idxmax(), 'x']) == 8

    def test_recovers_the_weekend_dip(self, structured_df):
        _, summary = time_variation(structured_df, 'NO2', random_state=0)
        weekday = summary[summary['panel'] == 'weekday'].set_index('x')['value']
        assert weekday['Saturday'] < weekday['Wednesday'] - 5
        assert weekday['Sunday'] < weekday['Wednesday'] - 5

    def test_weekday_hour_panel_spans_exactly_one_week(self, structured_df):
        """Regression: cat.codes is int8, so Sunday's 6 * 24 = 144 overflowed
        to -112 and its block landed off the left-hand end of the axis."""
        _, summary = time_variation(structured_df, 'NO2', random_state=0)
        x = sorted(summary[summary['panel'] == 'weekday.hour']['x'].astype(int))
        assert x == list(range(168)), 'panel must run 0-167 with no gaps'

    def test_weekday_hour_blocks_line_up_with_the_weekday_panel(self, structured_df):
        """Each 24-hour block must be the weekday its axis label claims.

        Cross-checked against the independently computed day-of-week panel, so
        this catches a wrong offset as well as a wrong order.
        """
        _, summary = time_variation(structured_df, 'NO2', random_state=0)
        blocks = summary[summary['panel'] == 'weekday.hour'].set_index('x')['value']
        by_day = summary[summary['panel'] == 'weekday'].set_index('x')['value']

        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                    'Saturday', 'Sunday']
        for index, day in enumerate(weekdays):
            block_mean = blocks.loc[index * 24:index * 24 + 23].mean()
            assert block_mean == pytest.approx(by_day[day], abs=0.5), day

    def test_weekday_hour_axis_is_pinned_to_the_full_week(self, structured_df):
        """The tick labels are positioned absolutely, so the range must be too,
        or a missing weekday would shift the data out from under its label."""
        fig, _ = time_variation(structured_df, 'NO2', random_state=0)
        assert list(fig.layout.xaxis.range) == [-1, 168]
        assert list(fig.layout.xaxis.tickvals) == [d * 24 + 12 for d in range(7)]

    def test_a_missing_weekday_does_not_shift_the_others(self, structured_df):
        """Dropping Wednesday must leave every other day where it was."""
        without_wednesday = structured_df[
            structured_df['date_time'].dt.day_name() != 'Wednesday'
        ]
        _, summary = time_variation(without_wednesday, 'NO2', random_state=0)
        x = set(summary[summary['panel'] == 'weekday.hour']['x'].astype(int))
        assert not (x & set(range(48, 72))), 'Wednesday block should be empty'
        assert set(range(0, 24)) <= x, 'Monday should still be at 0-23'
        assert set(range(144, 168)) <= x, 'Sunday should still be at 144-167'

    def test_confidence_interval_brackets_the_estimate(self, structured_df):
        _, summary = time_variation(structured_df, 'NO2', random_state=0)
        valid = summary.dropna(subset=['lower', 'upper'])
        assert (valid['lower'] <= valid['value']).all()
        assert (valid['value'] <= valid['upper']).all()

    def test_wider_confidence_level_gives_wider_interval(self, structured_df):
        _, narrow = time_variation(structured_df, 'NO2', conf_int=0.50,
                                   n_boot=200, random_state=0)
        _, wide = time_variation(structured_df, 'NO2', conf_int=0.99,
                                 n_boot=200, random_state=0)
        narrow_width = (narrow['upper'] - narrow['lower']).mean()
        wide_width = (wide['upper'] - wide['lower']).mean()
        assert wide_width > narrow_width

    def test_is_reproducible_with_a_seed(self, structured_df):
        _, a = time_variation(structured_df, 'NO2', random_state=7)
        _, b = time_variation(structured_df, 'NO2', random_state=7)
        pd.testing.assert_frame_equal(a, b)

    def test_multiple_pollutants(self, structured_df):
        fig, summary = time_variation(structured_df, ['NO2', 'PM10'], random_state=0)
        assert_is_populated_figure(fig)
        assert set(summary['pollutant']) == {'NO2', 'PM10'}

    def test_normalise_puts_series_on_a_common_scale(self, structured_df):
        _, summary = time_variation(structured_df, ['NO2', 'PM10'],
                                    normalise=True, random_state=0)
        hourly = summary[summary['panel'] == 'hour']
        # Each normalised series averages to roughly 1
        for name in ('NO2', 'PM10'):
            assert hourly[hourly['pollutant'] == name]['value'].mean() == pytest.approx(1.0, abs=0.1)

    def test_median_statistic(self, structured_df):
        fig, _ = time_variation(structured_df, 'NO2', statistic='median',
                                random_state=0)
        assert_is_populated_figure(fig)

    def test_ci_can_be_disabled(self, structured_df):
        with_ci, _ = time_variation(structured_df, 'NO2', ci=True, random_state=0)
        without_ci, _ = time_variation(structured_df, 'NO2', ci=False, random_state=0)
        assert len(without_ci.data) < len(with_ci.data)

    def test_invalid_statistic_raises(self, structured_df):
        with pytest.raises(ValueError, match="'mean' or 'median'"):
            time_variation(structured_df, 'NO2', statistic='mode')

    def test_invalid_conf_int_raises(self, structured_df):
        with pytest.raises(ValueError, match='strictly between 0 and 1'):
            time_variation(structured_df, 'NO2', conf_int=95)

    def test_missing_column_raises(self, structured_df):
        with pytest.raises(ValueError, match='not found'):
            time_variation(structured_df, 'NOPE')

    def test_does_not_mutate_input(self, structured_df):
        before = structured_df.copy()
        time_variation(structured_df, 'NO2', random_state=0)
        pd.testing.assert_frame_equal(structured_df, before)


class TestPercentileRose:
    def test_returns_figure_and_summary(self, structured_df):
        fig, summary = percentile_rose(structured_df, 'NO2')
        assert_is_populated_figure(fig)
        assert len(summary) == 36

    def test_percentiles_are_monotonic_within_each_sector(self, structured_df):
        _, summary = percentile_rose(structured_df, 'NO2', percentile=(25, 50, 75, 95))
        valid = summary.dropna()
        assert (valid['percentile.25'] <= valid['percentile.50']).all()
        assert (valid['percentile.50'] <= valid['percentile.75']).all()
        assert (valid['percentile.75'] <= valid['percentile.95']).all()

    def test_bin_count_is_respected(self, structured_df):
        _, summary = percentile_rose(structured_df, 'NO2', direction_bins=8)
        assert len(summary) == 8
        assert summary['wd'].tolist() == [0, 45, 90, 135, 180, 225, 270, 315]

    def test_directional_signal_is_detected(self, rng):
        """Concentrations elevated from the east must show up in that sector."""
        n = 20000
        wd = rng.uniform(0, 360, n)
        # Add 50 units when the wind is easterly (around 90 degrees)
        no2 = 20 + 50 * (np.abs(wd - 90) < 20) + rng.normal(0, 2, n)
        df = pd.DataFrame({'wd': wd, 'NO2': no2})
        _, summary = percentile_rose(df, 'NO2', direction_bins=36)
        peak = summary.loc[summary['percentile.50'].idxmax(), 'wd']
        assert 70 <= peak <= 110

    def test_invalid_percentile_raises(self, structured_df):
        with pytest.raises(ValueError, match='between 0 and 100'):
            percentile_rose(structured_df, 'NO2', percentile=(50, 150))

    def test_missing_column_raises(self, structured_df):
        with pytest.raises(ValueError, match='not found'):
            percentile_rose(structured_df, 'NOPE')


class TestTrendLevel:
    def test_default_axes(self, structured_df):
        fig, summary = trend_level(structured_df, 'NO2')
        assert_is_populated_figure(fig)
        assert {'month', 'hour', 'year'} <= set(summary.columns)

    def test_one_panel_per_year(self, structured_df):
        fig, _ = trend_level(structured_df, 'NO2', type='year')
        assert len(fig.data) == 2  # 2020 and 2021

    def test_type_none_gives_a_single_panel(self, structured_df):
        fig, _ = trend_level(structured_df, 'NO2', type=None)
        assert len(fig.data) == 1

    def test_recovers_the_diurnal_peak(self, structured_df):
        _, summary = trend_level(structured_df, 'NO2', x='month', y='hour', type=None)
        by_hour = summary.groupby('hour')['mean'].mean()
        assert by_hour.idxmax() == 8

    def test_alternative_axes(self, structured_df):
        fig, summary = trend_level(structured_df, 'NO2', x='weekday', y='hour',
                                   type='season')
        assert_is_populated_figure(fig)
        assert 'weekday' in summary.columns

    def test_unknown_statistic_raises(self, structured_df):
        with pytest.raises(ValueError, match='Unknown statistic'):
            trend_level(structured_df, 'NO2', statistic='nonsense')

    def test_missing_column_raises(self, structured_df):
        with pytest.raises(ValueError, match='not found'):
            trend_level(structured_df, 'NOPE')


class TestScatterPlot:
    def test_basic_scatter(self, structured_df):
        fig, summary = scatter_plot(structured_df, 'NO2', 'NOx')
        assert_is_populated_figure(fig)
        assert summary.empty  # no fit requested

    def test_linear_fit_recovers_a_known_slope(self, rng):
        x = rng.uniform(0, 100, 5000)
        y = 3.0 * x + 7.0 + rng.normal(0, 0.5, 5000)
        df = pd.DataFrame({'x': x, 'y': y})
        _, summary = scatter_plot(df, 'x', 'y', linear=True)
        assert summary['slope'].iloc[0] == pytest.approx(3.0, abs=0.02)
        assert summary['intercept'].iloc[0] == pytest.approx(7.0, abs=0.1)
        assert summary['r_squared'].iloc[0] > 0.99

    def test_smooth(self, structured_df):
        fig, _ = scatter_plot(structured_df, 'NO2', 'NOx', smooth=True)
        assert any(trace.name == 'smooth' for trace in fig.data)

    def test_one_to_one_line(self, structured_df):
        fig, _ = scatter_plot(structured_df, 'NO2', 'NOx', one_to_one=True)
        assert any(trace.name == '1:1' for trace in fig.data)

    @pytest.mark.parametrize('method', ['scatter', 'hexbin', 'density'])
    def test_methods(self, structured_df, method):
        fig, _ = scatter_plot(structured_df, 'NO2', 'NOx', method=method)
        assert_is_populated_figure(fig)

    def test_colour_by(self, structured_df):
        fig, _ = scatter_plot(structured_df, 'NO2', 'NOx', colour_by='ws')
        assert_is_populated_figure(fig)

    def test_invalid_method_raises(self, structured_df):
        with pytest.raises(ValueError, match="method must be"):
            scatter_plot(structured_df, 'NO2', 'NOx', method='nonsense')

    def test_no_complete_pairs_raises(self):
        df = pd.DataFrame({'x': [1.0, np.nan], 'y': [np.nan, 2.0]})
        with pytest.raises(ValueError, match='No complete'):
            scatter_plot(df, 'x', 'y')


class TestCorrPlot:
    def test_returns_figure_and_matrix(self, structured_df):
        fig, corr = corr_plot(structured_df, ['NO2', 'NOx', 'PM10', 'ws'])
        assert_is_populated_figure(fig)
        assert corr.shape == (4, 4)

    def test_diagonal_is_one(self, structured_df):
        _, corr = corr_plot(structured_df, ['NO2', 'NOx', 'PM10'])
        assert np.allclose(np.diag(corr.values), 1.0)

    def test_matrix_is_symmetric(self, structured_df):
        _, corr = corr_plot(structured_df, ['NO2', 'NOx', 'PM10'])
        assert np.allclose(corr.values, corr.values.T)

    def test_detects_the_known_strong_correlation(self, structured_df):
        _, corr = corr_plot(structured_df, ['NO2', 'NOx', 'ws'])
        assert corr.loc['NO2', 'NOx'] > 0.95

    @pytest.mark.parametrize('method', ['pearson', 'spearman', 'kendall'])
    def test_methods(self, structured_df, method):
        _, corr = corr_plot(structured_df, ['NO2', 'NOx', 'PM10'], method=method)
        assert corr.loc['NO2', 'NOx'] > 0.8

    def test_clustering_preserves_the_variable_set(self, structured_df):
        names = ['NO2', 'NOx', 'PM10', 'ws']
        _, clustered = corr_plot(structured_df, names, cluster=True)
        _, plain = corr_plot(structured_df, names, cluster=False)
        assert set(clustered.columns) == set(plain.columns)

    def test_defaults_to_all_numeric_columns(self, structured_df):
        _, corr = corr_plot(structured_df)
        assert 'NO2' in corr.columns and 'ws' in corr.columns

    def test_constant_columns_are_dropped(self, structured_df):
        df = structured_df.assign(constant=1.0)
        _, corr = corr_plot(df, ['NO2', 'NOx', 'constant'])
        assert 'constant' not in corr.columns

    def test_too_few_columns_raises(self, structured_df):
        with pytest.raises(ValueError, match='At least two'):
            corr_plot(structured_df, ['NO2'])

    def test_invalid_method_raises(self, structured_df):
        with pytest.raises(ValueError, match='method must be'):
            corr_plot(structured_df, ['NO2', 'NOx'], method='nonsense')


class TestConditionalProbabilityFunction:
    """percentile_rose(statistic='cpf'), the openair book's section 7.3.

    CPF asks: when the wind comes from this sector, how often is the
    concentration high? That isolates directions responsible for episodes,
    which a directional mean can miss when a source is intermittent.
    """

    @pytest.fixture
    def directional_source(self, rng):
        """20% of the time, easterly winds carry a strong plume."""
        n = 40000
        wd = rng.uniform(0, 360, n)
        easterly = np.abs(((wd - 90 + 180) % 360) - 180) < 25
        high = easterly & (rng.random(n) < 0.7)
        no2 = np.where(high, rng.gamma(9, 12, n), rng.gamma(2, 8, n))
        return pd.DataFrame({'wd': wd, 'NO2': no2})

    def test_returns_probabilities(self, directional_source):
        _, summary = percentile_rose(directional_source, 'NO2', statistic='cpf')
        assert 'cpf' in summary.columns
        valid = summary['cpf'].dropna()
        assert ((valid >= 0) & (valid <= 1)).all()

    def test_locates_the_source(self, directional_source):
        _, summary = percentile_rose(directional_source, 'NO2',
                                     statistic='cpf', percentile=90)
        peak = summary.loc[summary['cpf'].idxmax(), 'wd']
        assert 60 <= peak <= 120, f'CPF peaked at {peak}, source is at 90'

    def test_probability_matches_the_planted_rate(self, directional_source):
        """Within the plume sector roughly 70% of values should be high."""
        _, summary = percentile_rose(directional_source, 'NO2',
                                     statistic='cpf', percentile=90)
        assert summary['cpf'].max() > 0.5

    def test_threshold_defaults_to_the_requested_percentile(self, directional_source):
        _, summary = percentile_rose(directional_source, 'NO2',
                                     statistic='cpf', percentile=75)
        expected = np.percentile(directional_source['NO2'], 75)
        assert summary['threshold'].iloc[0] == pytest.approx(expected)

    def test_explicit_threshold_is_used(self, directional_source):
        _, summary = percentile_rose(directional_source, 'NO2',
                                     statistic='cpf', cpf_threshold=100.0)
        assert (summary['threshold'] == 100.0).all()

    def test_higher_threshold_gives_lower_probabilities(self, directional_source):
        _, low = percentile_rose(directional_source, 'NO2', statistic='cpf',
                                 percentile=50)
        _, high = percentile_rose(directional_source, 'NO2', statistic='cpf',
                                  percentile=95)
        assert high['cpf'].mean() < low['cpf'].mean()

    def test_radial_axis_is_fixed_to_probability_range(self, directional_source):
        """Two sites are only comparable if the scale does not float."""
        fig, _ = percentile_rose(directional_source, 'NO2', statistic='cpf')
        assert tuple(fig.layout.polar.radialaxis.range) == (0, 1)

    def test_counts_are_reported(self, directional_source):
        _, summary = percentile_rose(directional_source, 'NO2', statistic='cpf')
        assert (summary['n_above'] <= summary['n']).all()

    def test_invalid_statistic_raises(self, structured_df):
        with pytest.raises(ValueError, match="'percentile' or 'cpf'"):
            percentile_rose(structured_df, 'NO2', statistic='nonsense')


class TestPolarAnnulus:
    @pytest.mark.parametrize('period,levels', [
        ('hour', 24), ('month', 12), ('weekday', 7), ('season', 4),
    ])
    def test_periods(self, structured_df, period, levels):
        fig, summary = polar_annulus(structured_df, 'NO2', period=period)
        assert_is_populated_figure(fig)
        assert summary['level'].nunique() == levels

    def test_trend_period_uses_one_ring_per_year(self, structured_df):
        _, summary = polar_annulus(structured_df, 'NO2', period='trend')
        expected = structured_df['date_time'].dt.year.nunique()
        assert summary['level'].nunique() == expected

    def test_centre_is_hollow(self, structured_df):
        """The hole stops the innermost ring being squeezed to nothing."""
        fig, _ = polar_annulus(structured_df, 'NO2', inner_radius=0.4)
        z = np.asarray(fig.data[0].z, dtype=float)
        middle = z.shape[0] // 2
        assert np.isnan(z[middle, middle])

    def test_outside_the_annulus_is_blank(self, structured_df):
        fig, _ = polar_annulus(structured_df, 'NO2')
        z = np.asarray(fig.data[0].z, dtype=float)
        assert np.isnan(z[0, 0]) and np.isnan(z[-1, -1])
        assert np.isfinite(z).any()

    def test_recovers_a_source_active_only_at_night(self, rng):
        """A plume present only between 00:00 and 04:00 from the west."""
        dates = pd.date_range('2022-01-01', '2023-12-31 23:00', freq='h')
        n = len(dates)
        wd = rng.uniform(0, 360, n)
        westerly = np.abs(((wd - 270 + 180) % 360) - 180) < 30
        at_night = dates.hour.to_numpy() < 4
        no2 = 20 + np.where(westerly & at_night, 60.0, 0.0) + rng.normal(0, 2, n)
        df = pd.DataFrame({'date_time': dates, 'wd': wd, 'NO2': no2})

        _, summary = polar_annulus(df, 'NO2', period='hour', smooth=False)
        hot = summary.loc[summary['mean'].idxmax()]
        assert hot['level'] < 4, 'should be found in the small hours'
        assert 240 <= hot['wd'] <= 300, 'should be found to the west'

    def test_invalid_period_raises(self, structured_df):
        with pytest.raises(ValueError, match='Unknown period'):
            polar_annulus(structured_df, 'NO2', period='fortnight')

    def test_invalid_inner_radius_raises(self, structured_df):
        with pytest.raises(ValueError, match='inner_radius'):
            polar_annulus(structured_df, 'NO2', inner_radius=1.5)

    def test_missing_column_raises(self, structured_df):
        with pytest.raises(ValueError, match='not found'):
            polar_annulus(structured_df, 'NOPE')

    def test_does_not_mutate_input(self, structured_df):
        before = structured_df.copy()
        polar_annulus(structured_df, 'NO2')
        pd.testing.assert_frame_equal(structured_df, before)
