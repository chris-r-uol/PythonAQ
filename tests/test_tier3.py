"""Tests for the tier-3 ports: dist_plot, linear_relation, run_regression
and conditional_eval.

Where a function estimates something, the fixture builds data with a known
answer and the test requires it back.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from PythonAQ import conditional_eval, dist_plot, linear_relation, run_regression

TRUE_SLOPE = 0.30


@pytest.fixture
def linked(rng):
    """NO2 built as a known multiple of NOx, plus a model with a wind-speed bias."""
    n = 6000
    ws = rng.gamma(2.0, 2.0, n)
    nox = 60 + 200 * np.exp(-ws / 3.0) + rng.normal(0, 15, n)
    return pd.DataFrame({
        'date_time': pd.date_range('2022-01-01', periods=n, freq='h'),
        'NOx': nox,
        'NO2': TRUE_SLOPE * nox + 5 + rng.normal(0, 4, n),
        'ws': ws,
        'temp': 10 + rng.normal(0, 3, n),
        'obs': nox,
        # Under-predicts, and increasingly so as the wind drops.
        'mod': 0.85 * nox + 10 - 1.5 * ws + rng.normal(0, 8, n),
    })


class TestDistPlot:
    @pytest.mark.parametrize('kind', ['density', 'histogram', 'cdf'])
    def test_kinds_render(self, linked, kind):
        fig = dist_plot(linked, 'NOx', kind=kind)
        assert isinstance(fig, go.Figure) and len(fig.data) == 1

    def test_several_pollutants_overlay(self, linked):
        assert len(dist_plot(linked, ['NOx', 'NO2']).data) == 2

    def test_cdf_is_monotonic_and_reaches_one(self, linked):
        trace = dist_plot(linked, 'NOx', kind='cdf').data[0]
        y = np.array(trace.y)
        assert np.all(np.diff(y) >= 0)
        assert y[-1] == pytest.approx(1.0)
        assert np.all(np.diff(np.array(trace.x)) >= 0)

    def test_normalised_histogram_sums_to_one(self, linked):
        trace = dist_plot(linked, 'NOx', kind='histogram', normalise=True).data[0]
        assert np.sum(trace.y) == pytest.approx(1.0)

    def test_density_integrates_to_about_one(self, linked):
        """A density that does not integrate to 1 is not a density.

        The trapezoidal sum is written out rather than called: the function is
        np.trapz on numpy 1 and np.trapezoid on numpy 2, and the calendar
        extra pins numpy below 2.
        """
        trace = dist_plot(linked, 'NOx').data[0]
        x, y = np.array(trace.x), np.array(trace.y)
        area = float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x)))
        assert area == pytest.approx(1.0, abs=0.02)

    def test_log_x_drops_non_positive_values(self, rng):
        """Zero has no logarithm. Dropping is honest; substituting would
        invent a mode that is not in the data."""
        df = pd.DataFrame({'NO2': np.concatenate([
            rng.lognormal(2, 0.5, 1000), np.zeros(50), -np.ones(10)])})
        fig = dist_plot(df, 'NO2', kind='cdf', log_x=True)
        assert len(fig.data[0].x) == 1000

    def test_bad_kind_raises(self, linked):
        with pytest.raises(ValueError, match="kind must be"):
            dist_plot(linked, 'NOx', kind='violin')

    def test_missing_column_raises(self, linked):
        with pytest.raises(ValueError, match='not found'):
            dist_plot(linked, ['NOx', 'NOPE'])

    def test_all_missing_raises(self):
        df = pd.DataFrame({'NO2': [np.nan] * 20})
        with pytest.raises(ValueError, match='enough finite values'):
            dist_plot(df, 'NO2')

    def test_conditioning_gives_one_panel_per_level(self, linked):
        fig = dist_plot(linked, 'NOx', type='season')
        assert isinstance(fig, go.Figure) and len(fig.data) >= 2


class TestLinearRelation:
    def test_recovers_the_known_slope(self, linked):
        _, summary = linear_relation(linked, 'NOx', 'NO2', period='month')
        assert summary['slope'].mean() == pytest.approx(TRUE_SLOPE, rel=0.05)

    def test_one_row_per_period(self, linked):
        _, monthly = linear_relation(linked, 'NOx', 'NO2', period='month')
        _, weekly = linear_relation(linked, 'NOx', 'NO2', period='week')
        assert len(weekly) > len(monthly) > 1
        assert monthly['date'].is_monotonic_increasing

    def test_r_squared_is_a_proportion(self, linked):
        _, summary = linear_relation(linked, 'NOx', 'NO2', period='month')
        assert summary['r_squared'].between(0, 1).all()

    def test_standard_error_is_positive(self, linked):
        _, summary = linear_relation(linked, 'NOx', 'NO2', period='month')
        assert (summary['slope_se'] > 0).all()

    def test_sparse_periods_are_dropped_not_fitted(self, rng):
        """A slope from four points is not a slope; the period is dropped
        rather than fitted badly and plotted alongside the good ones."""
        dense = pd.date_range('2022-01-01', '2022-03-31 23:00', freq='h')
        sparse = pd.date_range('2022-04-01', periods=5, freq='D')
        stamps = dense.append(sparse)
        x = rng.normal(50, 10, len(stamps))
        df = pd.DataFrame({'date_time': stamps, 'x': x,
                           'y': 2 * x + rng.normal(0, 1, len(stamps))})
        _, summary = linear_relation(df, 'x', 'y', period='month',
                                     min_points=20)
        assert len(summary) == 3           # April, with five points, is gone
        assert (summary['n'] >= 20).all()
        assert summary['date'].max() < pd.Timestamp('2022-04-01')

    def test_no_intercept_forces_through_the_origin(self, linked):
        _, summary = linear_relation(linked, 'NOx', 'NO2', period='month',
                                     intercept=False)
        assert (summary['intercept'] == 0).all()

    def test_condition_gives_one_series_per_level(self, linked):
        df = linked.assign(site=np.where(np.arange(len(linked)) % 2, 'A', 'B'))
        fig, summary = linear_relation(df, 'NOx', 'NO2', period='month',
                                       condition='site')
        assert set(summary['level']) == {'A', 'B'}
        assert len(fig.data) == 4  # a band and a line per level

    def test_constant_predictor_is_skipped(self, rng):
        """A period where x never moves has no slope to estimate."""
        n = 200
        df = pd.DataFrame({
            'date_time': pd.date_range('2022-01-01', periods=n, freq='h'),
            'x': 5.0, 'y': rng.normal(0, 1, n),
        })
        with pytest.raises(ValueError, match='complete'):
            linear_relation(df, 'x', 'y', period='month')

    def test_bad_period_raises(self, linked):
        with pytest.raises(ValueError, match='period must be one of'):
            linear_relation(linked, 'NOx', 'NO2', period='fortnight')

    def test_missing_column_raises(self, linked):
        with pytest.raises(ValueError, match='not found'):
            linear_relation(linked, 'NOx', 'NOPE')


class TestRunRegression:
    def test_recovers_the_known_coefficient(self, linked):
        _, summary = run_regression(linked, y='NO2', x=['NOx', 'ws'],
                                    window=336, step=48)
        assert summary['NOx'].mean() == pytest.approx(TRUE_SLOPE, rel=0.05)

    def test_one_panel_per_predictor(self, linked):
        fig, _ = run_regression(linked, y='NO2', x=['NOx', 'ws'],
                                window=336, step=48)
        # A band and a line per predictor.
        assert len(fig.data) == 4

    def test_windows_are_labelled_at_their_centre(self, linked):
        """Labelling at an edge would shift every feature by half a window
        against the series it explains."""
        _, summary = run_regression(linked, y='NO2', x=['NOx'], window=336,
                                    step=48)
        half = pd.Timedelta(hours=336 // 2)
        assert summary['date'].min() >= linked['date_time'].min() + half * 0.9

    def test_standard_errors_accompany_every_coefficient(self, linked):
        _, summary = run_regression(linked, y='NO2', x=['NOx', 'ws'],
                                    window=336, step=48)
        for name in ('NOx', 'ws', 'intercept'):
            assert f'{name}_se' in summary
            assert (summary[f'{name}_se'] > 0).all()

    def test_standardise_rescales_without_changing_the_fit(self, linked):
        _, plain = run_regression(linked, y='NO2', x=['NOx', 'ws'],
                                  window=336, step=48)
        _, scaled = run_regression(linked, y='NO2', x=['NOx', 'ws'],
                                   window=336, step=48, standardise=True)
        # Same quality of fit, different coefficient scale.
        assert scaled['r_squared'].to_numpy() == pytest.approx(
            plain['r_squared'].to_numpy(), rel=1e-6)
        assert scaled['NOx'].mean() != pytest.approx(plain['NOx'].mean())

    def test_window_smaller_than_the_model_raises(self, linked):
        with pytest.raises(ValueError, match='window must exceed'):
            run_regression(linked, y='NO2', x=['NOx', 'ws'], window=3)

    def test_gappy_data_raises_rather_than_fitting_nothing(self, linked):
        blanked = linked.copy()
        blanked['NO2'] = np.nan
        with pytest.raises(ValueError, match='complete cases'):
            run_regression(blanked, y='NO2', x=['NOx'], window=336, step=48)

    def test_missing_column_raises(self, linked):
        with pytest.raises(ValueError, match='not found'):
            run_regression(linked, y='NO2', x=['NOPE'])


class TestConditionalEval:
    def test_finds_the_injected_bias_gradient(self, linked):
        """The model was built to under-predict more as the wind drops, so the
        bias must be more negative in the bins where the mean wind is lowest."""
        _, summary = conditional_eval(linked, variables=['ws'], bins=8)
        low_wind = summary.nsmallest(2, 'ws')['MB'].mean()
        high_wind = summary.nlargest(2, 'ws')['MB'].mean()
        assert low_wind < high_wind

    def test_one_panel_per_variable_plus_the_error(self, linked):
        fig, _ = conditional_eval(linked, variables=['ws', 'temp'], bins=8)
        assert len(fig.data) == 3

    def test_bins_carry_similar_counts(self, linked):
        """Quantile bins, not equal-width ones: the top bin of an equal-width
        split of a skewed pollutant holds almost nothing."""
        _, summary = conditional_eval(linked, bins=10)
        assert summary['n'].max() / summary['n'].min() < 1.5

    def test_every_statistic_is_reported(self, linked):
        _, summary = conditional_eval(linked, bins=8)
        for name in ('MB', 'NMB', 'MGE', 'RMSE', 'r', 'IOA'):
            assert name in summary
        assert summary['r'].between(-1, 1).all()
        assert summary['IOA'].between(-1, 1).all()
        assert (summary['MGE'] >= 0).all() and (summary['RMSE'] >= 0).all()

    def test_a_perfect_model_has_no_bias(self, linked):
        perfect = linked.assign(mod=linked['obs'])
        _, summary = conditional_eval(perfect, bins=6)
        assert summary['MB'].abs().max() == pytest.approx(0.0, abs=1e-9)
        assert summary['IOA'].min() == pytest.approx(1.0, abs=1e-9)

    def test_sparse_bins_are_dropped(self, linked):
        _, summary = conditional_eval(linked, bins=8, min_count=200)
        assert (summary['n'] >= 200).all()

    def test_bad_statistic_raises(self, linked):
        with pytest.raises(ValueError, match='statistic must be one of'):
            conditional_eval(linked, statistic='MAPE')

    def test_missing_conditioning_column_raises(self, linked):
        with pytest.raises(ValueError, match='Conditioning column'):
            conditional_eval(linked, variables=['ws', 'NOPE'])

    def test_too_few_pairs_raises(self):
        df = pd.DataFrame({'obs': [1.0, 2.0], 'mod': [1.0, 2.0]})
        with pytest.raises(ValueError, match='complete'):
            conditional_eval(df, bins=5, min_count=10)
