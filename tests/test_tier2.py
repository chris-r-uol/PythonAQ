"""Tests for the tier-2 openair ports.

Data utilities: date_pad, split_by_date, select_running, bin_data.
Model evaluation: conditional_quantile, taylor_diagram.
Time series: time_prop.
"""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from PythonAQ import (
    bin_data,
    conditional_quantile,
    date_pad,
    select_running,
    split_by_date,
    taylor_diagram,
    time_average,
    time_prop,
)


def assert_is_populated_figure(fig):
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


@pytest.fixture
def paired_models(rng):
    """Observations with three models of decreasing quality."""
    n = 6000
    observed = rng.gamma(3.0, 12.0, n)
    return pd.DataFrame({
        'date_time': pd.date_range('2022-01-01', periods=n, freq='h'),
        'obs': observed,
        'good': observed * 0.95 + rng.normal(0, 4, n),
        'fair': observed * 0.85 + rng.normal(0, 9, n),
        'poor': observed * 0.60 + rng.normal(0, 14, n),
    })


class TestDatePad:
    def test_fills_absent_rows(self):
        """Rows are inserted for the hours that are simply not there.

        The interval is stated: a regular three-hourly series is already
        complete as far as inference is concerned, which is the point of
        being able to say otherwise.
        """
        full = pd.date_range('2022-01-01', periods=24, freq='h')
        df = pd.DataFrame({'date_time': full[::3], 'x': 1.0})
        padded = date_pad(df, interval='hour')
        assert len(padded) == 22  # 00:00 to 21:00, the observed span
        assert padded['x'].notna().sum() == 8

    def test_inserted_rows_are_nan_not_filled(self):
        full = pd.date_range('2022-01-01', periods=10, freq='h')
        df = pd.DataFrame({'date_time': full[::2], 'x': 5.0})
        padded = date_pad(df)
        assert padded['x'].notna().sum() == 5

    def test_explicit_interval_overrides_inference(self):
        """Inference uses the modal gap, which is wrong when data is missing."""
        full = pd.date_range('2022-01-01', periods=24, freq='h')
        df = pd.DataFrame({'date_time': full[::2], 'x': 1.0})
        assert len(date_pad(df, interval='hour')) == 23

    def test_pads_each_series_over_its_own_span(self):
        """A site that started late must not gain rows before it existed."""
        early = pd.DataFrame({
            'date_time': pd.date_range('2022-01-01', periods=10, freq='h'),
            'site': 'A', 'x': 1.0,
        })
        late = pd.DataFrame({
            'date_time': pd.date_range('2022-06-01', periods=10, freq='h'),
            'site': 'B', 'x': 1.0,
        })
        padded = date_pad(pd.concat([early, late]), type='site')
        assert padded[padded['site'] == 'B']['date_time'].min() >= pd.Timestamp('2022-06-01')
        assert len(padded) == 20

    def test_already_complete_series_is_unchanged(self):
        df = pd.DataFrame({
            'date_time': pd.date_range('2022-01-01', periods=48, freq='h'),
            'x': 1.0,
        })
        assert len(date_pad(df)) == 48

    def test_missing_date_column_raises(self):
        with pytest.raises(ValueError, match='not found'):
            date_pad(pd.DataFrame({'x': [1]}), date_col='nope')


class TestTimeAverageInterval:
    """Capture is measured against a regular time base.

    Since 1.0 that base is inferred as the greatest common divisor of the
    gaps, which handles the ordinary case of an outage. One case remains
    genuinely undecidable from timestamps alone and is pinned here so the
    limit stays documented rather than discovered.
    """

    @pytest.fixture
    def half_hourly_gaps(self):
        full = pd.date_range('2022-01-01', '2022-01-02 23:00', freq='h')
        return pd.DataFrame({'date_time': full[::2], 'x': 1.0}), full

    def test_regular_decimation_is_undecidable(self, half_hourly_gaps):
        """A series with every second row absent and no gap shorter than two
        hours carries no evidence of the missing rows, so it is
        indistinguishable from a genuine two-hourly series and still reports
        full capture. No inference can fix this; only stating the interval.
        """
        df, _ = half_hourly_gaps
        result = time_average(df, 'day', data_thresh=75)
        assert result['x'].notna().all()

    def test_irregular_gaps_are_inferred_correctly(self, rng):
        """The ordinary case, and the one the 1.0 change fixes: an hourly
        series with outages of assorted lengths. Its modal gap can still be an
        hour, but a run of absent rows used to widen the inferred base and
        overstate capture. The divisor is unambiguously an hour.
        """
        full = pd.date_range('2022-01-01', '2022-01-31 23:00', freq='h')
        keep = np.ones(len(full), bool)
        for start in rng.choice(len(full), 12, replace=False):
            keep[start:start + rng.integers(3, 20)] = False
        df = pd.DataFrame({'date_time': full[keep], 'x': 1.0})

        stated = time_average(df, 'day', data_thresh=90, interval='hour')
        inferred = time_average(df, 'day', data_thresh=90)
        pd.testing.assert_frame_equal(stated, inferred)
        # Some days must actually fail, or the test proves nothing.
        assert inferred['x'].isna().any()

    def test_a_complete_series_infers_its_own_interval(self):
        full = pd.date_range('2022-01-01', periods=240, freq='h')
        df = pd.DataFrame({'date_time': full, 'x': 1.0})
        assert time_average(df, 'day', data_thresh=90)['x'].notna().all()

    def test_irregular_timestamps_warn(self):
        """A stray offset reading makes the divisor far smaller than any real
        interval, which would understate capture everywhere. Say so."""
        stamps = list(pd.date_range('2022-01-01', periods=200, freq='h'))
        stamps.append(stamps[50] + pd.Timedelta(seconds=137))
        df = pd.DataFrame({'date_time': sorted(stamps), 'x': 1.0})
        with pytest.warns(UserWarning, match='timestamps are irregular'):
            time_average(df, 'day', data_thresh=75)

    def test_no_warning_without_a_threshold(self):
        """The base is only used when capture is being measured."""
        stamps = list(pd.date_range('2022-01-01', periods=200, freq='h'))
        stamps.append(stamps[50] + pd.Timedelta(seconds=137))
        df = pd.DataFrame({'date_time': sorted(stamps), 'x': 1.0})
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            time_average(df, 'day')

    def test_explicit_interval_measures_capture_correctly(self, half_hourly_gaps):
        df, _ = half_hourly_gaps
        result = time_average(df, 'day', data_thresh=75, interval='hour')
        assert result['x'].isna().all()

    def test_date_pad_first_gives_the_same_answer(self, half_hourly_gaps):
        df, _ = half_hourly_gaps
        via_interval = time_average(df, 'day', data_thresh=75, interval='hour')
        via_padding = time_average(date_pad(df, interval='hour'), 'day',
                                   data_thresh=75)
        assert via_interval['x'].isna().all()
        assert via_padding['x'].isna().all()

    def test_interval_does_not_change_a_complete_series(self, aq_df):
        without = time_average(aq_df[['date_time', 'NO2']], 'day')
        with_it = time_average(aq_df[['date_time', 'NO2']], 'day', interval='hour')
        pd.testing.assert_frame_equal(without, with_it)


class TestSplitByDate:
    def test_two_periods_from_one_cut(self, aq_df):
        result = split_by_date(aq_df, '2021-01-01')
        assert result['split_by'].nunique() == 2

    def test_labels_are_used_and_ordered(self, aq_df):
        result = split_by_date(aq_df, '2021-01-01', labels=['before', 'after'])
        assert list(result['split_by'].cat.categories) == ['before', 'after']
        before = result[result['split_by'] == 'before']['date_time']
        assert before.max() < pd.Timestamp('2021-01-01')

    def test_multiple_cuts(self, aq_df):
        result = split_by_date(aq_df, ['2021-01-01', '2022-01-01'],
                               labels=['a', 'b', 'c'])
        assert result['split_by'].nunique() == 3

    def test_wrong_number_of_labels_raises(self, aq_df):
        with pytest.raises(ValueError, match='need 3 labels'):
            split_by_date(aq_df, ['2021-01-01', '2022-01-01'], labels=['a', 'b'])

    def test_every_row_is_assigned(self, aq_df):
        result = split_by_date(aq_df, '2021-06-01')
        assert result['split_by'].notna().all()


class TestSelectRunning:
    def test_finds_a_long_enough_run(self):
        values = [1.0] * 5 + [100.0] * 6 + [1.0] * 5
        df = pd.DataFrame({
            'date_time': pd.date_range('2022-01-01', periods=len(values), freq='h'),
            'x': values,
        })
        result = select_running(df, 'x', run_length=5, threshold=50)
        assert (result['criterion'] == 'yes').sum() == 6

    def test_ignores_a_run_that_is_too_short(self):
        values = [1.0] * 5 + [100.0] * 3 + [1.0] * 5
        df = pd.DataFrame({
            'date_time': pd.date_range('2022-01-01', periods=len(values), freq='h'),
            'x': values,
        })
        result = select_running(df, 'x', run_length=5, threshold=50)
        assert (result['criterion'] == 'yes').sum() == 0

    def test_a_gap_breaks_a_run(self):
        """Two four-hour stretches either side of a NaN are not one run of 8."""
        values = [100.0] * 4 + [np.nan] + [100.0] * 4
        df = pd.DataFrame({
            'date_time': pd.date_range('2022-01-01', periods=len(values), freq='h'),
            'x': values,
        })
        result = select_running(df, 'x', run_length=5, threshold=50)
        assert (result['criterion'] == 'yes').sum() == 0

    def test_filter_mode_returns_only_the_run(self):
        values = [1.0] * 5 + [100.0] * 6 + [1.0] * 5
        df = pd.DataFrame({
            'date_time': pd.date_range('2022-01-01', periods=len(values), freq='h'),
            'x': values,
        })
        result = select_running(df, 'x', run_length=5, threshold=50, mode='filter')
        assert len(result) == 6
        assert (result['x'] == 100.0).all()

    def test_threshold_defaults_to_the_95th_percentile(self, aq_df):
        result = select_running(aq_df, 'NO2', run_length=2)
        assert 'criterion' in result.columns

    def test_invalid_mode_raises(self, aq_df):
        with pytest.raises(ValueError, match="'flag' or 'filter'"):
            select_running(aq_df, 'NO2', mode='nonsense')


class TestBinData:
    def test_bin_count_and_columns(self, aq_df):
        result = bin_data(aq_df, 'ws', 'NO2', bins=10)
        assert len(result) == 10
        assert {'ws', 'n', 'mean', 'lower', 'upper'} <= set(result.columns)

    def test_interval_brackets_the_estimate(self, aq_df):
        result = bin_data(aq_df, 'ws', 'NO2', bins=10, random_state=0).dropna()
        assert (result['lower'] <= result['mean']).all()
        assert (result['mean'] <= result['upper']).all()

    def test_recovers_a_known_relationship(self, rng):
        x = rng.uniform(0, 10, 20000)
        y = 3.0 * x + rng.normal(0, 1, 20000)
        result = bin_data(pd.DataFrame({'x': x, 'y': y}), 'x', 'y', bins=10,
                          random_state=0)
        # The bin means should follow the line they were generated from
        assert np.allclose(result['mean'], 3.0 * result['x'], atol=0.3)

    def test_counts_sum_to_the_input(self, aq_df):
        subset = aq_df[['ws', 'NO2']].dropna()
        result = bin_data(subset, 'ws', 'NO2', bins=15)
        assert result['n'].sum() == len(subset)

    def test_is_reproducible_with_a_seed(self, aq_df):
        a = bin_data(aq_df, 'ws', 'NO2', random_state=3)
        b = bin_data(aq_df, 'ws', 'NO2', random_state=3)
        pd.testing.assert_frame_equal(a, b)


class TestConditionalQuantile:
    def test_returns_figure_and_summary(self, paired_models):
        fig, summary = conditional_quantile(paired_models, obs='obs', mod='good')
        assert_is_populated_figure(fig)
        assert {'median', 'lower_outer', 'upper_outer'} <= set(summary.columns)

    def test_quantiles_are_ordered_within_each_bin(self, paired_models):
        _, summary = conditional_quantile(paired_models, obs='obs', mod='good')
        assert (summary['lower_outer'] <= summary['lower_inner']).all()
        assert (summary['lower_inner'] <= summary['median']).all()
        assert (summary['median'] <= summary['upper_inner']).all()
        assert (summary['upper_inner'] <= summary['upper_outer']).all()

    def test_a_perfect_model_puts_the_median_inside_its_own_bin(self, rng):
        """When observed equals modelled, the observations in a bin are that
        bin's modelled values, so their median lies within the bin. Comparing
        to the exact centre would only hold for a uniform distribution."""
        values = rng.gamma(3.0, 12.0, 5000)
        df = pd.DataFrame({'obs': values, 'mod': values})
        _, summary = conditional_quantile(df, obs='obs', mod='mod', bins=10)
        half_width = np.diff(np.linspace(values.min(), values.max(), 11))[0] / 2
        assert (abs(summary['median'] - summary['mod']) <= half_width).all()

    def test_an_underpredicting_model_shows_the_median_above_the_line(self, paired_models):
        """'poor' is obs * 0.6, so observations exceed the model throughout."""
        _, summary = conditional_quantile(paired_models, obs='obs', mod='poor',
                                          bins=10)
        upper_half = summary[summary['mod'] > summary['mod'].median()]
        assert (upper_half['median'] > upper_half['mod']).all()

    def test_sparse_bins_are_dropped(self, paired_models):
        _, summary = conditional_quantile(paired_models, obs='obs', mod='good',
                                          bins=60, min_count=50)
        assert (summary['n'] >= 50).all()

    def test_too_few_pairs_raises(self, paired_models):
        with pytest.raises(ValueError, match='complete pairs'):
            conditional_quantile(paired_models.head(3), obs='obs', mod='good',
                                 min_count=10)

    def test_missing_column_raises(self, paired_models):
        with pytest.raises(ValueError, match='not found'):
            conditional_quantile(paired_models, obs='obs', mod='nope')


class TestTaylorDiagram:
    def test_returns_figure_and_summary(self, paired_models):
        fig, summary = taylor_diagram(paired_models, obs='obs',
                                      mod=['good', 'fair', 'poor'])
        assert_is_populated_figure(fig)
        assert set(summary['model']) == {'good', 'fair', 'poor'}

    def test_ranks_models_by_quality(self, paired_models):
        _, summary = taylor_diagram(paired_models, obs='obs',
                                    mod=['good', 'fair', 'poor'])
        ranked = summary.set_index('model')
        assert ranked.loc['good', 'r'] > ranked.loc['fair', 'r'] > ranked.loc['poor', 'r']
        assert (ranked.loc['good', 'centred_rmse']
                < ranked.loc['fair', 'centred_rmse']
                < ranked.loc['poor', 'centred_rmse'])

    def test_the_geometry_encodes_the_rms_error(self, paired_models):
        """The whole point of the diagram: the distance from a model's point to
        the reference point equals its centred RMS error."""
        _, summary = taylor_diagram(paired_models, obs='obs',
                                    mod=['good', 'fair', 'poor'], normalise=True)
        for row in summary.itertuples():
            angle = np.arccos(np.clip(row.r, -1, 1))
            x, y = row.sd_plot * np.cos(angle), row.sd_plot * np.sin(angle)
            distance = np.hypot(x - 1.0, y)  # reference sits at (1, 0)
            assert distance == pytest.approx(row.rmse_plot, rel=1e-6)

    def test_a_perfect_model_lands_on_the_reference(self, rng):
        values = rng.gamma(3.0, 12.0, 3000)
        df = pd.DataFrame({'obs': values, 'mod': values})
        _, summary = taylor_diagram(df, obs='obs', mod='mod')
        assert summary['r'].iloc[0] == pytest.approx(1.0)
        assert summary['sd_plot'].iloc[0] == pytest.approx(1.0)
        assert summary['centred_rmse'].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_normalise_puts_the_reference_at_one(self, paired_models):
        _, summary = taylor_diagram(paired_models, obs='obs', mod='good',
                                    normalise=True)
        assert summary['sd_plot'].iloc[0] == pytest.approx(
            summary['sd_mod'].iloc[0] / summary['sd_obs'].iloc[0]
        )

    def test_grouping(self, paired_models):
        grouped = paired_models.assign(
            season=np.where(paired_models['date_time'].dt.month < 7, 'H1', 'H2')
        )
        _, summary = taylor_diagram(grouped, obs='obs', mod='good', group='season')
        assert set(summary['group']) == {'H1', 'H2'}

    def test_bias_is_reported_separately(self, paired_models):
        """Centred RMSE removes the mean, so bias needs its own column."""
        biased = paired_models.assign(shifted=paired_models['obs'] + 20.0)
        _, summary = taylor_diagram(biased, obs='obs', mod='shifted')
        assert summary['bias'].iloc[0] == pytest.approx(20.0, abs=0.5)
        assert summary['centred_rmse'].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_missing_column_raises(self, paired_models):
        with pytest.raises(ValueError, match='not found'):
            taylor_diagram(paired_models, obs='obs', mod=['good', 'nope'])


class TestTimeProp:
    def test_returns_figure_and_summary(self, aq_df):
        fig, summary = time_prop(aq_df, 'NO2', 'wd', avg_time='month')
        assert_is_populated_figure(fig)
        assert {'share', 'value', 'count'} <= set(summary.columns)

    def test_is_a_stacked_bar_chart(self, aq_df):
        fig, _ = time_prop(aq_df, 'NO2', 'season', avg_time='month')
        assert fig.layout.barmode == 'stack'
        assert all(t.type == 'bar' for t in fig.data)

    def test_shares_sum_to_one_within_each_period(self, aq_df):
        _, summary = time_prop(aq_df, 'NO2', 'wd', avg_time='month')
        totals = summary.groupby('date_time')['share'].sum()
        assert np.allclose(totals.dropna(), 1.0)

    def test_normalise_gives_percentages(self, aq_df):
        _, summary = time_prop(aq_df, 'NO2', 'season', avg_time='year',
                               normalise=True)
        totals = summary.groupby('date_time')['value'].sum()
        assert np.allclose(totals.dropna(), 100.0)

    def test_segments_total_the_period_statistic(self, aq_df):
        """Bars must add up to the period mean, not to a sum of category means."""
        _, summary = time_prop(aq_df, 'NO2', 'wd', avg_time='year',
                               statistic='mean')
        stacked = summary.groupby('date_time')['value'].sum()
        expected = aq_df.dropna(subset=['NO2', 'wd']).groupby(
            aq_df['date_time'].dt.year
        )['NO2'].mean()
        assert np.allclose(sorted(stacked.values), sorted(expected.values), rtol=0.02)

    def test_numeric_column_splits_into_quantiles(self, aq_df):
        _, summary = time_prop(aq_df, 'NO2', 'ws', avg_time='year', n_levels=3)
        assert summary['ws'].nunique() == 3

    def test_invalid_statistic_raises(self, aq_df):
        with pytest.raises(ValueError, match="'mean' or 'sum'"):
            time_prop(aq_df, 'NO2', 'season', statistic='nonsense')

    def test_missing_column_raises(self, aq_df):
        with pytest.raises(ValueError, match='not found'):
            time_prop(aq_df, 'NOPE', 'season')
