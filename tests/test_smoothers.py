"""Tests for the smoothers.

Each test builds a series whose true shape is known and requires the smoother
to recover it more closely than the raw data, plus the properties that
distinguish the four from each other: which bridge gaps, which are robust to
an outlier, and which preserve the level.
"""

import numpy as np
import pandas as pd
import pytest

from PythonAQ import gaussian_smooth, kz_filter, rolling_quantile, whittaker_smooth

SMOOTHERS = [gaussian_smooth, kz_filter, rolling_quantile, whittaker_smooth]


@pytest.fixture
def noisy(rng):
    """A slow sinusoid buried in noise, with a gap."""
    n = 600
    signal = 20 + 10 * np.sin(np.arange(n) / 60.0)
    values = signal + rng.normal(0, 4, n)
    values[200:230] = np.nan
    return pd.DataFrame({
        'date_time': pd.date_range('2022-01-01', periods=n, freq='h'),
        'NO2': values,
    }), signal


def added_column(before, after):
    new = [c for c in after.columns if c not in before.columns]
    assert len(new) == 1, f'expected one new column, got {new}'
    return new[0]


class TestAllSmoothers:
    @pytest.mark.parametrize('smoother', SMOOTHERS)
    def test_recovers_the_underlying_signal(self, noisy, smoother):
        df, signal = noisy
        out = smoother(df, 'NO2')
        col = added_column(df, out)
        raw_error = np.nanmean(np.abs(df['NO2'] - signal))
        smoothed_error = np.nanmean(np.abs(out[col] - signal))
        assert smoothed_error < raw_error * 0.6

    @pytest.mark.parametrize('smoother', SMOOTHERS)
    def test_row_count_and_input_are_untouched(self, noisy, smoother):
        df, _ = noisy
        original = df.copy()
        out = smoother(df, 'NO2')
        assert len(out) == len(df)
        pd.testing.assert_frame_equal(df, original)

    @pytest.mark.parametrize('smoother', SMOOTHERS)
    def test_missing_column_raises(self, noisy, smoother):
        df, _ = noisy
        with pytest.raises(ValueError, match='not found'):
            smoother(df, 'NOPE')

    @pytest.mark.parametrize('smoother', SMOOTHERS)
    def test_new_name_is_honoured(self, noisy, smoother):
        df, _ = noisy
        assert 'custom' in smoother(df, 'NO2', new_name='custom').columns

    @pytest.mark.parametrize('smoother', SMOOTHERS)
    def test_a_constant_series_is_preserved(self, smoother):
        """Whatever else a smoother does, it must not move a flat line."""
        df = pd.DataFrame({
            'date_time': pd.date_range('2022-01-01', periods=200, freq='h'),
            'NO2': 42.0,
        })
        out = smoother(df, 'NO2')
        col = added_column(df, out)
        assert out[col].dropna().to_numpy() == pytest.approx(42.0, abs=1e-6)

    @pytest.mark.parametrize('smoother', SMOOTHERS)
    def test_unsorted_input_gives_the_same_answer(self, noisy, smoother):
        """The smoothers sort by date, so shuffled rows must not change the
        result once it is put back in order."""
        df, _ = noisy
        shuffled = df.sample(frac=1.0, random_state=0)
        col = added_column(df, smoother(df, 'NO2'))
        a = smoother(df, 'NO2').sort_values('date_time')[col].to_numpy()
        b = smoother(shuffled, 'NO2').sort_values('date_time')[col].to_numpy()
        assert a == pytest.approx(b, nan_ok=True)


class TestRollingQuantile:
    def test_median_ignores_a_single_spike(self):
        """The point of a median: one extreme hour must not move it, where it
        would drag a mean of the same window."""
        values = np.full(101, 10.0)
        values[50] = 10_000.0
        df = pd.DataFrame({
            'date_time': pd.date_range('2022-01-01', periods=101, freq='h'),
            'NO2': values,
        })
        out = rolling_quantile(df, 'NO2', width=9, quantile=0.5)
        col = added_column(df, out)
        assert out[col].iloc[50] == pytest.approx(10.0)

    def test_quantile_tracks_the_requested_level(self, rng):
        df = pd.DataFrame({
            'date_time': pd.date_range('2022-01-01', periods=2000, freq='h'),
            'NO2': rng.uniform(0, 100, 2000),
        })
        low = rolling_quantile(df, 'NO2', width=50, quantile=0.1)
        high = rolling_quantile(df, 'NO2', width=50, quantile=0.9)
        assert (low[added_column(df, low)].mean()
                < high[added_column(df, high)].mean())

    @pytest.mark.parametrize('bad', [-0.1, 1.1])
    def test_quantile_outside_zero_to_one_raises(self, noisy, bad):
        df, _ = noisy
        with pytest.raises(ValueError, match='between 0 and 1'):
            rolling_quantile(df, 'NO2', quantile=bad)

    def test_bad_align_raises(self, noisy):
        df, _ = noisy
        with pytest.raises(ValueError, match='align must be'):
            rolling_quantile(df, 'NO2', align='sideways')


class TestKzFilter:
    def test_more_iterations_smooth_further(self, noisy):
        df, signal = noisy
        one = kz_filter(df, 'NO2', width=15, iterations=1, new_name='a')
        five = kz_filter(df, 'NO2', width=15, iterations=5, new_name='b')
        assert np.nanstd(np.diff(five['b'])) < np.nanstd(np.diff(one['a']))

    def test_bridges_gaps(self, noisy):
        """Documented behaviour: the output has no holes even where the input
        did, because each pass ignores missing values."""
        df, _ = noisy
        out = kz_filter(df, 'NO2')
        assert df['NO2'].isna().any()
        assert out[added_column(df, out)].notna().all()

    @pytest.mark.parametrize('kwargs', [{'width': 0}, {'iterations': 0}])
    def test_degenerate_parameters_raise(self, noisy, kwargs):
        df, _ = noisy
        with pytest.raises(ValueError, match='at least 1'):
            kz_filter(df, 'NO2', **kwargs)


class TestWhittakerSmooth:
    def test_larger_lambda_is_smoother(self, noisy):
        df, _ = noisy
        loose = whittaker_smooth(df, 'NO2', lam=10, new_name='a')
        tight = whittaker_smooth(df, 'NO2', lam=100_000, new_name='b')
        assert np.nanstd(np.diff(tight['b'])) < np.nanstd(np.diff(loose['a']))

    def test_interpolates_across_a_gap(self, noisy):
        df, _ = noisy
        out = whittaker_smooth(df, 'NO2')
        assert out[added_column(df, out)].notna().all()

    def test_order_one_tends_towards_a_constant(self, rng):
        """An order-1 penalty punishes slope, so heavy smoothing flattens a
        ramp; an order-2 penalty punishes curvature and keeps it straight."""
        n = 300
        ramp = np.linspace(0, 100, n)
        df = pd.DataFrame({
            'date_time': pd.date_range('2022-01-01', periods=n, freq='h'),
            'NO2': ramp + rng.normal(0, 1, n),
        })
        first = whittaker_smooth(df, 'NO2', lam=1e7, order=1, new_name='a')['a']
        second = whittaker_smooth(df, 'NO2', lam=1e7, order=2, new_name='b')['b']
        assert np.ptp(first) < np.ptp(second)
        assert np.ptp(second) == pytest.approx(100, rel=0.1)

    def test_all_missing_gives_all_missing(self):
        df = pd.DataFrame({
            'date_time': pd.date_range('2022-01-01', periods=50, freq='h'),
            'NO2': np.nan,
        })
        out = whittaker_smooth(df, 'NO2')
        assert out[added_column(df, out)].isna().all()

    def test_too_short_for_the_penalty_raises(self):
        df = pd.DataFrame({
            'date_time': pd.date_range('2022-01-01', periods=2, freq='h'),
            'NO2': [1.0, 2.0],
        })
        with pytest.raises(ValueError, match='order-2 penalty'):
            whittaker_smooth(df, 'NO2', order=2)

    @pytest.mark.parametrize('kwargs,match', [({'lam': 0}, 'lam must be positive'),
                                              ({'order': 0}, 'order must be')])
    def test_degenerate_parameters_raise(self, noisy, kwargs, match):
        df, _ = noisy
        with pytest.raises(ValueError, match=match):
            whittaker_smooth(df, 'NO2', **kwargs)


class TestGaussianSmooth:
    def test_does_not_dip_towards_zero_beside_a_gap(self):
        """Treating missing values as zero would pull the smoothed series down
        near every gap, which reads as a real dip. The weights are
        renormalised instead."""
        n = 400
        values = np.full(n, 50.0)
        values[190:210] = np.nan
        df = pd.DataFrame({
            'date_time': pd.date_range('2022-01-01', periods=n, freq='h'),
            'NO2': values,
        })
        out = gaussian_smooth(df, 'NO2', sigma=5)
        col = added_column(df, out)
        beside = out[col].iloc[[185, 186, 187, 212, 213, 214]]
        assert beside.to_numpy() == pytest.approx(50.0, abs=1e-6)

    def test_larger_sigma_is_smoother(self, noisy):
        df, _ = noisy
        narrow = gaussian_smooth(df, 'NO2', sigma=1, new_name='a')
        wide = gaussian_smooth(df, 'NO2', sigma=10, new_name='b')
        assert np.nanstd(np.diff(wide['b'].dropna())) < \
            np.nanstd(np.diff(narrow['a'].dropna()))

    def test_missing_stays_missing(self, noisy):
        """Unlike the Whittaker smoother, this one does not invent values."""
        df, _ = noisy
        out = gaussian_smooth(df, 'NO2')
        col = added_column(df, out)
        assert out.loc[df['NO2'].isna(), col].isna().all()

    @pytest.mark.parametrize('kwargs', [{'sigma': 0}, {'truncate': 0}])
    def test_degenerate_parameters_raise(self, noisy, kwargs):
        df, _ = noisy
        with pytest.raises(ValueError, match='must be positive'):
            gaussian_smooth(df, 'NO2', **kwargs)
