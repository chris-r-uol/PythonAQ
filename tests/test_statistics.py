"""Tests for mod_stats and aq_stats.

The mod_stats expectations are computed independently from the formulas in the
openair R source, so these tests would catch a drift in either direction.
"""

import numpy as np
import pandas as pd
import pytest

from PythonAQ import aq_stats, mod_stats


@pytest.fixture
def paired():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mod = np.array([1.5, 2.5, 2.5, 5.0, 4.0])
    return pd.DataFrame({'obs': obs, 'mod': mod}), obs, mod


class TestModStats:
    def test_matches_openair_formulas(self, paired):
        df, obs, mod = paired
        result = mod_stats(df).iloc[0]

        residual = mod - obs
        abs_residual = np.abs(residual)
        obs_deviation = np.abs(obs - obs.mean()).sum()
        lhs, rhs = abs_residual.sum(), 2 * np.abs(obs - obs.mean()).sum()

        assert result['n'] == 5
        assert result['MB'] == pytest.approx(residual.mean())
        assert result['MGE'] == pytest.approx(abs_residual.mean())
        # NMB and NMGE are ratios of sums, not means of ratios
        assert result['NMB'] == pytest.approx(residual.sum() / obs.sum())
        assert result['NMGE'] == pytest.approx(abs_residual.sum() / obs.sum())
        assert result['RMSE'] == pytest.approx(np.sqrt(np.mean(residual ** 2)))
        assert result['COE'] == pytest.approx(1 - lhs / obs_deviation)
        assert result['IOA'] == pytest.approx(1 - lhs / rhs if lhs <= rhs else rhs / lhs - 1)

    def test_perfect_model(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mod_stats(pd.DataFrame({'obs': obs, 'mod': obs})).iloc[0]
        assert result['COE'] == pytest.approx(1.0)
        assert result['IOA'] == pytest.approx(1.0)
        assert result['r'] == pytest.approx(1.0)
        assert result['RMSE'] == pytest.approx(0.0)
        assert result['MB'] == pytest.approx(0.0)

    def test_mean_as_model_gives_zero_coe(self):
        """COE is 0 when the model is no better than the observed mean."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.full(5, obs.mean())
        assert mod_stats(pd.DataFrame({'obs': obs, 'mod': mod})).iloc[0]['COE'] == pytest.approx(0.0)

    def test_fac2_counts_within_factor_of_two(self):
        obs = np.array([10.0, 10.0, 10.0, 10.0])
        mod = np.array([5.0, 20.0, 4.9, 100.0])  # first two inside, last two outside
        assert mod_stats(pd.DataFrame({'obs': obs, 'mod': mod})).iloc[0]['FAC2'] == pytest.approx(0.5)

    def test_ioa_is_bounded(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            obs = rng.normal(50, 10, 50)
            mod = rng.normal(50, 10, 50)
            ioa = mod_stats(pd.DataFrame({'obs': obs, 'mod': mod})).iloc[0]['IOA']
            assert -1.0 <= ioa <= 1.0

    def test_nans_are_dropped_pairwise(self):
        df = pd.DataFrame({
            'obs': [1.0, 2.0, np.nan, 4.0],
            'mod': [1.0, np.nan, 3.0, 4.0],
        })
        assert mod_stats(df).iloc[0]['n'] == 2

    def test_group_by(self):
        df = pd.DataFrame({
            'site': ['A'] * 4 + ['B'] * 4,
            'obs': [1.0, 2.0, 3.0, 4.0] * 2,
            'mod': [1.0, 2.0, 3.0, 4.0] + [2.0, 4.0, 6.0, 8.0],
        })
        result = mod_stats(df, group_by='site')
        assert list(result['site']) == ['A', 'B']
        assert result.loc[result.site == 'A', 'COE'].iloc[0] == pytest.approx(1.0)
        # Site B is perfectly correlated but biased high
        assert result.loc[result.site == 'B', 'r'].iloc[0] == pytest.approx(1.0)
        assert result.loc[result.site == 'B', 'MB'].iloc[0] > 0

    def test_empty_input_returns_nan_row(self):
        df = pd.DataFrame({'obs': [np.nan], 'mod': [np.nan]})
        result = mod_stats(df).iloc[0]
        assert result['n'] == 0
        assert np.isnan(result['RMSE'])

    def test_missing_column_raises(self, paired):
        df, _, _ = paired
        with pytest.raises(ValueError, match='not found'):
            mod_stats(df, mod='nope')


class TestAqStats:
    def test_one_row_per_year(self, aq_df):
        result = aq_stats(aq_df, 'NO2')
        assert sorted(result['year']) == [2020, 2021, 2022]

    def test_expected_columns(self, aq_df):
        result = aq_stats(aq_df, 'NO2')
        for column in ['data_capture', 'mean', 'minimum', 'maximum', 'median',
                       'percentile.95', 'max_daily', 'max_rolling_8']:
            assert column in result.columns

    def test_no2_gets_an_exceedance_column(self, aq_df):
        """NO2 has a 200 ug/m3 hourly UK objective."""
        result = aq_stats(aq_df, 'NO2')
        assert 'days_hourly_gt_200' in result.columns

    def test_unknown_pollutant_has_no_exceedance_column(self, aq_df):
        result = aq_stats(aq_df.rename(columns={'NO2': 'XYZ'}), 'XYZ')
        assert not [c for c in result.columns if c.startswith('days_')]

    def test_ordering_of_summary_statistics(self, aq_df):
        row = aq_stats(aq_df, 'NO2').iloc[0]
        assert row['minimum'] <= row['median'] <= row['maximum']
        assert row['median'] <= row['percentile.95'] <= row['maximum']

    def test_data_capture_is_a_percentage(self, aq_df):
        assert (aq_stats(aq_df, 'NO2')['data_capture'].between(0, 100)).all()

    def test_transpose(self, aq_df):
        result = aq_stats(aq_df, 'NO2', transpose=True)
        assert list(result.columns) == [2020, 2021, 2022]

    def test_missing_column_raises(self, aq_df):
        with pytest.raises(ValueError, match='not found'):
            aq_stats(aq_df, 'NOPE')
