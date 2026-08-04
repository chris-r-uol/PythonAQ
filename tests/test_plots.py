"""Smoke tests for the visualisation functions.

These assert that each function runs end to end on realistic data and returns a
Plotly figure with actual traces. They are deliberately not pixel comparisons;
the goal is to catch import errors, pandas deprecations and shape bugs.
"""

import pandas as pd
import plotly.graph_objects as go
import pytest

from PythonAQ import (
    deseason_data,
    get_period,
    map_sites,
    polar_cluster,
    polar_frequency_plot,
    polar_plot,
    pollutant_rose,
    smooth_trend_plot,
    summary_plot,
    theil_sen_plot,
    time_plot,
    wind_rose,
)


def assert_is_populated_figure(fig):
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0, 'figure has no traces'


class TestWindRose:
    @pytest.mark.parametrize('group_by', ['none', 'year', 'quartile'])
    def test_grouping_modes(self, aq_df, group_by):
        fig, summary = wind_rose(aq_df, group_by=group_by, quartile_col='NO2')
        assert_is_populated_figure(fig)
        assert isinstance(summary, pd.DataFrame)
        assert not summary.empty

    @pytest.mark.parametrize('mode', ['count', 'percentage'])
    def test_modes(self, aq_df, mode):
        fig, _ = wind_rose(aq_df, group_by='none', mode=mode)
        assert_is_populated_figure(fig)

    def test_invalid_mode_raises(self, aq_df):
        with pytest.raises(ValueError, match='Invalid mode'):
            wind_rose(aq_df, mode='nonsense')

    def test_invalid_group_by_raises(self, aq_df):
        with pytest.raises(ValueError, match='Invalid group_by'):
            wind_rose(aq_df, group_by='nonsense')

    def test_does_not_mutate_input(self, aq_df):
        before = aq_df.copy()
        wind_rose(aq_df, group_by='none')
        pd.testing.assert_frame_equal(aq_df, before)


class TestPollutantRose:
    def test_runs(self, aq_df):
        fig, summary = pollutant_rose(aq_df, pollutant='NO2')
        assert_is_populated_figure(fig)
        assert isinstance(summary, pd.DataFrame)


class TestPolarPlots:
    def test_polar_plot(self, aq_df):
        assert_is_populated_figure(polar_plot(aq_df, conc_col='NO2'))

    @pytest.mark.parametrize('separate_by_year', [True, False])
    def test_polar_frequency(self, aq_df, separate_by_year):
        fig = polar_frequency_plot(aq_df, separate_by_year=separate_by_year)
        assert_is_populated_figure(fig)

    def test_polar_cluster(self, aq_df):
        fig = polar_cluster(aq_df, feature_cols=['NO2', 'PM10'], n_clusters=4)
        assert_is_populated_figure(fig)


class TestTimePlot:
    def test_single_column(self, aq_df):
        assert_is_populated_figure(time_plot(aq_df, columns_to_plot=['NO2']))

    def test_multiple_columns_grouped(self, aq_df):
        fig = time_plot(aq_df, columns_to_plot=['NO2', 'PM10'], group_data=True)
        assert_is_populated_figure(fig)

    def test_averaging_period(self, aq_df):
        fig = time_plot(aq_df, columns_to_plot=['NO2'], averaging_period='ME')
        assert_is_populated_figure(fig)

    def test_stacked_by_year(self, aq_df):
        fig = time_plot(aq_df, columns_to_plot=['NO2'], stack_data=True,
                        averaging_period='D')
        assert_is_populated_figure(fig)

    def test_does_not_mutate_input(self, aq_df):
        before = aq_df.copy()
        time_plot(aq_df, columns_to_plot=['NO2'])
        pd.testing.assert_frame_equal(aq_df, before)


class TestTrendPlots:
    def test_theil_sen(self, aq_df):
        assert_is_populated_figure(
            theil_sen_plot(aq_df, pollutant_col='NO2', agg_freq='ME')
        )

    def test_theil_sen_with_deseason(self, aq_df):
        fig = theil_sen_plot(aq_df, pollutant_col='NO2', agg_freq='ME',
                             deseason=True)
        assert_is_populated_figure(fig)

    def test_smooth_trend(self, aq_df):
        assert_is_populated_figure(
            smooth_trend_plot(aq_df, pollutant_col='NO2', avg_freq='MS')
        )


class TestSummaryPlot:
    def test_runs(self, aq_df):
        fig, summary = summary_plot(aq_df[['date_time', 'NO2', 'PM10']])
        assert_is_populated_figure(fig)
        assert isinstance(summary, pd.DataFrame)

    def test_does_not_mutate_input(self, aq_df):
        """Regression: summary_plot wrote back into the caller's DataFrame."""
        subset = aq_df[['date_time', 'NO2', 'PM10']].copy()
        before = subset.copy()
        summary_plot(subset)
        pd.testing.assert_frame_equal(subset, before)


class TestMapSites:
    def test_runs(self):
        meta = pd.DataFrame({
            'site_id': ['LEED', 'LED6'],
            'site_name': ['Leeds Centre', 'Leeds Headingley'],
            'latitude': [53.803, 53.819],
            'longitude': [-1.546, -1.576],
        })
        assert_is_populated_figure(map_sites(meta, sites=['LEED', 'LED6']))


class TestDeseason:
    def test_adds_deseasoned_column(self, aq_df):
        result = deseason_data(
            aq_df, pollutant_column='NO2', interval='7D',
            period=get_period('7D'), method='additive',
        )
        assert 'deseasoned_NO2' in result.columns
        assert result['deseasoned_NO2'].notna().any()

    def test_reduces_seasonal_variance(self, aq_df):
        """Removing the seasonal cycle should shrink the spread."""
        result = deseason_data(
            aq_df, pollutant_column='NO2', interval='7D',
            period=get_period('7D'), method='additive',
        )
        assert result['deseasoned_NO2'].std() < result['NO2'].std()

    def test_too_short_series_raises(self, aq_df):
        short = aq_df.head(24 * 30)
        with pytest.raises(ValueError, match='at least two times the period'):
            deseason_data(short, pollutant_column='NO2', interval='7D',
                          period=get_period('7D'))


class TestCalendar:
    """`calendar` depends on the optional, and rather fragile, plotly-calplot.

    plotly-calplot 0.1.20 is not compatible with pandas 3 (its calplot() uses a
    .dt accessor on a column that pandas 3 leaves as object dtype), hence the
    'pandas<3' bound on the `calendar` extra. Skip rather than fail so the rest
    of the suite still runs on modern pandas.
    """

    def test_runs(self, aq_df):
        pytest.importorskip('plotly_calplot')
        if pd.__version__ >= '3':
            pytest.skip('plotly-calplot 0.1.20 is incompatible with pandas 3')
        from PythonAQ import calendar
        assert_is_populated_figure(calendar(aq_df, value_column='NO2'))

    def test_raises_on_missing_column(self, aq_df):
        pytest.importorskip('plotly_calplot')
        from PythonAQ import calendar
        with pytest.raises(ValueError, match='not found'):
            calendar(aq_df, value_column='NOT_A_COLUMN')
