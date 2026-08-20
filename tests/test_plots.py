"""Smoke tests for the visualisation functions.

These assert that each function runs end to end on realistic data and returns a
Plotly figure with actual traces. They are deliberately not pixel comparisons;
the goal is to catch import errors, pandas deprecations and shape bugs.
"""

import numpy as np
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

    @pytest.mark.parametrize('group_by', ['none', 'year', 'quartile'])
    def test_does_not_mutate_input(self, aq_df, group_by):
        """Regression: the year and quartile branches wrote grouping columns
        back into the caller's DataFrame. Only 'none' was covered before, which
        is the one path that never writes."""
        before = aq_df.copy()
        wind_rose(aq_df, group_by=group_by, quartile_col='NO2')
        pd.testing.assert_frame_equal(aq_df, before)


class TestPollutantRose:
    def test_runs(self, aq_df):
        fig, summary = pollutant_rose(aq_df, pollutant='NO2')
        assert_is_populated_figure(fig)
        assert isinstance(summary, pd.DataFrame)


class TestPolarPlots:
    def test_polar_plot(self, aq_df):
        assert_is_populated_figure(polar_plot(aq_df, conc_col='NO2'))

    def test_default_render_is_a_continuous_surface(self, aq_df):
        """The default must be one raster trace, not thousands of polygons.

        Drawing a flat-filled polygon per bin is what made the output look
        blocky regardless of how smooth the underlying GAM was.
        """
        fig = polar_plot(aq_df, conc_col='NO2')
        assert len(fig.data) == 1
        assert fig.data[0].type == 'heatmap'
        assert fig.data[0].zsmooth == 'best'

    @pytest.mark.parametrize('render,expected_type', [
        ('raster', 'heatmap'), ('contour', 'contour'),
    ])
    def test_render_modes(self, aq_df, render, expected_type):
        fig = polar_plot(aq_df, conc_col='NO2', render=render)
        assert fig.data[0].type == expected_type

    def test_tile_render_still_available(self, aq_df):
        """The original rendering is kept for backwards compatibility."""
        fig = polar_plot(aq_df, conc_col='NO2', render='tile')
        assert len(fig.layout.shapes) > 100

    def test_tile_mode_does_not_punch_a_hole_at_north(self, aq_df):
        """Regression: the coverage mask is indexed by wind direction, which is
        circular. Morphological cleanup used to treat 0 and 360 degrees as
        opposite ends of a rectangle and erode the join, cutting a wedge out of
        the north of every tiled plot. Wind speed is not circular and is still
        allowed to erode at the rim.
        """
        from PythonAQ.polar_plot import _polar_surface, _prepare_polar_data

        data, ws_max = _prepare_polar_data(aq_df, 'ws', 'wd', 'NO2', 'auto')
        _, _, Z, _, _ = _polar_surface(data, 'ws', 'wd', 'NO2', ws_max, 16, 48,
                                       3, 10, 'tile', 300, None, None, True,
                                       None)
        populated = np.isfinite(Z).sum(axis=1)
        seam = populated[[-2, -1, 0, 1]]
        assert (seam > 0).all(), 'the north seam was erased'
        assert np.abs(seam - np.median(populated)).max() <= 2

    def test_invalid_render_raises(self, aq_df):
        with pytest.raises(ValueError, match='render must be'):
            polar_plot(aq_df, conc_col='NO2', render='nonsense')

    def test_higher_resolution_gives_a_finer_grid(self, aq_df):
        coarse = polar_plot(aq_df, conc_col='NO2', resolution=60)
        fine = polar_plot(aq_df, conc_col='NO2', resolution=150)
        assert np.asarray(fine.data[0].z).shape > np.asarray(coarse.data[0].z).shape

    def test_surface_is_masked_outside_the_wind_speed_limit(self, aq_df):
        """Corners of the square grid fall outside the circle and must be blank."""
        z = np.asarray(polar_plot(aq_df, conc_col='NO2').data[0].z, dtype=float)
        assert np.isnan(z[0, 0]) and np.isnan(z[-1, -1])
        assert np.isfinite(z).any()

    def test_compass_labels_are_inside_the_axis_range(self, aq_df):
        """Regression: labels sat beyond the axis range and were clipped."""
        fig = polar_plot(aq_df, conc_col='NO2')
        limit = fig.layout.xaxis.range[1]
        labels = {a.text.strip('<b>/'): (a.x, a.y)
                  for a in fig.layout.annotations if a.text and 'b>' in a.text}
        assert {'N', 'E', 'S', 'W'} <= set(labels)
        for name, (x, y) in labels.items():
            assert max(abs(x), abs(y)) <= limit, f'{name} label is clipped'

    def test_ws_limit_controls_the_radial_extent(self, aq_df):
        auto = polar_plot(aq_df, conc_col='NO2', ws_limit='auto')
        full = polar_plot(aq_df, conc_col='NO2', ws_limit='max')
        assert full.layout.xaxis.range[1] >= auto.layout.xaxis.range[1]

    def test_exclude_missing_blanks_an_empty_sector(self, aq_df):
        """With no easterly winds at all, the eastern side must stay blank."""
        no_easterlies = aq_df[(aq_df['wd'] < 45) | (aq_df['wd'] > 135)]
        z = np.asarray(
            polar_plot(no_easterlies, conc_col='NO2', exclude_missing=True).data[0].z,
            dtype=float,
        )
        mid = z.shape[0] // 2
        # Due east is the right-hand edge of the middle row, just inside the rim
        east = z[mid, int(z.shape[1] * 0.85)]
        assert np.isnan(east)

    def test_too_few_populated_bins_raises_clearly(self, aq_df):
        with pytest.raises(ValueError, match='too few to fit a surface'):
            polar_plot(aq_df.head(12), conc_col='NO2', min_count=5)

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
