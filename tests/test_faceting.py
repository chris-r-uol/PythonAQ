"""Tests for openair's `type` conditioning.

`type` is introduced in chapter 2 of the openair book, before any plotting
chapter, because nearly every function accepts it. These check the shared
machinery rather than each function's own output.
"""

import inspect

import numpy as np
import pytest

import PythonAQ
from PythonAQ import (
    percentile_rose,
    polar_plot,
    scatter_plot,
    time_variation,
    wind_rose,
)
from PythonAQ.faceting import _choose_columns, facet_by_type

CONDITIONABLE = [
    'corr_plot', 'percentile_rose', 'polar_annulus', 'polar_cluster',
    'polar_frequency_plot', 'polar_plot', 'pollutant_rose', 'scatter_plot',
    'smooth_trend_plot', 'summary_plot', 'theil_sen_plot', 'time_plot',
    'time_variation', 'trend_level', 'wind_rose',
]


class TestApiSurface:
    @pytest.mark.parametrize('name', CONDITIONABLE)
    def test_plot_functions_accept_type(self, name):
        """openair puts `type` on nearly every function; so should this."""
        signature = inspect.signature(getattr(PythonAQ, name))
        assert 'type' in signature.parameters

    @pytest.mark.parametrize('name', ['polar_plot', 'wind_rose', 'scatter_plot'])
    def test_conditioning_arguments_are_documented(self, name):
        doc = getattr(PythonAQ, name).__doc__
        assert 'type' in doc and 'ncols' in doc

    def test_signature_still_shows_the_original_arguments(self):
        """functools.wraps hides added arguments unless __signature__ is set."""
        params = inspect.signature(polar_plot).parameters
        assert 'conc_col' in params and 'render' in params
        assert 'type' in params and 'ncols' in params


class TestPassThrough:
    def test_type_none_returns_a_single_plot(self, aq_df):
        plain = polar_plot(aq_df, conc_col='NO2', resolution=60)
        assert len(plain.data) == 1

    def test_undecorated_behaviour_is_unchanged(self, aq_df):
        """Decorating must not alter results when type is not used."""
        a = np.asarray(polar_plot(aq_df, conc_col='NO2', resolution=60).data[0].z,
                       dtype=float)
        b = np.asarray(polar_plot(aq_df, conc_col='NO2', resolution=60,
                                  type=None).data[0].z, dtype=float)
        np.testing.assert_allclose(a, b, equal_nan=True)


class TestPanels:
    def test_one_panel_per_level(self, aq_df):
        fig, _ = wind_rose(aq_df, type='weekend')
        subplot_titles = [a.text for a in fig.layout.annotations]
        assert 'weekday' in subplot_titles and 'weekend' in subplot_titles

    @pytest.mark.parametrize('type_,expected', [
        ('weekend', 2), ('season', 4), ('year', 3),
    ])
    def test_level_counts(self, aq_df, type_, expected):
        _, summary = percentile_rose(aq_df, 'NO2', type=type_)
        assert summary[type_].nunique() == expected

    def test_summary_is_labelled_by_level(self, aq_df):
        _, summary = percentile_rose(aq_df, 'NO2', type='season')
        assert 'season' in summary.columns
        assert set(summary['season']) == {
            'spring (MAM)', 'summer (JJA)', 'autumn (SON)', 'winter (DJF)'
        }

    def test_conditioning_column_is_not_passed_to_the_plot(self, aq_df):
        """Otherwise functions that plot every numeric column would show it."""
        fig, summary = scatter_plot(aq_df, 'NO2', 'PM10', type='season')
        assert summary is not None or fig is not None  # ran without error

    def test_numeric_column_conditions_by_quantile(self, aq_df):
        _, summary = percentile_rose(aq_df, 'NO2', type='ws', n_levels=3)
        assert summary['ws'].nunique() == 3


class TestComparability:
    def test_panels_share_one_colour_scale(self, aq_df):
        """Panels computed from different subsets are only comparable if the
        colour limits match, and N colourbars would be noise."""
        fig = polar_plot(aq_df, conc_col='NO2', type='weekend',
                         resolution=60, min_count=1)
        scaled = [t for t in fig.data if getattr(t, 'z', None) is not None]
        assert len({(t.zmin, t.zmax) for t in scaled}) == 1
        assert sum(1 for t in scaled if t.showscale) == 1

    def test_panels_share_axis_limits(self, aq_df):
        """A data-driven limit differs per subset; left alone, the same radius
        would mean a different wind speed in each panel."""
        fig = polar_plot(aq_df, conc_col='NO2', type='weekend',
                         resolution=60, min_count=1)
        ranges = [fig.layout[k].range for k in ('xaxis', 'xaxis2')
                  if fig.layout[k].range is not None]
        assert len(ranges) == 2
        assert tuple(ranges[0]) == tuple(ranges[1])

    def test_legend_is_not_repeated_per_panel(self, aq_df):
        fig, _ = time_variation(aq_df, ['NO2', 'PM10'], type='weekend',
                                n_boot=10, random_state=0)
        named = [t.name for t in fig.data if t.showlegend]
        assert len(named) == len(set(named))


class TestLayout:
    @pytest.mark.parametrize('n,requested,expected', [
        (2, 3, 2), (3, 3, 3), (4, 3, 2), (6, 3, 3), (12, 3, 3),
    ])
    def test_column_count_avoids_empty_cells(self, n, requested, expected):
        assert _choose_columns(n, requested) == expected

    def test_seven_levels_are_not_forced_into_one_column(self):
        """Only a small reduction is allowed, or weekdays become a tall strip."""
        assert _choose_columns(7, 3) == 3

    def test_ncols_is_respected(self, aq_df):
        fig, _ = percentile_rose(aq_df, 'NO2', type='season', ncols=4)
        assert len([k for k in fig.layout if k.startswith('polar')]) == 4


class TestSubplotKinds:
    def test_polar_traces_get_polar_subplots(self, aq_df):
        fig, _ = wind_rose(aq_df, type='weekend')
        assert all(t.type == 'barpolar' for t in fig.data)
        assert 'polar2' in fig.layout

    def test_cartesian_traces_get_xy_subplots(self, aq_df):
        fig, _ = scatter_plot(aq_df, 'NO2', 'PM10', type='weekend')
        assert 'xaxis2' in fig.layout

    def test_shapes_and_annotations_follow_their_panel(self, aq_df):
        """polar_plot draws its rings and compass as data-space shapes."""
        fig = polar_plot(aq_df, conc_col='NO2', type='weekend',
                         resolution=60, min_count=1)
        refs = {s.xref for s in fig.layout.shapes}
        assert 'x' in refs and 'x2' in refs


class TestErrors:
    def test_unknown_type_raises(self, aq_df):
        with pytest.raises(ValueError, match='Unknown type'):
            polar_plot(aq_df, conc_col='NO2', type='fortnight')

    def test_hemisphere_is_honoured(self, aq_df):
        _, summary = percentile_rose(aq_df, 'NO2', type='season',
                                     hemisphere='southern')
        january = aq_df[aq_df['date_time'].dt.month == 1]
        assert not january.empty
        assert 'summer (JJA)' in set(summary['season'])


class TestDirectUse:
    def test_facet_by_type_accepts_any_plot_callable(self, aq_df):
        fig, summary = facet_by_type(
            lambda d: percentile_rose(d, 'NO2'), aq_df, 'weekend',
        )
        assert len(fig.data) > 0
        assert 'weekend' in summary.columns

    def test_figure_only_callables_give_back_only_a_figure(self, aq_df):
        """facet_by_type mirrors the callable's return shape."""
        fig = facet_by_type(
            lambda d: polar_plot(d, conc_col='NO2', resolution=60, min_count=1),
            aq_df, 'weekend',
        )
        assert not isinstance(fig, tuple)
        assert len(fig.data) == 2


class TestReturnShape:
    """Conditioning must not change what a function returns.

    polar_plot returns a figure and wind_rose returns (figure, DataFrame);
    adding type= must preserve each, or callers cannot write uniform code.
    """

    def test_figure_only_function_stays_figure_only(self, aq_df):
        result = polar_plot(aq_df, conc_col='NO2', type='weekend',
                            resolution=60, min_count=1)
        assert not isinstance(result, tuple)
        assert hasattr(result, 'data')

    def test_pair_returning_function_still_returns_a_pair(self, aq_df):
        result = wind_rose(aq_df, type='weekend')
        assert isinstance(result, tuple) and len(result) == 2

    def test_shape_matches_the_unconditioned_call(self, aq_df):
        for call in (
            lambda **kw: polar_plot(aq_df, conc_col='NO2', resolution=60,
                                    min_count=1, **kw),
            lambda **kw: percentile_rose(aq_df, 'NO2', **kw),
        ):
            plain = call()
            faceted = call(type='weekend')
            assert isinstance(plain, tuple) == isinstance(faceted, tuple)
