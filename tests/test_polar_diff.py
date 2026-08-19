"""Tests for polar_diff.

The substantive checks inject a known change into one wind sector and require
it back out, and pin the properties that make a difference plot readable at
all: a scale symmetric about zero, and blanks where only one period has data.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from PythonAQ import polar_diff

RESOLUTION = 70  # keep the GAM grid small; these tests fit two surfaces each


@pytest.fixture
def periods(rng):
    """Two periods differing only by an added easterly source."""
    def make(extra_east):
        n = 3500
        ws = rng.gamma(2.0, 2.0, n)
        wd = rng.uniform(0, 360, n)
        conc = (20 + 25 * np.exp(-((wd - 225) ** 2) / (2 * 35 ** 2)) * (ws / 8)
                + extra_east * np.exp(-((wd - 90) ** 2) / (2 * 30 ** 2))
                + rng.normal(0, 2, n))
        return pd.DataFrame({'ws': ws, 'wd': wd, 'NO2': conc})
    return make(0.0), make(18.0)


def sector_mean(fig, centre, half_width=30):
    """Mean of the difference surface over a wind-direction sector."""
    z = np.array(fig.data[0].z, dtype=float)
    u, v = np.array(fig.data[0].x), np.array(fig.data[0].y)
    U, V = np.meshgrid(u, v)
    bearing = np.degrees(np.arctan2(U, V)) % 360
    offset = np.abs((bearing - centre + 180) % 360 - 180)
    return np.nanmean(np.where(offset < half_width, z, np.nan))


class TestPolarDiff:
    def test_recovers_an_injected_sector(self, periods):
        """A source added from the east must show as an increase in the east
        and leave the west alone."""
        before, after = periods
        fig = polar_diff(before, after, resolution=RESOLUTION)
        assert sector_mean(fig, 90) > 5.0
        assert abs(sector_mean(fig, 270)) < 3.0

    def test_reversing_the_arguments_flips_the_sign(self, periods):
        before, after = periods
        forward = sector_mean(polar_diff(before, after, resolution=RESOLUTION), 90)
        backward = sector_mean(polar_diff(after, before, resolution=RESOLUTION), 90)
        assert forward == pytest.approx(-backward, rel=0.05)

    def test_identical_inputs_give_no_change(self, periods):
        before, _ = periods
        fig = polar_diff(before, before, resolution=RESOLUTION)
        z = np.array(fig.data[0].z, dtype=float)
        assert np.nanmax(np.abs(z)) == pytest.approx(0.0, abs=1e-9)

    def test_colour_scale_is_symmetric_about_zero(self, periods):
        """A diverging scale off-centre would put the midpoint colour somewhere
        other than 'no change', misstating the sign over part of the plot."""
        before, after = periods
        trace = polar_diff(before, after, resolution=RESOLUTION).data[0]
        assert trace.zmin == pytest.approx(-trace.zmax)
        assert trace.zmax > 0

    def test_explicit_limit_is_honoured_both_ways(self, periods):
        before, after = periods
        trace = polar_diff(before, after, limit=25, resolution=RESOLUTION).data[0]
        assert (trace.zmin, trace.zmax) == (-25, 25)

    def test_blank_where_only_one_period_has_data(self, rng):
        """A sector sampled in one period and not the other is not a change of
        unknown size; it is unmeasured, and must not be drawn."""
        n = 3000
        wd_full = rng.uniform(0, 360, n)
        wd_half = rng.uniform(0, 180, n)   # nothing from the west at all

        def frame(wd):
            ws = rng.gamma(2.0, 2.0, len(wd))
            return pd.DataFrame({'ws': ws, 'wd': wd,
                                 'NO2': 20 + rng.normal(0, 2, len(wd))})
        fig = polar_diff(frame(wd_full), frame(wd_half), resolution=RESOLUTION)
        z = np.array(fig.data[0].z, dtype=float)
        u, v = np.array(fig.data[0].x), np.array(fig.data[0].y)
        U, V = np.meshgrid(u, v)
        bearing = np.degrees(np.arctan2(U, V)) % 360
        # An annulus, not a disc: near the origin every direction converges,
        # so a low-wind-speed cell nominally in the west sits within the
        # coverage radius of easterly observations and is legitimately drawn.
        radius = np.sqrt(U ** 2 + V ** 2)
        west = ((np.abs(bearing - 270) < 40)
                & (radius > np.max(u) * 0.35) & (radius < np.max(u) * 0.8))
        assert west.any()
        assert np.isnan(z[west]).all()

    def test_both_surfaces_share_one_radius(self, rng):
        """Two periods with different wind speed distributions must still be
        drawn on one grid, or the subtraction compares different speeds."""
        def frame(scale):
            n = 3000
            return pd.DataFrame({'ws': rng.gamma(2.0, scale, n),
                                 'wd': rng.uniform(0, 360, n),
                                 'NO2': 20 + rng.normal(0, 2, n)})
        fig = polar_diff(frame(1.5), frame(3.5), resolution=RESOLUTION)
        x, y = np.array(fig.data[0].x), np.array(fig.data[0].y)
        assert x.min() == pytest.approx(y.min())
        assert x.max() == pytest.approx(y.max())

    @pytest.mark.parametrize('render', ['raster', 'contour', 'tile'])
    def test_render_modes(self, periods, render):
        before, after = periods
        fig = polar_diff(before, after, render=render, resolution=RESOLUTION)
        assert isinstance(fig, go.Figure) and len(fig.data) > 0

    def test_missing_column_names_the_offending_frame(self, periods):
        before, after = periods
        with pytest.raises(ValueError, match='after DataFrame'):
            polar_diff(before, after.drop(columns=['NO2']), resolution=RESOLUTION)
        with pytest.raises(ValueError, match='before DataFrame'):
            polar_diff(before.drop(columns=['ws']), after, resolution=RESOLUTION)

    def test_bad_render_raises(self, periods):
        before, after = periods
        with pytest.raises(ValueError, match='render must be'):
            polar_diff(before, after, render='nope')

    def test_negative_limit_raises(self, periods):
        before, after = periods
        with pytest.raises(ValueError, match='limit must be positive'):
            polar_diff(before, after, limit=-1, resolution=RESOLUTION)

    def test_title_defaults_to_the_pollutant(self, periods):
        before, after = periods
        fig = polar_diff(before, after, resolution=RESOLUTION)
        assert 'NO' in fig.layout.title.text
        custom = polar_diff(before, after, title='Lockdown', resolution=RESOLUTION)
        assert custom.layout.title.text == 'Lockdown'

    def test_auto_limit_ignores_the_extreme_edge_cells(self, periods):
        """The largest differences sit at the rim where each fit is least
        supported. Scaling to the maximum lets a handful of those cells wash
        out the interior, so the default is the 99th percentile."""
        before, after = periods
        trace = polar_diff(before, after, resolution=RESOLUTION).data[0]
        z = np.array(trace.z, dtype=float)
        assert trace.zmax < np.nanmax(np.abs(z))
        # Only a sliver may be clipped, or the scale is hiding real structure.
        drawn = np.isfinite(z).sum()
        assert 0 < (np.abs(z) > trace.zmax).sum() < 0.02 * drawn
