"""Tests for the directional analysis maps.

The substantive test is geographic: a source injected from a known compass
bearing must be drawn on the correct side of its site. Everything else in these
maps is presentation, but that one property is what makes them worth having,
and it is the one a coordinate-sign error silently destroys.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from PythonAQ import (annulus_map, freq_map, percentile_rose_map, polar_map,
                      pollutant_rose_map, wind_rose_map)
from PythonAQ.maps import _auto_radius, _ground_offset, _sector

SITES = [('EAST', 53.80, -1.55, 90.0),
         ('NORTH', 53.80, -1.45, 0.0),
         ('SOUTHWEST', 53.86, -1.50, 225.0)]

ALL_MAPS = [polar_map, freq_map, wind_rose_map, pollutant_rose_map,
            percentile_rose_map, annulus_map]


@pytest.fixture
def network(rng):
    """Three sites, each with a source from a different known bearing."""
    def one(name, lat, lon, bearing):
        n = 4000
        ws = rng.gamma(2.0, 2.0, n)
        wd = rng.uniform(0, 360, n)
        offset = np.abs((wd - bearing + 180) % 360 - 180)
        return pd.DataFrame({
            'site_id': name, 'latitude': lat, 'longitude': lon,
            'date_time': pd.date_range('2022-01-01', periods=n, freq='h'),
            'ws': ws, 'wd': wd,
            'NO2': (20 + 60 * np.exp(-(offset ** 2) / (2 * 25 ** 2)) * (ws / 8)
                    + rng.normal(0, 3, n)),
        })
    return pd.concat([one(*s) for s in SITES], ignore_index=True)


def bands_of(fig):
    return [t for t in fig.data if getattr(t, 'fill', None) == 'toself']


def weighted_bearing(fig, lat0, lon0, window=0.045):
    """Bearing of the drawn colour, weighted towards the hot end.

    Each band trace is ranked by value, so weighting by rank cubed finds the
    direction the strongest colours lie in relative to the site.
    """
    north = east = 0.0
    for rank, trace in enumerate(bands_of(fig)):
        lats = np.array([v for v in trace.lat if v is not None], dtype=float)
        lons = np.array([v for v in trace.lon if v is not None], dtype=float)
        if not lats.size:
            continue
        near = (np.abs(lats - lat0) < window) & (np.abs(lons - lon0) < window)
        if not near.any():
            continue
        weight = rank ** 3
        north += weight * (lats[near] - lat0).sum() / np.cos(np.deg2rad(lat0))
        east += weight * (lons[near] - lon0).sum()
    return np.degrees(np.arctan2(east, north)) % 360


class TestGeography:
    def test_sources_are_drawn_on_the_correct_side_of_each_site(self, network):
        """The whole point of the map. A sign error here would still produce a
        plausible-looking figure, so it is asserted rather than eyeballed."""
        fig = polar_map(network, 'NO2', ws_bins=14, wd_bins=36)
        for name, lat, lon, bearing in SITES:
            got = weighted_bearing(fig, lat, lon)
            error = abs((got - bearing + 180) % 360 - 180)
            assert error < 25, f'{name}: drawn at {got:.0f}, expected {bearing}'

    def test_ground_offset_moves_the_right_way(self):
        lat, lon = _ground_offset(53.8, -1.5, north_km=10.0, east_km=0.0)
        assert lat > 53.8 and lon == pytest.approx(-1.5)
        lat, lon = _ground_offset(53.8, -1.5, north_km=0.0, east_km=10.0)
        assert lon > -1.5 and lat == pytest.approx(53.8)

    def test_longitude_degrees_shorten_towards_the_poles(self):
        """The same distance east is more degrees of longitude further north."""
        _, near_equator = _ground_offset(0.0, 0.0, 0.0, 100.0)
        _, far_north = _ground_offset(60.0, 0.0, 0.0, 100.0)
        assert far_north > near_equator * 1.9

    def test_a_sector_closes_on_itself(self):
        lats, lons = _sector(53.8, -1.5, 0.5, 1.0, 0.0, 30.0)
        assert (lats[0], lons[0]) == (lats[-1], lons[-1])
        assert len(lats) > 4

    def test_sector_bearings_are_compass_bearings(self):
        """Zero is north and ninety is east, as wind direction is reported."""
        north, _ = _sector(53.8, -1.5, 1.0, 1.0, -1.0, 1.0)
        _, east = _sector(53.8, -1.5, 1.0, 1.0, 89.0, 91.0)
        assert max(north) > 53.8
        assert max(east) > -1.5


class TestAutoRadius:
    def test_markers_do_not_overlap(self):
        """Two overlapping polar plots cannot be read at all."""
        positions = [(53.80, -1.55), (53.80, -1.45)]
        radius = _auto_radius(positions)
        separation_km = 0.10 * 111.32 * np.cos(np.deg2rad(53.8))
        assert 2 * radius < separation_km

    def test_a_lone_site_gets_a_sensible_radius(self):
        assert 0 < _auto_radius([(53.8, -1.5)]) <= 25

    def test_closer_sites_get_smaller_markers(self):
        wide = _auto_radius([(53.8, -1.9), (53.8, -1.1)])
        tight = _auto_radius([(53.8, -1.51), (53.8, -1.50)])
        assert tight < wide


class TestEveryMap:
    @pytest.mark.parametrize('fn', ALL_MAPS)
    def test_renders_with_bands_and_a_basemap(self, network, fn):
        fig = fn(network)
        assert isinstance(fig, go.Figure)
        assert bands_of(fig), 'no filled bands drawn'
        assert fig.layout.map.style == 'carto-positron'

    @pytest.mark.parametrize('fn', ALL_MAPS)
    def test_dark_basemap(self, network, fn):
        assert fn(network, map_style='carto-darkmatter').layout.map.style == \
            'carto-darkmatter'

    @pytest.mark.parametrize('fn', ALL_MAPS)
    def test_carries_a_colourbar(self, network, fn):
        fig = fn(network)
        assert any(getattr(t.marker, 'showscale', False) for t in fig.data
                   if hasattr(t, 'marker'))

    @pytest.mark.parametrize('fn', ALL_MAPS)
    def test_view_is_centred_on_the_sites(self, network, fn):
        fig = fn(network)
        assert 53.7 < fig.layout.map.center.lat < 53.95
        assert -1.7 < fig.layout.map.center.lon < -1.3
        assert 1 <= fig.layout.map.zoom <= 16

    @pytest.mark.parametrize('fn', ALL_MAPS)
    def test_missing_position_columns_raise(self, network, fn):
        with pytest.raises(ValueError, match='latitude'):
            fn(network.drop(columns=['latitude']))

    @pytest.mark.parametrize('fn', ALL_MAPS)
    def test_explicit_radius_is_honoured(self, network, fn):
        """A bigger radius must actually draw bigger markers."""
        def extent(radius):
            fig = fn(network, radius_km=radius)
            lats = [v for t in bands_of(fig) for v in t.lat if v is not None]
            return max(lats) - min(lats)
        assert extent(3.0) > extent(1.0)

    @pytest.mark.parametrize('fn', ALL_MAPS)
    def test_sites_without_coordinates_are_dropped_not_placed_at_zero(
            self, network, fn):
        broken = network.copy()
        broken.loc[broken['site_id'] == 'NORTH', ['latitude', 'longitude']] = np.nan
        fig = fn(broken)
        lats = [v for t in bands_of(fig) for v in t.lat if v is not None]
        assert min(lats) > 50, 'a site was drawn near the equator'


class TestPolarMap:
    def test_site_column_is_found_automatically(self, network):
        renamed = network.rename(columns={'site_id': 'code'})
        assert bands_of(polar_map(renamed, 'NO2', ws_bins=10, wd_bins=24))

    def test_unknown_site_column_raises(self, network):
        with pytest.raises(ValueError, match='not found'):
            polar_map(network, 'NO2', site='nope')

    def test_no_site_column_at_all_raises(self, network):
        with pytest.raises(ValueError, match='No site column'):
            polar_map(network.drop(columns=['site_id']), 'NO2')

    def test_fixed_limits_share_one_scale(self, network):
        """The reason to put several sites on one map is to compare them, so
        the same colour must mean the same concentration in every marker."""
        fig = polar_map(network, 'NO2', ws_bins=10, wd_bins=24, limits='fixed')
        bar = [t for t in fig.data if getattr(t.marker, 'showscale', False)][0]
        assert bar.marker.cmin < bar.marker.cmax

    def test_free_limits_still_render(self, network):
        assert bands_of(polar_map(network, 'NO2', ws_bins=10, wd_bins=24,
                                  limits='free'))

    def test_bad_limits_raises(self, network):
        with pytest.raises(ValueError, match="limits must be"):
            polar_map(network, 'NO2', limits='elastic')

    @pytest.mark.parametrize('n_levels', [0, 1])
    def test_too_few_levels_raises(self, network, n_levels):
        with pytest.raises(ValueError, match='n_levels must be at least 2'):
            polar_map(network, 'NO2', n_levels=n_levels)

    def test_negative_radius_raises(self, network):
        with pytest.raises(ValueError, match='radius_km must be positive'):
            polar_map(network, 'NO2', radius_km=-1)

    def test_missing_pollutant_raises(self, network):
        with pytest.raises(ValueError, match='not found'):
            polar_map(network, 'NOPE')

    def test_one_site_still_maps(self, network):
        single = network[network['site_id'] == 'EAST']
        fig = polar_map(single, 'NO2', ws_bins=10, wd_bins=24)
        assert bands_of(fig)
        assert weighted_bearing(fig, 53.80, -1.55) == pytest.approx(90, abs=30)

    def test_a_sparse_site_is_skipped_not_fatal(self, network):
        """One unusable site must not hide every good one."""
        sparse = network[network['site_id'] == 'EAST'].head(5).copy()
        sparse['site_id'] = 'SPARSE'
        sparse['latitude'] = 53.90
        fig = polar_map(pd.concat([network, sparse]), 'NO2', ws_bins=10,
                        wd_bins=24)
        assert bands_of(fig)


class TestRoseMaps:
    def test_petals_point_at_the_prevailing_wind(self, rng):
        """A rose's longest petal is the direction the wind blows from most."""
        n = 6000
        wd = np.concatenate([rng.normal(90, 12, n), rng.uniform(0, 360, n // 4)])
        df = pd.DataFrame({
            'site_id': 'A', 'latitude': 53.8, 'longitude': -1.5,
            'wd': wd % 360, 'ws': rng.gamma(2.0, 2.0, len(wd)),
            'NO2': rng.gamma(3.0, 10.0, len(wd)),
        })
        fig = wind_rose_map(df, radius_km=2.0)
        assert weighted_bearing(fig, 53.8, -1.5, window=0.06) \
            == pytest.approx(90, abs=35)

    def test_too_few_sectors_raises(self, network):
        with pytest.raises(ValueError, match='wd_bins must be at least 4'):
            wind_rose_map(network, wd_bins=3)

    def test_pollutant_rose_uses_the_pollutant_for_colour(self, network):
        fig = pollutant_rose_map(network, 'NO2')
        bar = [t for t in fig.data if getattr(t.marker, 'showscale', False)][0]
        assert bar.marker.cmax > 20


class TestPercentileAndAnnulus:
    def test_percentile_outside_range_raises(self, network):
        with pytest.raises(ValueError, match='percentile must be between'):
            percentile_rose_map(network, 'NO2', percentile=150)

    def test_a_higher_percentile_is_a_higher_scale(self, network):
        def top(p):
            fig = percentile_rose_map(network, 'NO2', percentile=p)
            bar = [t for t in fig.data
                   if getattr(t.marker, 'showscale', False)][0]
            return bar.marker.cmax
        assert top(95) > top(50)

    def test_annulus_bad_period_raises(self, network):
        with pytest.raises(ValueError, match='period must be one of'):
            annulus_map(network, 'NO2', period='fortnight')

    @pytest.mark.parametrize('period', ['hour', 'weekday', 'month', 'season'])
    def test_annulus_periods(self, network, period):
        assert bands_of(annulus_map(network, 'NO2', period=period,
                                    wd_bins=24))

    def test_annulus_needs_a_date_column(self, network):
        with pytest.raises(ValueError, match='Date column'):
            annulus_map(network.drop(columns=['date_time']), 'NO2')
