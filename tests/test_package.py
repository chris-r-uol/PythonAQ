"""Tests for the package's public interface.

The package previously had no ``__init__.py`` at all, so it resolved to an
empty namespace package and every documented import failed.
"""

import importlib

import pytest

import PythonAQ

EXPECTED_API = {
    'get_r_data', 'import_aq_meta', 'download_aurn_data', 'download_noaa_data',
    'parse_noaa_data', 'calendar', 'map_sites', 'polar_cluster',
    'polar_frequency_plot', 'polar_plot', 'pollutant_rose', 'summary_plot',
    'smooth_trend_plot', 'theil_sen_plot', 'time_plot', 'wind_rose',
    'deseason_data', 'e_sat', 'rh', 'get_period',
}


def test_package_is_a_real_package():
    assert PythonAQ.__file__ is not None
    assert PythonAQ.__file__.endswith('__init__.py')


def test_version_is_exposed():
    assert isinstance(PythonAQ.__version__, str)


def test_all_advertised_names_are_importable():
    assert EXPECTED_API <= set(PythonAQ.__all__)


@pytest.mark.parametrize('name', sorted(EXPECTED_API))
def test_each_public_name_resolves_and_is_callable(name):
    assert callable(getattr(PythonAQ, name))


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        PythonAQ.no_such_function


def test_dir_lists_the_public_api():
    assert EXPECTED_API <= set(dir(PythonAQ))


@pytest.mark.parametrize('module', [
    'data', 'utilities', 'wind_rose', 'pollutant_rose', 'polar_plot',
    'polar_cluster', 'polar_frequency', 'time_plot', 'theil_sen_plot',
    'smooth_trend_plot', 'summary_plot', 'map_sites', 'deweather_deseason',
])
def test_submodules_import_standalone(module):
    """Submodules must import via the package, not via a flat sys.path hack."""
    assert importlib.import_module(f'PythonAQ.{module}') is not None


def test_library_does_not_depend_on_streamlit():
    """A UI framework must not be required to import the analysis library."""
    import sys
    for name in list(sys.modules):
        if name == 'streamlit' or name.startswith('streamlit.'):
            del sys.modules[name]
    for module in ['PythonAQ.data', 'PythonAQ.time_plot',
                   'PythonAQ.polar_cluster', 'PythonAQ.deweather_deseason']:
        importlib.reload(importlib.import_module(module))
    assert 'streamlit' not in sys.modules
