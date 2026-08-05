"""Tests for the package's public interface.

The package previously had no ``__init__.py`` at all, so it resolved to an
empty namespace package and every documented import failed.
"""

import importlib
import importlib.util

import pytest

import PythonAQ

# Public names that need an optional dependency. Everything else must resolve
# from a plain `pip install .`. Both names are recorded because they differ:
# the import name uses an underscore, while the error message quotes the PyPI
# name so it can be pasted straight into pip.
OPTIONAL_API = {
    'calendar': {'import_name': 'plotly_calplot', 'pypi_name': 'plotly-calplot'},
}

EXPECTED_API = {
    # Data retrieval and parsing
    'get_r_data', 'import_aq_meta', 'download_aurn_data', 'download_noaa_data',
    'parse_noaa_data',
    # Visualisation
    'calendar', 'corr_plot', 'map_sites', 'percentile_rose', 'polar_cluster',
    'polar_frequency_plot', 'polar_plot', 'pollutant_rose', 'scatter_plot',
    'summary_plot', 'smooth_trend_plot', 'theil_sen_plot', 'time_plot',
    'time_variation', 'trend_level', 'wind_rose',
    # Statistics
    'aq_stats', 'mod_stats',
    # Utilities
    'calc_percentile', 'cut_data', 'deseason_data', 'e_sat', 'get_period',
    'rh', 'rolling_mean', 'select_by_date', 'time_average',
}


def test_package_is_a_real_package():
    assert PythonAQ.__file__ is not None
    assert PythonAQ.__file__.endswith('__init__.py')


def test_version_is_exposed():
    assert isinstance(PythonAQ.__version__, str)


def test_version_is_a_valid_release_number():
    parts = PythonAQ.__version__.split('.')
    assert len(parts) >= 2, PythonAQ.__version__
    assert all(p.isdigit() for p in parts[:2]), PythonAQ.__version__


def test_installed_metadata_matches_the_module_version():
    """The packaged version must track ``__version__``.

    pyproject declares the version dynamically from this attribute, so the two
    cannot drift; this fails if anyone reintroduces a hardcoded literal.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version('PythonAQ')
    except PackageNotFoundError:
        pytest.skip('PythonAQ is not installed in this environment')
    assert installed == PythonAQ.__version__


def test_all_advertised_names_are_importable():
    assert EXPECTED_API <= set(PythonAQ.__all__)


def _is_installed(name):
    return importlib.util.find_spec(OPTIONAL_API[name]['import_name']) is not None


@pytest.mark.parametrize('name', sorted(EXPECTED_API))
def test_each_public_name_resolves_and_is_callable(name):
    if name in OPTIONAL_API and not _is_installed(name):
        pytest.skip(f"'{name}' needs the optional extra to be installed")
    assert callable(getattr(PythonAQ, name))


@pytest.mark.parametrize('name', sorted(OPTIONAL_API))
def test_optional_member_reports_its_missing_dependency(name):
    """A member behind an extra must say how to install it.

    The message must quote the *PyPI* name, since that is what gets pasted
    into pip, and name the extra. Importing the package as a whole must still
    succeed, so a missing extra never costs access to the rest of the API.
    """
    if _is_installed(name):
        pytest.skip('the optional dependency is installed, so nothing raises')

    pypi_name = OPTIONAL_API[name]['pypi_name']
    with pytest.raises(ImportError) as excinfo:
        getattr(PythonAQ, name)

    message = str(excinfo.value)
    assert pypi_name in message, 'the error should quote the installable name'
    assert 'pip install' in message, 'the error should give an install command'


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        PythonAQ.no_such_function


def test_dir_lists_the_public_api():
    assert EXPECTED_API <= set(dir(PythonAQ))


@pytest.mark.parametrize('module', [
    'data', 'utilities', 'wind_rose', 'pollutant_rose', 'polar_plot',
    'polar_cluster', 'polar_frequency', 'time_plot', 'theil_sen_plot',
    'smooth_trend_plot', 'summary_plot', 'map_sites', 'deweather_deseason',
    'data_utils', 'time_variation', 'percentile_rose', 'trend_level',
    'scatter_plot', 'corr_plot', 'statistics',
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
