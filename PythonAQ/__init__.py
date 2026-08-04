"""PythonAQ - Air quality data processing and visualisation toolkit.

A Python toolkit for downloading, processing and visualising air quality and
meteorological data, inspired by the R `openair` package.

The public API is exposed directly on the package::

    from PythonAQ import import_aq_meta, wind_rose

    meta = import_aq_meta("aurn")
    fig, summary = wind_rose(df)

Most names are bound eagerly. Nine submodules share a name with the function
they export (``wind_rose``, ``polar_plot``, ...), and the import system binds
the *submodule* to that name on the package, so lazy resolution via
``__getattr__`` would never fire for them and ``PythonAQ.wind_rose`` would be a
module rather than the callable. Binding eagerly makes the function win.

``calendar`` is the exception: it needs the optional ``plotly-calplot``
dependency and does not clash with its module name (``calendar_plot``), so it
stays lazy and only raises if it is actually used.
"""

__version__ = "0.2.0"

# Data retrieval and parsing
from .data import (
    download_aurn_data,
    download_noaa_data,
    get_r_data,
    import_aq_meta,
    parse_noaa_data,
)

# Visualisation
from .map_sites import map_sites
from .polar_cluster import polar_cluster
from .polar_frequency import polar_frequency_plot
from .polar_plot import polar_plot
from .pollutant_rose import pollutant_rose
from .smooth_trend_plot import smooth_trend_plot
from .summary_plot import summary_plot
from .theil_sen_plot import theil_sen_plot
from .time_plot import time_plot
from .wind_rose import wind_rose

# Utilities
from .deweather_deseason import deseason_data
from .utilities import e_sat, get_period, rh

__all__ = [
    # Data retrieval and parsing
    "download_aurn_data",
    "download_noaa_data",
    "get_r_data",
    "import_aq_meta",
    "parse_noaa_data",
    # Visualisation
    "calendar",
    "map_sites",
    "polar_cluster",
    "polar_frequency_plot",
    "polar_plot",
    "pollutant_rose",
    "smooth_trend_plot",
    "summary_plot",
    "theil_sen_plot",
    "time_plot",
    "wind_rose",
    # Utilities
    "deseason_data",
    "e_sat",
    "get_period",
    "rh",
    "__version__",
]


def __getattr__(name):
    """Resolve ``calendar`` lazily so plotly-calplot stays optional."""
    if name == "calendar":
        try:
            from .calendar_plot import calendar
        except ImportError as exc:
            if "plotly_calplot" in str(exc):
                raise ImportError(
                    "'calendar' requires the optional dependency "
                    "'plotly-calplot'. Install it with: "
                    "pip install 'PythonAQ[calendar]'"
                ) from exc
            raise
        globals()["calendar"] = calendar  # cache
        return calendar
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
