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

__version__ = "0.3.1"

# Data retrieval and parsing
from .data import (
    download_aurn_data,
    download_noaa_data,
    get_r_data,
    import_aq_meta,
    parse_noaa_data,
)

# Visualisation
from .corr_plot import corr_plot
from .map_sites import map_sites
from .percentile_rose import percentile_rose
from .polar_annulus import polar_annulus
from .polar_cluster import polar_cluster
from .polar_frequency import polar_frequency_plot
from .polar_plot import polar_plot
from .pollutant_rose import pollutant_rose
from .scatter_plot import scatter_plot
from .smooth_trend_plot import smooth_trend_plot
from .summary_plot import summary_plot
from .theil_sen_plot import theil_sen_plot
from .time_plot import time_plot
from .time_variation import time_variation
from .trend_level import trend_level
from .wind_rose import wind_rose

# Statistics
from .statistics import aq_stats, mod_stats

# Utilities
from .data_utils import (
    calc_percentile,
    cut_data,
    rolling_mean,
    select_by_date,
    time_average,
)
from .deweather_deseason import deseason_data
from .text import quick_text
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
    "corr_plot",
    "map_sites",
    "percentile_rose",
    "polar_annulus",
    "polar_cluster",
    "polar_frequency_plot",
    "polar_plot",
    "pollutant_rose",
    "scatter_plot",
    "smooth_trend_plot",
    "summary_plot",
    "theil_sen_plot",
    "time_plot",
    "time_variation",
    "trend_level",
    "wind_rose",
    # Statistics
    "aq_stats",
    "mod_stats",
    # Utilities
    "calc_percentile",
    "cut_data",
    "deseason_data",
    "e_sat",
    "get_period",
    "quick_text",
    "rh",
    "rolling_mean",
    "select_by_date",
    "time_average",
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
