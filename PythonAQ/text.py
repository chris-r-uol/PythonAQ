"""Automatic formatting of pollutant names and units. Port of ``quickText``.

Turns the short column names air quality data actually uses into something
readable on a plot: ``'no2'`` becomes NO<sub>2</sub>, ``'pm2.5'`` becomes
PM<sub>2.5</sub>, ``'ug/m3'`` becomes µg m<sup>-3</sup>.

The markup is Plotly's HTML subset, which is what its titles, axis labels,
legends and colourbars render.
"""

import re

__all__ = ['quick_text']

# Chemical species and common measurement columns, keyed in lower case.
_SPECIES = {
    'nox': 'NO<sub>x</sub>',
    'noxasno2': 'NO<sub>x</sub> as NO<sub>2</sub>',
    'no': 'NO',
    'no2': 'NO<sub>2</sub>',
    'o3': 'O<sub>3</sub>',
    'so2': 'SO<sub>2</sub>',
    'co': 'CO',
    'co2': 'CO<sub>2</sub>',
    'ch4': 'CH<sub>4</sub>',
    'nh3': 'NH<sub>3</sub>',
    'h2s': 'H<sub>2</sub>S',
    'hcl': 'HCl',
    'hno3': 'HNO<sub>3</sub>',
    'pm1': 'PM<sub>1</sub>',
    'pm10': 'PM<sub>10</sub>',
    'pm25': 'PM<sub>2.5</sub>',
    'pm2.5': 'PM<sub>2.5</sub>',
    'pmc': 'PM<sub>coarse</sub>',
    'bc': 'BC',
    'ec': 'EC',
    'oc': 'OC',
    'v10': 'V<sub>10</sub>',
    'v25': 'V<sub>2.5</sub>',
    'nv10': 'NV<sub>10</sub>',
    'nv25': 'NV<sub>2.5</sub>',
    # Meteorological and housekeeping columns
    'ws': 'wind speed',
    'wd': 'wind direction',
    'temp': 'temperature',
    'air_temp': 'air temperature',
    'rh': 'relative humidity',
    'dew_point': 'dew point',
    'atmospheric_pressure': 'pressure',
    'date_time': 'date',
}

# Units. Written with the negative exponents the literature uses.
_UNITS = {
    'ug/m3': 'µg m<sup>-3</sup>',
    'ug.m-3': 'µg m<sup>-3</sup>',
    'mg/m3': 'mg m<sup>-3</sup>',
    'ng/m3': 'ng m<sup>-3</sup>',
    'm/s': 'm s<sup>-1</sup>',
    'm.s-1': 'm s<sup>-1</sup>',
    'degc': '°C',
    'degrees': '°',
    'ppb': 'ppb',
    'ppm': 'ppm',
    'ppt': 'ppt',
    'hpa': 'hPa',
    'mbar': 'mbar',
}

_LOOKUP = {**_UNITS, **_SPECIES}

# Longest first, so 'noxasno2' wins over 'no' and 'pm2.5' over 'pm2'.
_PATTERN = re.compile(
    r'(?<![\w.])(' + '|'.join(
        re.escape(key) for key in sorted(_LOOKUP, key=len, reverse=True)
    ) + r')(?![\w])',
    re.IGNORECASE,
)

# Plotly renders a small HTML subset; strip the tags for plain-text output.
_TAGS = re.compile(r'</?su[bp]>')
_PLAIN = {'<sub>': '', '</sub>': '', '<sup>': '', '</sup>': ''}


def quick_text(text, auto=True, plain=False):
    """Format pollutant names and units for display.

    Parameters:
    - text (str): Label to format, e.g. a column name such as ``'no2'`` or a
      phrase such as ``'NO2 concentration (ug/m3)'``.
    - auto (bool): If False, return `text` unchanged. Lets callers expose a
      switch without branching at every call site.
    - plain (bool): Drop the markup, for contexts that do not render HTML.

    Returns:
    - str: The formatted label. Anything not recognised is left alone, so
      arbitrary titles pass through safely.

    Examples:
        >>> quick_text('no2')
        'NO<sub>2</sub>'
        >>> quick_text('PM2.5 (ug/m3)')
        'PM<sub>2.5</sub> (µg m<sup>-3</sup>)'
        >>> quick_text('Site 3 kerbside')
        'Site 3 kerbside'
    """
    if not auto or not isinstance(text, str) or not text:
        return text

    formatted = _PATTERN.sub(lambda m: _LOOKUP[m.group(0).lower()], text)
    if plain:
        for tag, replacement in _PLAIN.items():
            formatted = formatted.replace(tag, replacement)
    return formatted
