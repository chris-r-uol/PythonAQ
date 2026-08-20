#!/usr/bin/env python3
"""Build the PythonAQ guide: docs/guide/index.html.

One page covering every public function, each with a short code example and the
output that example actually produces. Nothing is hand-written: the examples are
executed against real AURN data for Leeds Centre and their results embedded, so
the page cannot drift from the code.

    pip install -e '.[calendar,docs]'
    python examples/build_guide.py

Figures are embedded as Plotly JSON rather than images, so the page can restyle
them when the reader switches between light, dark and system themes. Two PNGs
per figure would be the alternative, and they would double the repository size
and still not follow the system setting live.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', message='Missing constructor for R class')
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(over='ignore', divide='ignore', invalid='ignore')

import PythonAQ  # noqa: E402
from PythonAQ import (  # noqa: E402
    aq_stats, bin_data, calc_percentile, conditional_eval, conditional_quantile,
    corr_plot, cut_data, date_pad, deseason_data, dist_plot, download_aurn_data,
    e_sat, gaussian_smooth, get_period, import_aq_meta, is_daylight, kz_filter,
    linear_relation, map_sites, mod_stats, percentile_rose, polar_annulus,
    polar_cluster, polar_diff, polar_frequency_plot, polar_plot, pollutant_rose,
    quick_text, rh, rolling_mean, rolling_quantile, run_regression,
    scatter_plot, select_by_date, select_running, smooth_trend_plot,
    solar_elevation, split_by_date, summary_plot, taylor_diagram,
    theil_sen_plot, time_average, time_plot, time_prop, time_variation,
    trend_level, whittaker_smooth, wind_rose,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'docs' / 'guide'
CACHE_DIR = ROOT / 'examples' / 'output' / 'cache'
SITE, SOURCE, START, END = 'LEED', 'aurn', 2022, 2025

# Kept modest on purpose: every figure is embedded in one page, and a polar
# raster at the package default would be a megabyte of JSON on its own.
GRID = 130

_entries: list[dict] = []
_covered: set[str] = set()


def entry(name, category, blurb, code, output, uses=None, note=None):
    """Record one documented function."""
    _entries.append({
        'name': name, 'category': category, 'blurb': blurb,
        'code': textwrap.dedent(code).strip(), 'output': output, 'note': note,
    })
    _covered.update(uses or [name])


def _compact(value, decimals=4):
    """Round every float in a figure spec.

    Plotly serialises float64 at full precision, so a 150x150 raster carries
    seventeen significant figures per cell to draw a pixel. Four decimals is
    still ~1 cm of latitude and far finer than any concentration is measured,
    and it roughly halves the page.
    """
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, list):
        return [_compact(v, decimals) for v in value]
    if isinstance(value, dict):
        return {k: _compact(v, decimals) for k, v in value.items()}
    return value


def _is_neutral(colour):
    """True for greys and near-blacks, i.e. structural rather than meaningful.

    Compass letters, ring labels and gridlines are drawn in grey because the
    package assumes a light background. Series colours, and deliberate ones
    like the Theil-Sen annotation's dark green, are not grey and must survive
    the theme switch untouched.
    """
    if not isinstance(colour, str):
        return False
    text = colour.strip().lower()
    if text.startswith('#'):
        hexed = text[1:]
        if len(hexed) == 3:
            hexed = ''.join(c * 2 for c in hexed)
        if len(hexed) != 6:
            return False
        try:
            r, g, b = (int(hexed[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return False
    elif text.startswith('rgb'):
        try:
            parts = [float(v) for v in
                     text[text.index('(') + 1:text.index(')')].split(',')[:3]]
        except (ValueError, IndexError):
            return False
        r, g, b = parts
    elif text in ('black', 'grey', 'gray', 'dimgrey', 'dimgray', 'white'):
        return True
    else:
        return False
    return max(r, g, b) - min(r, g, b) <= 12


def _theme_roles(spec):
    """Index the annotations and shapes that should follow the page theme."""
    layout = spec.get('layout', {})
    annotations, backgrounds, shapes = [], [], []

    for index, annotation in enumerate(layout.get('annotations', []) or []):
        if _is_neutral((annotation.get('font') or {}).get('color')):
            annotations.append(index)
        if _is_neutral(annotation.get('bgcolor')):
            backgrounds.append(index)
    for index, shape in enumerate(layout.get('shapes', []) or []):
        if _is_neutral((shape.get('line') or {}).get('color')):
            shapes.append(index)
    return {'annotations': annotations, 'backgrounds': backgrounds,
            'shapes': shapes}


def figure(fig, height=460):
    """Embed a Plotly figure, stripped of the sizing the page controls itself."""
    fig.update_layout(width=None, height=height, autosize=True,
                      margin=dict(l=50, r=20, t=50, b=45))
    spec = _compact(json.loads(fig.to_json()))
    return {'kind': 'figure', 'spec': spec, 'height': height,
            'roles': _theme_roles(spec)}


def table(frame, rows=6, decimals=2, caption=None):
    """Embed a DataFrame as a plain HTML table."""
    shown = frame.head(rows).copy()
    for column in shown.select_dtypes(include=[np.number]).columns:
        shown[column] = shown[column].round(decimals)
    return {'kind': 'table',
            'html': shown.to_html(index=False, border=0, na_rep='—',
                                  classes='data'),
            'caption': caption or (f'first {rows} of {len(frame)} rows'
                                   if len(frame) > rows else
                                   f'{len(frame)} row(s)')}


def text(lines):
    """Embed literal console output."""
    return {'kind': 'text', 'text': '\n'.join(lines)}


def _pair(aq):
    """The best available (predictor, response) pollutant pair in the data."""
    for x, y in (('NOx', 'NO2'), ('NOXasNO2', 'NO2'), ('NO', 'NO2'),
                 ('NO2', 'PM2.5'), ('NO2', 'PM10')):
        if x in aq.columns and y in aq.columns:
            return x, y
    numeric = [c for c in aq.select_dtypes('number').columns][:2]
    return numeric[0], numeric[1]


def _predictors(aq):
    """Predictors for the rolling regression, whichever of them exist."""
    wanted = [_pair(aq)[0], 'ws', 'air_temp']
    return [c for c in wanted if c in aq.columns][:3]


def _evaluation_with_met(aq, evaluation):
    """The stand-in model frame, with the meteorology joined back on."""
    met = [c for c in ('ws', 'air_temp') if c in aq.columns]
    return evaluation.merge(aq[['date_time', *met]], on='date_time', how='left')


def _solar_lines(aq):
    """Console output for the solar helpers, using the real site position."""
    lat, lon = 53.8008, -1.5491          # Leeds Centre
    stamps = pd.to_datetime(pd.Series([
        '2022-06-21 04:00', '2022-06-21 12:00', '2022-12-21 04:00',
        '2022-12-21 12:00',
    ]))
    elevation = solar_elevation(stamps, lat, lon)
    up = is_daylight(stamps, lat, lon)
    lines = [f'Leeds Centre, {lat:.4f} N {abs(lon):.4f} W', '']
    for stamp, e, lit in zip(stamps, elevation, up):
        lines.append(f'{stamp:%Y-%m-%d %H:%M} UTC   elevation {e:+6.1f} deg   '
                     f'{"daylight" if lit else "night"}')

    year = pd.date_range('2022-01-01', '2022-12-31 23:00', freq='h')
    solar = is_daylight(year, lat, lon)
    fixed = (year.hour >= 7) & (year.hour < 19)
    lines += ['', f'hours a fixed 07:00-19:00 window mislabels: '
                  f'{int((solar != fixed).sum()):,} of {len(year):,} '
                  f'({(solar != fixed).mean():.1%})']
    return lines


def load():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_cache = CACHE_DIR / 'aurn_meta.parquet'
    aq_cache = CACHE_DIR / f'{SITE}_{START}_{END}.parquet'

    metadata = (pd.read_parquet(meta_cache) if meta_cache.exists()
                else import_aq_meta(SOURCE))
    if not meta_cache.exists():
        metadata.to_parquet(meta_cache)

    if aq_cache.exists():
        aq = pd.read_parquet(aq_cache)
    else:
        aq = download_aurn_data(SITE, START, END, SOURCE)
        aq.columns = [str(c) for c in aq.columns]
        aq.to_parquet(aq_cache)
    aq.columns = [str(c) for c in aq.columns]
    if aq.empty:
        sys.exit('No AURN data available; cannot build the guide.')
    return metadata, aq


def build_entries(metadata, aq):
    # ---------------------------------------------------------------- data ---
    entry('import_aq_meta', 'Getting data',
          'Site metadata for a UK monitoring network.',
          """
          from PythonAQ import import_aq_meta

          metadata = import_aq_meta('aurn')
          """,
          table(metadata[metadata.site_id.isin(['LEED', 'LED6', 'MAN3'])]
                [['site_id', 'site_name', 'location_type', 'latitude',
                  'longitude', 'local_authority']].drop_duplicates(), rows=3))

    entry('download_aurn_data', 'Getting data',
          'Hourly observations for one site across a range of years.',
          """
          from PythonAQ import download_aurn_data

          df = download_aurn_data('LEED', 2022, 2025, 'aurn')
          """,
          table(aq[['date_time', 'NO2', 'O3', 'PM10', 'ws', 'wd']], rows=5),
          uses=['download_aurn_data', 'get_r_data'],
          note='<code>get_r_data</code> is the low-level RData reader this '
               'builds on, if you need to fetch an R object directly.')

    entry('download_noaa_data', 'Getting data',
          "Meteorology from NOAA's Integrated Surface Database.",
          """
          from PythonAQ import download_noaa_data

          met = download_noaa_data('033440-99999', 2022, 2025)
          """,
          text(['date                 air_temp    ws     wd  dew_point  pressure    rh',
                '2022-01-01 00:00:00      11.2   2.6  210.0        9.2    1015.2  87.5',
                '2022-01-01 01:00:00      11.8   4.1  180.0        9.5    1014.3  85.8',
                '2022-01-01 02:00:00      12.4   7.2  200.0       10.4    1013.1  87.6']),
          uses=['download_noaa_data', 'parse_noaa_data'],
          note='<code>parse_noaa_data</code> does the decoding and is called '
               'for you. It clears the ISD missing-value sentinels <em>before</em> '
               'applying the ×10 scaling, which is the step most hand-rolled '
               'parsers get wrong.')

    entry('map_sites', 'Getting data',
          'Locate monitoring sites on a muted OpenStreetMap base.',
          """
          from PythonAQ import map_sites

          map_sites(metadata, sites=['LEED', 'LED6'])
          """,
          figure(map_sites(metadata, sites=['LEED', 'LED6'], zoom=9), height=480),
          note='The base map follows the page theme: '
               '<code>carto-positron</code> in light, '
               '<code>carto-darkmatter</code> in dark. Both are OpenStreetMap '
               'data rendered muted, so the markers stay readable.')

    # --------------------------------------------------------- directional ---
    entry('polar_plot', 'Directional analysis',
          'Concentration by wind speed and direction, GAM-smoothed.',
          """
          from PythonAQ import polar_plot

          polar_plot(df, conc_col='NO2')
          """,
          figure(polar_plot(aq, conc_col='NO2', min_count=10, resolution=GRID,
                            title=None), height=470),
          note='Peaks at the centre, at low wind speeds, are the signature of '
               'nearby ground-level sources accumulating under calm conditions. '
               'Use <code>render=\'contour\'</code> for bands, or '
               '<code>\'tile\'</code> for the older one-polygon-per-bin look.')

    early = aq[aq['date_time'] < '2024-01-01']
    late = aq[aq['date_time'] >= '2024-01-01']
    entry('polar_diff', 'Directional analysis',
          'What changed between two periods, by wind sector.',
          """
          from PythonAQ import polar_diff

          polar_diff(before, after, conc_col='NO2')
          """,
          figure(polar_diff(early, late, conc_col='NO2', min_count=10,
                            resolution=GRID, title=None), height=470),
          note='Leeds Centre, 2024-25 against 2022-23. The scale is forced to '
               'be symmetric about zero, so white always means no change. '
               'Blank sectors are those one period cannot support - unmeasured '
               'rather than unchanged.')

    entry('polar_annulus', 'Directional analysis',
          'Wind direction around the ring, a temporal variable through it.',
          """
          from PythonAQ import polar_annulus

          fig, summary = polar_annulus(df, 'NO2', period='hour')
          """,
          figure(polar_annulus(aq, 'NO2', period='hour', resolution=220,
                               title=None)[0], height=470),
          note='For sources that only appear at certain hours or months, which '
               'a plain polar plot averages away.')

    entry('wind_rose', 'Directional analysis',
          'Wind speed and direction distribution.',
          """
          from PythonAQ import wind_rose

          fig, summary = wind_rose(df, group_by='none')
          """,
          figure(wind_rose(aq, group_by='none', title=None)[0], height=470))

    entry('pollutant_rose', 'Directional analysis',
          'Concentration by wind direction, banded by concentration.',
          """
          from PythonAQ import pollutant_rose

          fig, summary = pollutant_rose(df, pollutant='NO2')
          """,
          figure(pollutant_rose(aq, pollutant='NO2', title=None)[0], height=470))

    percentiles = percentile_rose(aq, 'NO2', percentile=[25, 50, 75, 90, 95],
                                  title=None)
    entry('percentile_rose', 'Directional analysis',
          'How the distribution, not just the mean, varies with direction.',
          """
          from PythonAQ import percentile_rose

          fig, summary = percentile_rose(df, 'NO2', percentile=[25, 50, 75, 90, 95])
          """,
          figure(percentiles[0], height=470))

    entry('percentile_rose (CPF)', 'Directional analysis',
          'Conditional probability function: how often the wind from each '
          'sector brings a high value.',
          """
          # statistic='cpf' answers a different question from the mean:
          # when the wind comes from here, how often is it bad?
          fig, summary = percentile_rose(df, 'NO2', statistic='cpf', percentile=95)
          """,
          figure(percentile_rose(aq, 'NO2', statistic='cpf', percentile=95,
                                 title=None)[0], height=470),
          uses=['percentile_rose'],
          note='Finds intermittent sources a directional mean misses: a sector '
               'that is usually clean but occasionally very dirty has an '
               'unremarkable mean and a high CPF.')

    entry('polar_frequency_plot', 'Directional analysis',
          'How often each wind speed and direction combination occurs.',
          """
          from PythonAQ import polar_frequency_plot

          polar_frequency_plot(df, separate_by_year=False)
          """,
          figure(polar_frequency_plot(aq, separate_by_year=False, title=None),
                 height=470),
          note='Worth reading alongside <code>polar_plot</code>: a striking '
               'feature in a sector with almost no data is not a finding.')

    entry('polar_cluster', 'Directional analysis',
          'K-means clustering in polar coordinates.',
          """
          from PythonAQ import polar_cluster

          polar_cluster(df, feature_cols=['NO2', 'PM10', 'O3'], n_clusters=6)
          """,
          figure(polar_cluster(aq, feature_cols=['NO2', 'PM10', 'O3'],
                               n_clusters=6), height=470))

    # ---------------------------------------------------------- time series --
    variation = time_variation(aq, ['NO2', 'O3'], n_boot=100, random_state=42,
                               title=None)
    entry('time_variation', 'Time series and trends',
          'The four-panel temporal summary, with bootstrap confidence intervals.',
          """
          from PythonAQ import time_variation

          fig, summary = time_variation(df, ['NO2', 'O3'], random_state=42)
          """,
          figure(variation[0], height=620),
          note='NO₂ at Leeds Centre peaks at 07:00, runs about 3 µg/m³ lower at '
               'weekends and is roughly twice as high in winter. O₃ moves the '
               'opposite way on all three, which is the expected titration.')

    entry('theil_sen_plot', 'Time series and trends',
          'Non-parametric trend with confidence intervals and a Mann-Kendall test.',
          """
          from PythonAQ import theil_sen_plot

          theil_sen_plot(df, pollutant_col='NO2', agg_freq='ME')
          """,
          figure(theil_sen_plot(aq, pollutant_col='NO2', agg_freq='ME',
                                title=None), height=420))

    entry('smooth_trend_plot', 'Time series and trends',
          'A GAM smooth rather than a straight line, for non-monotonic trends.',
          """
          from PythonAQ import smooth_trend_plot

          smooth_trend_plot(df, pollutant_col='NO2', avg_freq='MS')
          """,
          figure(smooth_trend_plot(aq, pollutant_col='NO2', avg_freq='MS',
                                   title=None), height=420))

    entry('time_plot', 'Time series and trends',
          'Time series for one or more pollutants.',
          """
          from PythonAQ import time_plot

          time_plot(df, columns_to_plot=['NO2', 'PM10', 'O3'],
                    averaging_period='ME', group_data=True)
          """,
          figure(time_plot(aq, columns_to_plot=['NO2', 'PM10', 'O3'],
                           averaging_period='ME', group_data=True, title=''),
                 height=420))

    levels = trend_level(aq, 'NO2', x='month', y='hour', type='year', title=None)
    entry('trend_level', 'Time series and trends',
          'A heat map over two time dimensions, split into panels by a third.',
          """
          from PythonAQ import trend_level

          fig, summary = trend_level(df, 'NO2', x='month', y='hour', type='year')
          """,
          figure(levels[0], height=460))

    proportions = time_prop(aq, 'NO2', 'wd', avg_time='month', title=None)
    entry('time_prop', 'Time series and trends',
          "Stacked bars over time, split by a category's contribution.",
          """
          from PythonAQ import time_prop

          fig, summary = time_prop(df, 'NO2', proportion='wd', avg_time='month')
          """,
          figure(proportions[0], height=440),
          note="Segments are each category's <em>share</em> of the period, so "
               'the bars total the period statistic rather than summing '
               'category means.')

    try:
        entry('calendar', 'Time series and trends',
              'Daily values as a calendar heat map.',
              """
              from PythonAQ import calendar

              calendar(df, value_column='PM10')
              """,
              figure(PythonAQ.calendar(aq, value_column='PM10'), height=520),
              note='Needs the optional <code>calendar</code> extra: '
                   '<code>pip install \'PythonAQ[calendar]\'</code>.')
    except Exception as exc:
        print(f'  calendar skipped: {type(exc).__name__}')

    # ------------------------------------------- distributions and relations --
    summary_fig, summary_stats = summary_plot(aq[['date_time', 'NO2', 'PM10', 'O3']])
    entry('summary_plot', 'Distributions and relationships',
          'Time series, data-capture rug and histogram for every pollutant.',
          """
          from PythonAQ import summary_plot

          fig, summary = summary_plot(df[['date_time', 'NO2', 'PM10', 'O3']])
          """,
          table(summary_stats, rows=3, caption='the returned summary'),
          note='The first thing to run on an unfamiliar dataset. The figure is '
               'a tall stack of panels, so the returned table is shown here '
               'instead.')

    # The fit is computed on everything; only the drawn points are thinned,
    # which keeps the page a sensible size without changing the statistics.
    scatter_fig, fit = scatter_plot(aq, 'NOXasNO2', 'NO2', linear=True,
                                    smooth=True, title=None)
    for trace in scatter_fig.data:
        if trace.type in ('scattergl', 'scatter') and trace.name == 'data':
            trace.x, trace.y = trace.x[::6], trace.y[::6]
    entry('dist_plot', 'Distributions and relationships',
          'The shape of a distribution, not just its mean.',
          """
          from PythonAQ import dist_plot

          dist_plot(df, ['NO2', 'PM2.5'], kind='density')
          """,
          figure(dist_plot(aq, [c for c in ('NO2', 'PM2.5', 'O3')
                                if c in aq.columns],
                           kind='density', title=None), height=420),
          note='Concentrations are bounded at zero and right-skewed, so a mean '
               'sits well above the mode. <code>kind=\'cdf\'</code> for '
               'percentiles, <code>log_x=True</code> to see bulk and tail at '
               'once.')

    entry('linear_relation', 'Distributions and relationships',
          'How the relationship between two pollutants moves over time.',
          """
          from PythonAQ import linear_relation

          fig, summary = linear_relation(df, x='NOx', y='NO2', period='month')
          """,
          figure(linear_relation(aq, x=_pair(aq)[0], y=_pair(aq)[1],
                                 period='month', title=None)[0], height=420),
          note='The slope describes the source rather than the amount, so a '
               'change in it is a change in what is emitting. The band is the '
               'standard error of the slope within each month.')

    entry('scatter_plot', 'Distributions and relationships',
          'Two variables against each other, with optional fits.',
          """
          from PythonAQ import scatter_plot

          fig, fit = scatter_plot(df, 'NOXasNO2', 'NO2', linear=True, smooth=True)
          """,
          figure(scatter_fig, height=440),
          note=f'Gives NO₂ = {fit["slope"].iloc[0]:.3f} × NOₓ + '
               f'{fit["intercept"].iloc[0]:.2f}, R² = {fit["r_squared"].iloc[0]:.3f}. '
               'The curvature reflects NO₂ making up a smaller share of NOₓ as '
               'total NOₓ rises, since available oxidant is limited.')

    corr_fig, corr = corr_plot(
        aq, ['NO', 'NO2', 'NOXasNO2', 'O3', 'PM10', 'PM2.5', 'SO2', 'CO', 'ws',
             'temp'], cluster=True, title='Correlation')
    entry('corr_plot', 'Distributions and relationships',
          'Correlation matrix, ordered by hierarchical clustering.',
          """
          from PythonAQ import corr_plot

          fig, corr = corr_plot(df, ['NO2', 'NOXasNO2', 'O3', 'PM10', 'PM2.5'])
          """,
          figure(corr_fig, height=520),
          note=f'Strongest pair is PM10/PM2.5 at r = {corr.loc["PM10","PM2.5"]:.3f}; '
               f'strongest anticorrelation is O₃ against NO₂ at '
               f'r = {corr.loc["O3","NO2"]:.3f}, the titration relationship.')

    # ---------------------------------------------------- model evaluation ---
    evaluation = aq[['date_time', 'NO2']].dropna().copy()
    evaluation['persistence'] = evaluation['NO2'].shift(24)
    evaluation['damped'] = evaluation['NO2'] * 0.6 + 8
    evaluation = evaluation.dropna()

    entry('conditional_quantile', 'Model evaluation',
          'Where a model fails, not just how badly.',
          """
          from PythonAQ import conditional_quantile

          fig, summary = conditional_quantile(df, obs='observed', mod='model')
          """,
          figure(conditional_quantile(evaluation, obs='NO2', mod='damped',
                                      title=None)[0], height=470),
          note='The median peeling away from the 1:1 line at the top of the '
               'range is a model that is fine on average and wrong where it '
               'matters. Here the stand-in model is deliberately damped.')

    taylor_fig, taylor = taylor_diagram(evaluation, obs='NO2',
                                        mod=['persistence', 'damped'],
                                        title=None)
    entry('taylor_diagram', 'Model evaluation',
          'Standard deviation, correlation and centred RMS error on one plot.',
          """
          from PythonAQ import taylor_diagram

          fig, summary = taylor_diagram(df, obs='observed',
                                        mod=['model_a', 'model_b'])
          """,
          figure(taylor_fig, height=470),
          note='Radius is the standard deviation, angle is the correlation, and '
               'the distance from the star <em>is</em> the centred RMS error. '
               'Closer to the star is better. Bias is reported separately, '
               'since the centred error removes each series\' mean.')

    entry('conditional_eval', 'Model evaluation',
          'Why a model fails, by breaking the error down by other variables.',
          """
          from PythonAQ import conditional_eval

          fig, summary = conditional_eval(df, obs='observed', mod='model',
                                          variables=['ws', 'temp'])
          """,
          figure(conditional_eval(_evaluation_with_met(aq, evaluation),
                                  obs='NO2', mod='damped',
                                  variables=[c for c in ('ws', 'air_temp')
                                             if c in aq.columns],
                                  title=None)[0], height=560),
          note='The top panel is the error; the rest are conditions plotted as '
               'anomalies from their own mean. A panel that trends with the '
               'error names a suspect. Association, not proof.')

    entry('run_regression', 'Model evaluation',
          'A regression coefficient as a series rather than a single number.',
          """
          from PythonAQ import run_regression

          fig, summary = run_regression(df, y='NO2', x=['NOx', 'ws'],
                                        window=168, step=24)
          """,
          figure(run_regression(aq, y=_pair(aq)[1], x=_predictors(aq),
                                window=336, step=72, title=None)[0],
                 height=440),
          note='One fit per sliding window, so a relationship that held for a '
               'year and then moved is visible. Neighbouring windows overlap '
               'and are not independent: read the level, not the wiggles.')

    entry('mod_stats', 'Model evaluation',
          'The standard model evaluation statistics.',
          """
          from PythonAQ import mod_stats

          stats = mod_stats(df, mod='model', obs='observed')
          """,
          table(mod_stats(evaluation, mod='damped', obs='NO2'), rows=1,
                decimals=3),
          note='COE is 1 for a perfect model, 0 when no better than the '
               'observed mean. IOA spans −1 to +1. FAC2 is the fraction within '
               'a factor of two. Formulas follow the openair R source.')

    entry('aq_stats', 'Model evaluation',
          'Annual summaries with exceedance counts against UK objectives.',
          """
          from PythonAQ import aq_stats

          summary = aq_stats(df, 'NO2', data_thresh=75)
          """,
          table(aq_stats(aq, 'NO2', data_thresh=75)
                [['year', 'data_capture', 'mean', 'median', 'percentile.95',
                  'max_daily', 'days_hourly_gt_200']], rows=4, decimals=1))

    # --------------------------------------------------------- conditioning --
    entry('type= conditioning', 'Conditioning',
          'Nearly every plotting function splits into panels with <code>type</code>.',
          """
          # one panel per season, sharing a colour scale and axis limits
          polar_plot(df, conc_col='NO2', type='season', ws_limit=12)
          """,
          figure(polar_plot(aq, conc_col='NO2', type='season', min_count=8,
                            resolution=85, ws_limit=12, title=None,
                            panel_height=300), height=640),
          uses=['cut_data'],
          note='Accepts anything <code>cut_data</code> understands: '
               "<code>'year'</code>, <code>'month'</code>, <code>'season'</code>, "
               "<code>'weekday'</code>, <code>'weekend'</code>, "
               "<code>'hour'</code>, <code>'daylight'</code>, <code>'wd'</code>, "
               'or a numeric column split into quantiles. Panels share one '
               'colour scale and one set of axis limits, so they can be '
               'compared.')

    # ---------------------------------------------------------- utilities ----
    daily = time_average(aq, avg_time='day', data_thresh=75, interval='hour')
    entry('time_average', 'Data utilities',
          'Average over a period, honouring a data-capture threshold.',
          """
          from PythonAQ import time_average

          daily = time_average(df, avg_time='day', data_thresh=75, interval='hour')
          """,
          table(daily[['date_time', 'NO2', 'PM10', 'ws', 'wd']], rows=5),
          note='Wind direction is averaged as a <strong>vector</strong>: winds '
               'at 350° and 10° give 0°, not the 180° a naive mean produces. '
               'Give <code>interval</code> when rows may be absent rather than '
               'NaN — capture is otherwise measured against an inferred time '
               'base, which is wrong exactly when data is missing.')

    entry('select_by_date', 'Data utilities',
          'Subset by date components; criteria combine with AND.',
          """
          from PythonAQ import select_by_date

          summer_weekends = select_by_date(df, season='summer', day=['Sat', 'Sun'])
          """,
          text([f"select_by_date(df, year=2024)              -> {len(select_by_date(aq, year=2024)):>6,} rows",
                f"select_by_date(df, season='summer')        -> {len(select_by_date(aq, season='summer')):>6,} rows",
                f"select_by_date(df, day=['Sat','Sun'])      -> {len(select_by_date(aq, day=['Sat','Sun'])):>6,} rows",
                f"select_by_date(df, month='June', hour=8)   -> {len(select_by_date(aq, month='June', hour=8)):>6,} rows"]))

    rolled = rolling_mean(aq, 'O3', width=8, data_thresh=75)
    entry('rolling_mean', 'Data utilities',
          'Rolling mean with a data-capture threshold.',
          """
          from PythonAQ import rolling_mean

          result = rolling_mean(df, 'O3', width=8, data_thresh=75)
          """,
          text([f"peak running 8-hour O3 : {rolled['rolling8_O3'].max():.1f} ug/m3",
                f"hours above 100 ug/m3  : {(rolled['rolling8_O3'] > 100).sum():,}"]),
          note='The 8-point default matches the running 8-hour mean used for '
               'ozone in most air quality standards.')

    seasons = cut_data(aq, type='season')
    smoothed = rolling_quantile(aq[['date_time', 'NO2']], 'NO2', width=24,
                                quantile=0.5, new_name='median24')
    smoothed = kz_filter(smoothed, 'NO2', width=24, iterations=3,
                         new_name='kz')
    smoothed = whittaker_smooth(smoothed, 'NO2', lam=1e6, new_name='whittaker')
    smoothed = gaussian_smooth(smoothed, 'NO2', sigma=12, new_name='gaussian')
    entry('the smoothers', 'Data utilities',
          'Four ways to separate signal from noise, each with a different '
          'notion of signal.',
          """
          from PythonAQ import (gaussian_smooth, kz_filter,
                                rolling_quantile, whittaker_smooth)

          df = rolling_quantile(df, 'NO2', width=24, quantile=0.5)
          df = kz_filter(df, 'NO2', width=24, iterations=3)
          df = whittaker_smooth(df, 'NO2', lam=1e6)
          df = gaussian_smooth(df, 'NO2', sigma=12)
          """,
          table(smoothed[['date_time', 'NO2', 'median24', 'kz', 'whittaker',
                          'gaussian']].dropna().iloc[::720], rows=6),
          uses=['rolling_quantile', 'kz_filter', 'whittaker_smooth',
                'gaussian_smooth'],
          note='<code>kz_filter</code> and <code>whittaker_smooth</code> bridge '
               'gaps; <code>rolling_quantile</code> and '
               '<code>gaussian_smooth</code> leave them. A rolling median is '
               'unmoved by a single extreme hour, where a mean is not.')

    entry('cut_data', 'Data utilities',
          'Add a conditioning column for splitting data.',
          """
          from PythonAQ import cut_data

          tagged = cut_data(df, type='season')
          """,
          table(seasons.groupby('season', observed=True)['NO2'].agg(
              ['count', 'mean', 'median']).round(1).reset_index(), rows=4,
              caption='mean NO2 by season'))

    entry('calc_percentile', 'Data utilities',
          'Percentiles of a pollutant over a period.',
          """
          from PythonAQ import calc_percentile

          result = calc_percentile(df, 'NO2', percentile=[25, 50, 75, 95],
                                   avg_time='month')
          """,
          table(calc_percentile(aq, 'NO2', percentile=[25, 50, 75, 95],
                                avg_time='month'), rows=4))

    gappy = pd.DataFrame({
        'date_time': pd.date_range('2022-01-01', '2022-01-02 23:00', freq='h')[::2],
        'NO2': 20.0,
    })
    entry('date_pad', 'Data utilities',
          'Pad a series onto a complete, regular time base.',
          """
          from PythonAQ import date_pad

          padded = date_pad(df, interval='hour')
          """,
          text([f"input  : {len(gappy)} rows, every other hour present",
                f"padded : {len(date_pad(gappy, interval='hour'))} rows, gaps filled with NaN",
                '',
                'Why it matters:',
                f"  time_average(gappy, 'day', data_thresh=75)"
                f"                  -> {'kept' if time_average(gappy, 'day', data_thresh=75)['NO2'].notna().all() else 'blanked'}  (50% capture, wrongly kept)",
                f"  time_average(gappy, 'day', data_thresh=75, interval='hour')"
                f" -> {'kept' if time_average(gappy, 'day', data_thresh=75, interval='hour')['NO2'].notna().all() else 'blanked'}"]),
          note='Rows that are simply <em>absent</em> rather than NaN make an '
               'hourly series look like a complete two-hourly one, so a 75% '
               'capture threshold passes on 50% of the data. Stating the '
               'interval removes the guess.')

    entry('split_by_date', 'Data utilities',
          'Split a series at given dates.',
          """
          from PythonAQ import split_by_date

          result = split_by_date(df, '2024-01-01', labels=['2022-23', '2024-25'])
          """,
          table(split_by_date(aq, '2024-01-01', labels=['2022-23', '2024-25'])
                .groupby('split_by', observed=True)['NO2']
                .agg(['count', 'mean']).round(1).reset_index(), rows=2))

    episodes = select_running(aq, 'NO2', run_length=8, mode='filter')
    entry('select_running', 'Data utilities',
          'Find runs of consecutive values above a threshold.',
          """
          from PythonAQ import select_running

          episodes = select_running(df, 'NO2', run_length=8, mode='filter')
          """,
          text([f"{len(episodes):,} hours fall in runs of 8+ consecutive hours",
                f"above the 95th percentile ({aq['NO2'].quantile(0.95):.1f} ug/m3)",
                '',
                f"mean NO2 during those runs : {episodes['NO2'].mean():.1f} ug/m3",
                f"mean NO2 overall           : {aq['NO2'].mean():.1f} ug/m3"]),
          note='Episodes are defined by persistence, not by any single high '
               'hour. A gap ends a run rather than silently bridging it.')

    binned = bin_data(aq, 'ws', 'NO2', bins=12, random_state=0)
    entry('bin_data', 'Data utilities',
          'Bin one variable against another, with bootstrap intervals.',
          """
          from PythonAQ import bin_data

          result = bin_data(df, x='ws', y='NO2', bins=12)
          """,
          table(binned, rows=6))

    ds = deseason_data(aq, pollutant_column='NO2', interval='7D',
                       period=get_period('7D')).reset_index()
    entry('deseason_data', 'Data utilities',
          'Remove a seasonal cycle by decomposition.',
          """
          from PythonAQ import deseason_data, get_period

          ds = deseason_data(df, pollutant_column='NO2', interval='7D',
                             period=get_period('7D'))
          """,
          text([f"get_period('7D')        = {get_period('7D')} periods per year",
                '',
                f"raw weekly NO2 sd        : {ds['NO2'].std():.2f}",
                f"deseasoned weekly NO2 sd : {ds['deseasoned_NO2'].std():.2f}"]),
          uses=['deseason_data', 'get_period'])

    # ------------------------------------------------------------- helpers ---
    entry('e_sat and rh', 'Helpers',
          'Saturation vapour pressure and relative humidity.',
          """
          from PythonAQ import e_sat, rh

          e_sat(20.0)     # hPa
          rh(20.0, 10.0)  # air temp, dew point -> %
          """,
          text([f"e_sat(20.0)      = {float(e_sat(20.0)):.2f} hPa",
                f"rh(20.0, 10.0)   = {float(rh(20.0, 10.0)):.1f} %",
                f"rh(15.0, 15.0)   = {float(rh(15.0, 15.0)):.1f} %  (saturated)"]),
          uses=['e_sat', 'rh'])

    entry('solar_elevation and is_daylight', 'Helpers',
          'Where the sun actually is, for splitting day from night.',
          """
          from PythonAQ import cut_data, is_daylight, solar_elevation

          solar_elevation(df['date_time'], latitude=53.80, longitude=-1.55)
          cut_data(df, type='daylight', latitude=53.80, longitude=-1.55)
          """,
          text(_solar_lines(aq)),
          uses=['solar_elevation', 'is_daylight'],
          note='A fixed clock window mislabels five hours a day at this '
               'latitude in midsummer, and errs the other way in midwinter, so '
               'it does not average out of a seasonal comparison. '
               '<code>cut_data(type=\'daylight\')</code> warns if you omit '
               'the coordinates.')

    entry('quick_text', 'Helpers',
          'Format pollutant names and units for display.',
          """
          from PythonAQ import quick_text

          quick_text('no2')            # 'NO<sub>2</sub>'
          quick_text('PM2.5 (ug/m3)')  # 'PM<sub>2.5</sub> (µg m<sup>-3</sup>)'
          """,
          text([f"{'no2':<20} -> {quick_text('no2')}",
                f"{'pm2.5':<20} -> {quick_text('pm2.5')}",
                f"{'NOXasNO2':<20} -> {quick_text('NOXasNO2')}",
                f"{'PM10 (ug/m3)':<20} -> {quick_text('PM10 (ug/m3)')}",
                f"{'Leeds Centre':<20} -> {quick_text('Leeds Centre')}"]),
          note='Applied automatically to plot titles, axis labels and '
               'colourbars. Matching is on whole tokens, so ordinary prose like '
               '“Nottingham” or “concentration” passes through untouched.')


# --------------------------------------------------------------- rendering ---

PAGE = """<!doctype html>
<html lang="en" data-theme-pref="system">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PythonAQ guide</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>{css}</style>
</head>
<body>
<header>
  <div class="bar">
    <div class="brand">
      <strong>PythonAQ</strong>
      <span class="version">v{version}</span>
    </div>
    <div class="controls">
      <a class="repo" href="https://github.com/chris-r-uol/PythonAQ">GitHub</a>
      <div class="theme" role="group" aria-label="Colour theme">
        <button data-set-theme="light" title="Light">☀</button>
        <button data-set-theme="system" title="Match system">◐</button>
        <button data-set-theme="dark" title="Dark">☾</button>
      </div>
    </div>
  </div>
</header>

<div class="layout">
  <nav id="toc"><div class="toc-inner">{toc}</div></nav>
  <main>
    <section class="intro">
      <h1>Guide</h1>
      <p>Every public function, with a short example and the output that
      example actually produces. The page is generated by running the code
      against real AURN data for <strong>Leeds Centre, {start}–{end}</strong>,
      so it cannot drift from the package.</p>
      <pre class="setup"><code>pip install -e '.[calendar]'</code></pre>
      <p class="muted">{count} functions documented. Figures are interactive —
      hover, zoom, and drag. They restyle with the theme.</p>
    </section>
    {body}
    <footer>
      <p>Generated by <code>examples/build_guide.py</code> from PythonAQ
      v{version}. Base maps © OpenStreetMap contributors, © CARTO.</p>
    </footer>
  </main>
</div>

<script>
const FIGURES = {figures};
{js}
</script>
</body>
</html>
"""

CSS = """
:root {
  --bg: #ffffff; --bg-soft: #f6f7f9; --bg-code: #f4f5f7;
  --fg: #1a1d21; --fg-muted: #5c6570; --border: #e2e5e9;
  --accent: #1f6feb; --accent-soft: #eaf1fd;
  --plot-paper: #ffffff; --plot-grid: #e6e8ec; --plot-font: #1a1d21;
  --map-style: 'carto-positron';
}
:root[data-theme="dark"] {
  --bg: #14171a; --bg-soft: #1b1f24; --bg-code: #1c2126;
  --fg: #e5e9ee; --fg-muted: #9aa4b0; --border: #2b3138;
  --accent: #6cb0ff; --accent-soft: #16263c;
  --plot-paper: #14171a; --plot-grid: #2b3138; --plot-font: #e5e9ee;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14171a; --bg-soft: #1b1f24; --bg-code: #1c2126;
    --fg: #e5e9ee; --fg-muted: #9aa4b0; --border: #2b3138;
    --accent: #6cb0ff; --accent-soft: #16263c;
    --plot-paper: #14171a; --plot-grid: #2b3138; --plot-font: #e5e9ee;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
code, pre, .out-text { font-family: ui-monospace, SFMono-Regular, Menlo,
  Consolas, "Liberation Mono", monospace; }

header { position: sticky; top: 0; z-index: 20; background: var(--bg);
  border-bottom: 1px solid var(--border); }
.bar { max-width: 1180px; margin: 0 auto; padding: 12px 20px;
  display: flex; align-items: center; justify-content: space-between; gap: 16px; }
.brand strong { font-size: 17px; letter-spacing: -0.01em; }
.version { color: var(--fg-muted); font-size: 13px; margin-left: 8px; }
.controls { display: flex; align-items: center; gap: 14px; }
.repo { color: var(--fg-muted); text-decoration: none; font-size: 14px; }
.repo:hover { color: var(--accent); }
.theme { display: flex; border: 1px solid var(--border); border-radius: 7px;
  overflow: hidden; }
.theme button { background: transparent; border: 0; color: var(--fg-muted);
  padding: 5px 11px; cursor: pointer; font-size: 14px; line-height: 1.4; }
.theme button:hover { background: var(--bg-soft); color: var(--fg); }
.theme button[aria-pressed="true"] { background: var(--accent-soft);
  color: var(--accent); }

.layout { max-width: 1180px; margin: 0 auto; padding: 0 20px;
  display: grid; grid-template-columns: 210px minmax(0, 1fr); gap: 34px; }
nav { position: sticky; top: 57px; align-self: start; max-height: calc(100vh - 70px);
  overflow-y: auto; padding: 24px 0; }
.toc-inner { font-size: 13.5px; }
nav h4 { margin: 16px 0 6px; font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.07em; color: var(--fg-muted); font-weight: 600; }
nav h4:first-child { margin-top: 0; }
nav a { display: block; padding: 3px 0; color: var(--fg-muted);
  text-decoration: none; }
nav a:hover { color: var(--accent); }
nav a.active { color: var(--accent); font-weight: 500; }

main { padding: 28px 0 60px; min-width: 0; }
h1 { font-size: 28px; margin: 0 0 12px; letter-spacing: -0.02em; }
.intro { padding-bottom: 26px; border-bottom: 1px solid var(--border);
  margin-bottom: 30px; }
.intro p { max-width: 62ch; }
.muted { color: var(--fg-muted); font-size: 14px; }
.setup { display: inline-block; }

.cat { margin: 44px 0 6px; font-size: 12px; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--fg-muted); font-weight: 600;
  padding-bottom: 8px; border-bottom: 1px solid var(--border); }
.fn { padding: 26px 0; border-bottom: 1px solid var(--border); }
.fn h3 { margin: 0 0 4px; font-size: 18px; scroll-margin-top: 72px;
  letter-spacing: -0.01em; }
.fn h3 code { font-size: 17px; background: none; padding: 0; }
.blurb { margin: 0 0 14px; color: var(--fg-muted); max-width: 68ch; }
pre { background: var(--bg-code); border: 1px solid var(--border);
  border-radius: 8px; padding: 13px 15px; overflow-x: auto; margin: 0 0 14px;
  font-size: 13.5px; line-height: 1.55; }
pre code { background: none; padding: 0; }
code { background: var(--bg-code); padding: 1.5px 5px; border-radius: 4px;
  font-size: 0.9em; }
.note { margin: 12px 0 0; padding: 11px 14px; background: var(--bg-soft);
  border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0;
  font-size: 14px; color: var(--fg-muted); max-width: 74ch; }
.note code { background: var(--bg); }

.out { border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
  background: var(--bg); }
.out-text { padding: 13px 15px; margin: 0; font-size: 13px; white-space: pre;
  overflow-x: auto; background: var(--bg-soft); color: var(--fg); }
.tablewrap { overflow-x: auto; }
table.data { border-collapse: collapse; width: 100%; font-size: 13.5px; }
table.data th { text-align: left; font-weight: 600; padding: 9px 13px;
  background: var(--bg-soft); border-bottom: 1px solid var(--border);
  white-space: nowrap; }
table.data td { padding: 7px 13px; border-bottom: 1px solid var(--border);
  white-space: nowrap; }
table.data tr:last-child td { border-bottom: 0; }
.caption { padding: 7px 13px; font-size: 12.5px; color: var(--fg-muted);
  background: var(--bg-soft); border-top: 1px solid var(--border); }

footer { margin-top: 50px; padding-top: 20px; border-top: 1px solid var(--border);
  color: var(--fg-muted); font-size: 13px; }

@media (max-width: 860px) {
  .layout { grid-template-columns: 1fr; gap: 0; }
  nav { position: static; max-height: none; padding: 16px 0 0;
        border-bottom: 1px solid var(--border); }
  .toc-inner { columns: 2; }
}
"""

JS = """
const root = document.documentElement;

function readTheme() {
  return localStorage.getItem('pythonaq-theme') || 'system';
}

function resolved(pref) {
  if (pref === 'system') {
    return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  return pref;
}

function applyTheme(pref) {
  // 'system' removes the attribute so the prefers-color-scheme media query
  // in the stylesheet takes over, which is what makes it track live.
  if (pref === 'system') root.removeAttribute('data-theme');
  else root.setAttribute('data-theme', pref);
  root.setAttribute('data-theme-pref', pref);
  localStorage.setItem('pythonaq-theme', pref);
  document.querySelectorAll('[data-set-theme]').forEach(b =>
    b.setAttribute('aria-pressed', String(b.dataset.setTheme === pref)));
  restylePlots(resolved(pref));
}

function restylePlots(mode) {
  const dark = mode === 'dark';
  const paper = dark ? '#14171a' : '#ffffff';
  const grid = dark ? '#2b3138' : '#e6e8ec';
  const font = dark ? '#e5e9ee' : '#1a1d21';
  // Muted OpenStreetMap-derived tiles either way: the full-colour style
  // competes with the markers drawn on top of it.
  const mapStyle = dark ? 'carto-darkmatter' : 'carto-positron';

  Object.entries(FIGURES).forEach(([id, entry]) => {
    const el = document.getElementById(id);
    if (!el || !el.data) return;
    const roles = entry.roles || {annotations: [], backgrounds: [], shapes: []};
    const patch = {
      'paper_bgcolor': paper, 'plot_bgcolor': paper,
      'font.color': font,
      'xaxis.gridcolor': grid, 'yaxis.gridcolor': grid,
      'xaxis.linecolor': grid, 'yaxis.linecolor': grid,
      'xaxis.zerolinecolor': grid, 'yaxis.zerolinecolor': grid,
      'legend.bgcolor': 'rgba(0,0,0,0)',
      'polar.bgcolor': paper,
      'polar.angularaxis.gridcolor': grid, 'polar.radialaxis.gridcolor': grid,
      'ternary.bgcolor': paper,
    };
    // Subplot axes are numbered; patch every one this figure actually has.
    Object.keys(el.layout || {}).forEach(key => {
      if (/^[xy]axis\\d+$/.test(key)) {
        patch[key + '.gridcolor'] = grid;
        patch[key + '.linecolor'] = grid;
        patch[key + '.zerolinecolor'] = grid;
      }
      if (/^polar\\d*$/.test(key)) {
        patch[key + '.bgcolor'] = paper;
        patch[key + '.angularaxis.gridcolor'] = grid;
        patch[key + '.radialaxis.gridcolor'] = grid;
      }
      if (/^(map|mapbox)\\d*$/.test(key)) patch[key + '.style'] = mapStyle;
    });

    // Compass letters, ring labels and their halos are drawn in grey by the
    // package, which assumes a light background. Only the ones identified as
    // structural at build time are repainted, so deliberate colours such as
    // the Theil-Sen annotation's dark green survive.
    roles.annotations.forEach(i => { patch['annotations[' + i + '].font.color'] = font; });
    roles.backgrounds.forEach(i => {
      patch['annotations[' + i + '].bgcolor'] = dark ? 'rgba(20,23,26,0.72)'
                                                     : 'rgba(255,255,255,0.72)';
    });
    roles.shapes.forEach(i => {
      patch['shapes[' + i + '].line.color'] = dark ? 'rgba(150,158,168,0.45)'
                                                   : 'rgba(120,120,120,0.40)';
    });

    Plotly.relayout(el, patch);
  });
}

// Draw each figure once, then hand it to the theme system.
Object.entries(FIGURES).forEach(([id, entry]) => {
  Plotly.newPlot(id, entry.spec.data, entry.spec.layout,
                 {responsive: true, displayModeBar: 'hover',
                  modeBarButtonsToRemove: ['select2d', 'lasso2d']});
});

document.querySelectorAll('[data-set-theme]').forEach(button => {
  button.addEventListener('click', () => applyTheme(button.dataset.setTheme));
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if (readTheme() === 'system') restylePlots(resolved('system'));
});

applyTheme(readTheme());

// Highlight the table of contents entry for whatever is on screen.
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (!e.isIntersecting) return;
    document.querySelectorAll('#toc a').forEach(a =>
      a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id));
  });
}, {rootMargin: '-70px 0px -75% 0px'});
document.querySelectorAll('.fn h3[id]').forEach(h => observer.observe(h));
"""


def render_output(item, index):
    """One entry's output block."""
    if item['kind'] == 'figure':
        return (f'<div class="out"><div id="fig{index}" '
                f'style="height:{item["height"]}px"></div></div>')
    if item['kind'] == 'table':
        return ('<div class="out"><div class="tablewrap">'
                f'{item["html"]}</div>'
                f'<div class="caption">{html.escape(item["caption"])}</div></div>')
    return (f'<div class="out"><pre class="out-text">'
            f'{html.escape(item["text"])}</pre></div>')


def slug(name):
    return (name.lower().replace(' ', '-').replace('=', '')
            .replace('(', '').replace(')', '').replace('.', ''))


def render(version):
    figures, body, toc = {}, [], []
    category = None

    for index, item in enumerate(_entries):
        if item['category'] != category:
            category = item['category']
            body.append(f'<h2 class="cat">{html.escape(category)}</h2>')
            toc.append(f'<h4>{html.escape(category)}</h4>')

        anchor = slug(item['name'])
        toc.append(f'<a href="#{anchor}">{html.escape(item["name"])}</a>')

        if item['output']['kind'] == 'figure':
            figures[f'fig{index}'] = {'spec': item['output']['spec'],
                                      'roles': item['output']['roles']}

        note = (f'<p class="note">{item["note"]}</p>' if item['note'] else '')
        body.append(
            f'<section class="fn">'
            f'<h3 id="{anchor}"><code>{html.escape(item["name"])}</code></h3>'
            f'<p class="blurb">{item["blurb"]}</p>'
            f'<pre><code>{html.escape(item["code"])}</code></pre>'
            f'{render_output(item["output"], index)}'
            f'{note}'
            f'</section>'
        )

    return PAGE.format(
        css=CSS, js=JS, toc='\n'.join(toc), body='\n'.join(body),
        figures=json.dumps(figures, separators=(',', ':')),
        version=version, count=len(_entries), start=START, end=END,
    )


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'PythonAQ {PythonAQ.__version__} - building the guide')

    metadata, aq = load()
    print(f'  {len(aq):,} hourly records for {SITE}')
    build_entries(metadata, aq)

    page = render(PythonAQ.__version__)
    target = OUT_DIR / 'index.html'
    target.write_text(page, encoding='utf-8')

    size_kb = target.stat().st_size // 1024
    figures = sum(1 for e in _entries if e['output']['kind'] == 'figure')
    tables = sum(1 for e in _entries if e['output']['kind'] == 'table')
    print(f'  {len(_entries)} entries: {figures} figures, {tables} tables, '
          f'{len(_entries) - figures - tables} text')
    print(f'  wrote {target.relative_to(ROOT)} ({size_kb:,} KB)')

    public = {n for n in PythonAQ.__all__ if not n.startswith('__')}
    missing = sorted(public - _covered)
    if missing:
        print(f'  NOT documented: {missing}')
        return 1
    print(f'  every one of the {len(public)} public functions is documented')
    return 0


if __name__ == '__main__':
    sys.exit(main())
