#!/usr/bin/env python3
"""End-to-end PythonAQ demo: Leeds Centre (LEED), AURN, 2022-2025.

Exercises every public function in the package against real data, writes each
figure to examples/output/ as a standalone HTML file, and prints the tabular
results to the console.

Run with:

    pip install -e '.[calendar]'
    python examples/demo_leeds.py

The `calendar` extra is needed for the calendar plot; without it that one
section is skipped and everything else still runs. Note the extra pins
pandas<3, because plotly-calplot is not compatible with pandas 3.

Data is downloaded from DEFRA (uk-air.defra.gov.uk) and NOAA
(ncei.noaa.gov) on each run and cached under examples/output/cache/.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import PythonAQ
from PythonAQ import (
    aq_stats,
    calc_percentile,
    corr_plot,
    cut_data,
    deseason_data,
    download_aurn_data,
    download_noaa_data,
    e_sat,
    get_period,
    get_r_data,
    import_aq_meta,
    map_sites,
    mod_stats,
    parse_noaa_data,
    percentile_rose,
    polar_cluster,
    polar_frequency_plot,
    polar_plot,
    pollutant_rose,
    rh,
    rolling_mean,
    scatter_plot,
    select_by_date,
    smooth_trend_plot,
    summary_plot,
    theil_sen_plot,
    time_average,
    time_plot,
    time_variation,
    trend_level,
    wind_rose,
)

# Third-party noise that would otherwise bury the demo's own output. None of
# these indicate a problem with the results:
#   - rdata warns about R's POSIXt classes on every AURN file; the dates still
#     convert correctly.
#   - the GAM fitting inside polar_plot trips harmless overflow/divide warnings
#     deep in scipy's linear algebra while searching its penalty grid.
warnings.filterwarnings('ignore', message='Missing constructor for R class')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
np.seterr(over='ignore', divide='ignore', invalid='ignore')

SITE = 'LEED'
SOURCE = 'aurn'
START_YEAR, END_YEAR = 2022, 2025
# Bingley Number 2: the closest NOAA station to Leeds Centre, ~21 km west.
NOAA_STATION = '033440-99999'

OUTPUT_DIR = Path(__file__).parent / 'output'
CACHE_DIR = OUTPUT_DIR / 'cache'

_figure_count = 0
_covered: set[str] = set()


def heading(text: str) -> None:
    print(f'\n{"=" * 78}\n{text}\n{"=" * 78}')


def step(text: str) -> None:
    print(f'\n--- {text} ---')


def used(*names: str) -> None:
    """Record which public functions a section exercised."""
    _covered.update(names)


def save(fig, name: str, description: str) -> None:
    """Write a figure to a standalone HTML file."""
    global _figure_count
    _figure_count += 1
    path = OUTPUT_DIR / f'{_figure_count:02d}_{name}.html'
    fig.write_html(path, include_plotlyjs='cdn')
    print(f'  saved {path.name:<34} {description}')


def show(frame: pd.DataFrame, rows: int = 8, decimals: int = 2) -> None:
    """Print a DataFrame compactly."""
    with pd.option_context('display.max_columns', 40, 'display.width', 200):
        print(frame.head(rows).round(decimals).to_string())
    if len(frame) > rows:
        print(f'  ... {len(frame) - rows} more rows')


def load_data(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch site metadata, AURN observations and NOAA meteorology."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_cache = CACHE_DIR / 'aurn_meta.parquet'
    aq_cache = CACHE_DIR / f'{SITE}_{START_YEAR}_{END_YEAR}.parquet'
    met_cache = CACHE_DIR / f'noaa_{NOAA_STATION}_{START_YEAR}_{END_YEAR}.parquet'

    heading('1. DATA RETRIEVAL')

    step('import_aq_meta - AURN site metadata')
    if meta_cache.exists() and not refresh:
        metadata = pd.read_parquet(meta_cache)
        print(f'  loaded {len(metadata)} sites from cache')
    else:
        metadata = import_aq_meta(SOURCE)
        metadata.to_parquet(meta_cache)
        print(f'  downloaded {len(metadata)} sites')
    used('import_aq_meta')

    site_row = metadata[metadata['site_id'] == SITE]
    if site_row.empty:
        sys.exit(f'Site {SITE} not found in the {SOURCE} metadata.')
    show(site_row[['site_id', 'site_name', 'location_type', 'latitude',
                   'longitude', 'zone', 'local_authority', 'start_date',
                   'end_date']], rows=1)

    step('get_r_data - the low-level RData reader behind the importers')
    if refresh or not meta_cache.exists():
        raw = get_r_data('https://uk-air.defra.gov.uk/openair/R_data/AURN_metadata.RData')
        print(f'  fetched R objects: {list(raw)}')
    else:
        print('  skipped (cached); this is the primitive import_aq_meta builds on')
    used('get_r_data')

    step(f'download_aurn_data - {SITE}, {START_YEAR}-{END_YEAR}')
    if aq_cache.exists() and not refresh:
        aq = pd.read_parquet(aq_cache)
        print(f'  loaded {len(aq):,} hourly records from cache')
    else:
        aq = download_aurn_data(SITE, START_YEAR, END_YEAR, SOURCE)
        # rdata returns numpy string objects as column labels; normalise them
        # so ordinary string indexing behaves predictably downstream.
        aq.columns = [str(c) for c in aq.columns]
        aq.to_parquet(aq_cache)
        print(f'  downloaded {len(aq):,} hourly records')
    aq.columns = [str(c) for c in aq.columns]
    used('download_aurn_data')

    if aq.empty:
        sys.exit('No AURN data returned; cannot continue.')

    print(f'  period      : {aq["date_time"].min()} -> {aq["date_time"].max()}')
    print(f'  pollutants  : {[c for c in aq.columns if c not in ("date", "date_time", "site", "code")]}')
    print('\n  data capture (%):')
    span = len(pd.date_range(aq['date_time'].min(), aq['date_time'].max(), freq='h'))
    capture = (aq.notna().sum() / span * 100).round(1)
    print('   ', capture.drop(['date', 'date_time', 'site', 'code']).to_dict())

    step(f'download_noaa_data + parse_noaa_data - station {NOAA_STATION}')
    if met_cache.exists() and not refresh:
        met = pd.read_parquet(met_cache)
        print(f'  loaded {len(met):,} hourly records from cache')
    else:
        try:
            met = download_noaa_data(NOAA_STATION, START_YEAR, END_YEAR)
            if not met.empty:
                met.to_parquet(met_cache)
            print(f'  downloaded {len(met):,} hourly records')
        except Exception as exc:  # network problems must not kill the demo
            print(f'  NOAA download failed ({exc}); continuing without met data')
            met = pd.DataFrame()
    used('download_noaa_data')

    if not met.empty:
        print(f'  columns     : {[c for c in met.columns]}')
        show(met.head(3))

    step('parse_noaa_data - the raw ISD parser, called directly')
    print('  download_noaa_data uses this internally; calling it on one raw year')
    print('  shows what it does: split the packed comma-separated ISD fields,')
    print('  clear the missing-value sentinels, then apply the x10 scaling.')
    try:
        import io

        import requests

        code = NOAA_STATION.replace('-', '')
        url = f'https://www.ncei.noaa.gov/data/global-hourly/access/{END_YEAR}/{code}.csv'
        raw = pd.read_csv(io.StringIO(requests.get(url, timeout=60).text))
        print(f'  raw ISD columns  : {[c for c in raw.columns if c in ("TMP", "DEW", "SLP", "WND")]}')
        row = raw.iloc[0]
        print(f'  first raw record : TMP={row["TMP"]!r}  DEW={row["DEW"]!r}  '
              f'SLP={row["SLP"]!r}  WND={row["WND"]!r}')
        print(f'    -> decodes to  : air_temp {int(row["TMP"].split(",")[0]) / 10:.1f} degC, '
              f'dew_point {int(row["DEW"].split(",")[0]) / 10:.1f} degC, '
              f'pressure {int(row["SLP"].split(",")[0]) / 10:.1f} hPa, '
              f'ws {int(row["WND"].split(",")[3]) / 10:.1f} m/s')

        parsed = parse_noaa_data(raw)
        # parse_noaa_data resamples to hourly means, so its first row is the
        # average of every observation in that hour, not this single record.
        first = parsed.dropna(subset=['air_temp']).iloc[0]
        print(f'  parsed output    : {len(raw):,} raw records -> {len(parsed):,} hourly means')
        print(f'    first hour     : air_temp {first["air_temp"]:.1f} degC, '
              f'dew_point {first["dew_point"]:.1f} degC, '
              f'pressure {first["atmospheric_pressure"]:.1f} hPa, '
              f'RH {first["relative_humidity"]:.1f}%')
        used('parse_noaa_data')
    except Exception as exc:
        print(f'  raw fetch failed ({exc}); parse_noaa_data still ran via '
              f'download_noaa_data above')
        if not met.empty:
            used('parse_noaa_data')

    return metadata, aq, met


def section_utilities(aq: pd.DataFrame, met: pd.DataFrame) -> pd.DataFrame:
    heading('2. DATA UTILITIES')

    step('time_average - daily and monthly means, 75% data capture required')
    daily = time_average(aq, avg_time='day', data_thresh=75)
    monthly = time_average(aq, avg_time='month', data_thresh=75)
    print(f'  hourly {len(aq):,} -> daily {len(daily):,} -> monthly {len(monthly):,}')
    print('  note: wind direction is vector-averaged, not scalar-averaged')
    show(daily[['date_time', 'NO2', 'PM10', 'ws', 'wd']])
    used('time_average')

    step('select_by_date - flexible subsetting')
    winter = select_by_date(aq, season='winter')
    weekday_rush = select_by_date(aq, day=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], hour=8)
    year_2024 = select_by_date(aq, year=2024)
    june_2023 = select_by_date(aq, start='2023-06-01', end='2023-06-30')
    print(f'  winter months        : {len(winter):,} hours')
    print(f'  weekday 08:00 only   : {len(weekday_rush):,} hours')
    print(f'  2024                 : {len(year_2024):,} hours')
    print(f'  June 2023            : {len(june_2023):,} hours')
    print(f'  mean NO2 winter {winter["NO2"].mean():.1f} vs '
          f'summer {select_by_date(aq, season="summer")["NO2"].mean():.1f} ug/m3')
    used('select_by_date')

    step('rolling_mean - running 8-hour ozone mean, as used for the UK objective')
    with_rolling = rolling_mean(aq, 'O3', width=8, data_thresh=75)
    peak = with_rolling['rolling8_O3'].max()
    exceedances = (with_rolling['rolling8_O3'] > 100).sum()
    print(f'  peak running 8-hour O3     : {peak:.1f} ug/m3')
    print(f'  hours above 100 ug/m3      : {exceedances:,}')
    used('rolling_mean')

    step('cut_data - conditioning splits')
    for split in ['season', 'weekday', 'weekend', 'daylight', 'wd']:
        tagged = cut_data(aq, type=split)
        means = tagged.groupby(split, observed=True)['NO2'].mean().round(1)
        print(f'  by {split:<9}: {means.to_dict()}')
    used('cut_data')

    step('calc_percentile - monthly NO2 percentiles')
    pctiles = calc_percentile(aq, 'NO2', percentile=[25, 50, 75, 95], avg_time='month')
    show(pctiles)
    used('calc_percentile')

    step('deseason_data + get_period - remove the seasonal cycle')
    period = get_period('7D')
    print(f'  get_period("7D") = {period} periods per year')
    deseasoned = deseason_data(aq, pollutant_column='NO2', interval='7D',
                               period=period, method='additive',
                               date_column='date_time').reset_index()
    print(f'  raw weekly NO2 sd        : {deseasoned["NO2"].std():.2f}')
    print(f'  deseasoned weekly NO2 sd : {deseasoned["deseasoned_NO2"].std():.2f}')
    used('deseason_data', 'get_period')

    step('e_sat + rh - saturation vapour pressure and relative humidity')
    print(f'  e_sat(20 degC)        = {float(e_sat(20.0)):.2f} hPa')
    print(f'  rh(20 degC, 10 degC)  = {float(rh(20.0, 10.0)):.1f} %')
    print(f'  rh(15 degC, 15 degC)  = {float(rh(15.0, 15.0)):.1f} % (saturated)')
    if not met.empty and 'relative_humidity' in met.columns:
        observed_rh = met['relative_humidity'].dropna()
        print(f'  NOAA-derived RH at Bingley: mean {observed_rh.mean():.1f}%, '
              f'range {observed_rh.min():.1f}-{observed_rh.max():.1f}%')
    used('e_sat', 'rh')

    return deseasoned


def section_plots(metadata: pd.DataFrame, aq: pd.DataFrame,
                  deseasoned: pd.DataFrame) -> None:
    heading('3. VISUALISATION')

    step('map_sites')
    save(map_sites(metadata, sites=[SITE, 'LED6']), 'map_sites',
         'Leeds monitoring sites')
    used('map_sites')

    step('summary_plot')
    fig, summary = summary_plot(aq[['date_time', 'NO2', 'PM10', 'O3']])
    save(fig, 'summary_plot', 'time series, rug and histogram per pollutant')
    show(summary)
    used('summary_plot')

    step('time_plot')
    save(time_plot(aq, columns_to_plot=['NO2', 'PM10', 'O3'],
                   averaging_period='ME', group_data=True),
         'time_plot', 'monthly means, overlaid')
    used('time_plot')

    step('calendar')
    try:
        calendar = PythonAQ.calendar
        save(calendar(aq, value_column='PM10', date_column='date_time'),
             'calendar', 'daily mean PM10 as a calendar heatmap')
        used('calendar')
    except ImportError as exc:
        print(f'  skipped: {exc}')

    step('time_variation - the four-panel temporal summary')
    fig, variation = time_variation(aq, ['NO2', 'O3'], n_boot=200, random_state=42)
    save(fig, 'time_variation', 'hour/weekday/month, 95% bootstrap CI')
    hourly = variation[(variation['panel'] == 'hour') & (variation['pollutant'] == 'NO2')]
    peak_hour = int(hourly.loc[hourly['value'].idxmax(), 'x'])
    weekday = variation[(variation['panel'] == 'weekday') & (variation['pollutant'] == 'NO2')]
    weekday = weekday.set_index('x')['value']
    print(f'  NO2 peaks at hour {peak_hour}:00')
    print(f'  weekday mean {weekday[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]].mean():.1f} '
          f'vs weekend {weekday[["Saturday", "Sunday"]].mean():.1f} ug/m3')
    used('time_variation')

    step('trend_level - heat map of NO2 by month and hour, per year')
    fig, levels = trend_level(aq, 'NO2', x='month', y='hour', type='year')
    save(fig, 'trend_level', 'NO2 by month x hour, one panel per year')
    show(levels)
    used('trend_level')

    step('wind_rose')
    fig, rose = wind_rose(aq, group_by='none')
    save(fig, 'wind_rose', 'wind speed and direction distribution')
    fig_year, _ = wind_rose(aq, group_by='year')
    save(fig_year, 'wind_rose_by_year', 'one wind rose per year')
    used('wind_rose')

    step('pollutant_rose')
    fig, prose = pollutant_rose(aq, pollutant='NO2')
    save(fig, 'pollutant_rose', 'NO2 concentration by wind direction')
    used('pollutant_rose')

    step('percentile_rose')
    fig, proses = percentile_rose(aq, 'NO2', percentile=[25, 50, 75, 90, 95])
    save(fig, 'percentile_rose', 'NO2 percentiles by wind direction')
    worst = proses.loc[proses['percentile.95'].idxmax()]
    print(f'  highest 95th percentile from {worst["wd"]:.0f} degrees: '
          f'{worst["percentile.95"]:.1f} ug/m3')
    used('percentile_rose')

    step('polar_plot')
    save(polar_plot(aq, conc_col='NO2'), 'polar_plot',
         'NO2 by wind speed and direction, GAM-smoothed')
    used('polar_plot')

    step('polar_frequency_plot')
    save(polar_frequency_plot(aq, separate_by_year=False), 'polar_frequency',
         'wind speed/direction frequency')
    used('polar_frequency_plot')

    step('polar_cluster')
    save(polar_cluster(aq, feature_cols=['NO2', 'PM10', 'O3'], n_clusters=6),
         'polar_cluster', 'k-means clusters in polar coordinates')
    used('polar_cluster')

    step('theil_sen_plot - non-parametric trend')
    save(theil_sen_plot(aq, pollutant_col='NO2', agg_freq='ME'),
         'theil_sen', 'monthly NO2 with Theil-Sen slope and CI')
    save(theil_sen_plot(aq, pollutant_col='NO2', agg_freq='ME', deseason=True),
         'theil_sen_deseasoned', 'as above, seasonal cycle removed')
    used('theil_sen_plot')

    step('smooth_trend_plot - GAM trend')
    save(smooth_trend_plot(aq, pollutant_col='NO2', avg_freq='MS'),
         'smooth_trend', 'monthly NO2 with a GAM smooth and CI')
    used('smooth_trend_plot')

    step('scatter_plot')
    fig, fit = scatter_plot(aq, 'NOXasNO2', 'NO2', linear=True, smooth=True)
    save(fig, 'scatter_plot', 'NO2 against NOx, with linear and LOWESS fits')
    print(f'  NO2 = {fit["slope"].iloc[0]:.3f} x NOx + {fit["intercept"].iloc[0]:.2f}, '
          f'R2 = {fit["r_squared"].iloc[0]:.3f}, n = {int(fit["n"].iloc[0]):,}')
    save(scatter_plot(aq, 'ws', 'PM10', method='hexbin')[0], 'scatter_hexbin',
         'PM10 against wind speed, binned')
    used('scatter_plot')

    step('corr_plot')
    fig, corr = corr_plot(
        aq, ['NO', 'NO2', 'NOXasNO2', 'O3', 'PM10', 'PM2.5', 'SO2', 'CO', 'ws', 'temp'],
        cluster=True,
    )
    save(fig, 'corr_plot', 'correlation matrix, clustered')
    print('  strongest and weakest pairs:')
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    print(f'    {pairs.idxmax()}: r = {pairs.max():.3f}')
    print(f'    {pairs.idxmin()}: r = {pairs.min():.3f}')
    used('corr_plot')


def section_statistics(aq: pd.DataFrame, met: pd.DataFrame) -> None:
    heading('4. STATISTICS')

    step('aq_stats - annual summaries and objective exceedances')
    for pollutant in ['NO2', 'PM10', 'O3']:
        print(f'\n  {pollutant}:')
        show(aq_stats(aq, pollutant, data_thresh=75), rows=5)
    used('aq_stats')

    step('mod_stats - evaluating a 24-hour persistence forecast for NO2')
    print('  "model" = the value 24 hours earlier. A deliberately naive baseline,')
    print('  used here only to exercise the statistics against real data.')
    persistence = aq[['date_time', 'NO2']].copy()
    persistence['forecast'] = persistence['NO2'].shift(24)
    stats = mod_stats(persistence, mod='forecast', obs='NO2')
    show(stats, decimals=3)
    print('\n  interpretation: COE = 1 is perfect, 0 means no better than the')
    print('  observed mean; IOA spans -1 to +1; FAC2 is the fraction within a')
    print('  factor of two.')

    if not met.empty and 'air_temp' in met.columns:
        print('\n  A second, more meaningful comparison: NOAA Bingley air temperature')
        print('  against the AURN Leeds Centre temperature, ~21 km apart.')
        met_hourly = met[['date', 'air_temp']].copy()
        met_hourly['date'] = pd.to_datetime(met_hourly['date'])
        merged = pd.merge(
            aq[['date_time', 'temp']].rename(columns={'date_time': 'date'}),
            met_hourly, on='date', how='inner',
        ).dropna()
        if len(merged) > 100:
            show(mod_stats(merged, mod='air_temp', obs='temp'), decimals=3)
            seasonal = mod_stats(
                cut_data(merged.rename(columns={'date': 'date_time'}), type='season'),
                mod='air_temp', obs='temp', group_by='season',
            )
            print('\n  by season:')
            show(seasonal, decimals=3)
        else:
            print(f'  only {len(merged)} overlapping hours; skipping')
    used('mod_stats')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true',
                        help='re-download instead of using the cache')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'PythonAQ {PythonAQ.__version__} demo')
    print(f'site {SITE} ({SOURCE}), {START_YEAR}-{END_YEAR}')
    print(f'output -> {OUTPUT_DIR}')

    metadata, aq, met = load_data(refresh=args.refresh)
    deseasoned = section_utilities(aq, met)
    section_plots(metadata, aq, deseasoned)
    section_statistics(aq, met)

    heading('COVERAGE')
    public = {n for n in PythonAQ.__all__ if not n.startswith('__')}
    missing = sorted(public - _covered)
    print(f'  exercised {len(_covered)} of {len(public)} public functions')
    print(f'  wrote {_figure_count} figures to {OUTPUT_DIR}')
    if missing:
        print(f'  NOT exercised: {missing}')
        return 1
    print('  every public function was exercised')
    return 0


if __name__ == '__main__':
    sys.exit(main())
