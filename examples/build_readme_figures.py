#!/usr/bin/env python3
"""Generate the figures embedded in README.md.

Every image in docs/images/ is produced by this script from real AURN data for
Leeds Centre, so the manual shows what the package actually renders rather than
hand-drawn mock-ups. Re-run it after changing any plotting function:

    pip install -e '.[calendar,docs]'
    python examples/build_readme_figures.py

Data is cached under examples/output/cache/, shared with demo_leeds.py, so
repeat runs do not re-download. Pass --refresh to force a re-download.

Chart rendering is deterministic, so re-running without code changes reproduces
the same bytes. map_sites is the exception: it fetches map tiles from a remote
server, and those come back subtly different each run, so its PNG would churn
in git on every rebuild. To avoid that, a freshly rendered image is only written
when it actually differs from the one on disk by more than a negligible amount.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Third-party noise that would otherwise bury this script's own output. None of
# it indicates a problem with the figures; see demo_leeds.py for the detail.
warnings.filterwarnings('ignore', message='Missing constructor for R class')
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(over='ignore', divide='ignore', invalid='ignore')

import PythonAQ  # noqa: E402
from PythonAQ import (  # noqa: E402
    aq_stats, conditional_eval, conditional_quantile, corr_plot,
    dist_plot, download_aurn_data, gaussian_smooth, import_aq_meta,
    kz_filter, linear_relation, map_sites, percentile_rose,
    polar_annulus, polar_cluster, polar_diff, polar_frequency_plot,
    polar_plot, pollutant_rose, rolling_quantile, run_regression,
    scatter_plot, smooth_trend_plot, summary_plot, taylor_diagram,
    theil_sen_plot, time_plot, time_prop, time_variation, trend_level,
    whittaker_smooth, wind_rose,
)

SITE, SOURCE = 'LEED', 'aurn'
START_YEAR, END_YEAR = 2022, 2025

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / 'docs' / 'images'
CACHE_DIR = ROOT / 'examples' / 'output' / 'cache'

# A moderate scale keeps the images sharp on high-DPI screens while holding the
# repository size down; GitHub renders README images at roughly 900px wide.
SCALE = 1.5

_written: list[tuple[str, int]] = []


def _rms_difference(a, b):
    """Root-mean-square per-pixel difference between two images, 0-255."""
    import math

    from PIL import ImageChops

    histogram = ImageChops.difference(a.convert('RGB'), b.convert('RGB')) \
        .convert('L').histogram()
    total = sum(histogram)
    return math.sqrt(sum(i * i * c for i, c in enumerate(histogram)) / total)


def _shrink(png_bytes):
    """Re-encode a PNG as small as it will go without a visible change.

    Charts use few distinct colours, so an adaptive 256-colour palette is
    typically several times smaller than full RGB. It is only accepted when the
    RMS difference is imperceptible, which keeps it away from the photographic
    map tiles in map_sites, where it would band the imagery.
    """
    import io

    from PIL import Image

    original = Image.open(io.BytesIO(png_bytes)).convert('RGB')
    lossless = io.BytesIO()
    original.save(lossless, 'PNG', optimize=True)

    best, note = lossless, 'lossless'
    palette = original.quantize(colors=256, method=Image.MEDIANCUT,
                                dither=Image.FLOYDSTEINBERG)
    error = _rms_difference(original, palette)
    if error < 2.0:
        buffer = io.BytesIO()
        palette.save(buffer, 'PNG', optimize=True)
        if buffer.tell() < best.tell():
            best, note = buffer, f'256 colours, rms {error:.2f}'

    return (best.getvalue() if best.tell() < len(png_bytes) else png_bytes), note


def _is_visually_unchanged(path, new_bytes, tolerance=1.0):
    """True if `new_bytes` is indistinguishable from the PNG already at `path`.

    map_sites re-fetches its map tiles on every run and comes back subtly
    different each time. Rewriting it would churn a few hundred KB of binary in
    git for no visible change, so an imperceptible difference counts as no
    change at all.
    """
    import io

    from PIL import Image

    if not path.exists():
        return False
    try:
        existing = Image.open(path)
        candidate = Image.open(io.BytesIO(new_bytes))
        if existing.size != candidate.size:
            return False
        return _rms_difference(existing, candidate) < tolerance
    except Exception:
        return False


def save(fig, name: str, width: int = 900, height: int = 620,
         scale: float = SCALE) -> None:
    """Render one figure, shrink it, and write it if it actually changed."""
    import plotly.io as pio

    path = IMAGE_DIR / f'{name}.png'
    rendered = pio.to_image(fig, format='png', width=width, height=height,
                            scale=scale)
    shrunk, note = _shrink(rendered)

    if _is_visually_unchanged(path, shrunk):
        size_kb = path.stat().st_size // 1024
        _written.append((name, size_kb))
        print(f'  {name + ".png":<26} {size_kb:>5} KB  unchanged, kept on disk')
        return

    path.write_bytes(shrunk)
    size_kb = path.stat().st_size // 1024
    _written.append((name, size_kb))
    print(f'  {name + ".png":<26} {size_kb:>5} KB  {note}')


def load_data(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_cache = CACHE_DIR / 'aurn_meta.parquet'
    aq_cache = CACHE_DIR / f'{SITE}_{START_YEAR}_{END_YEAR}.parquet'

    if meta_cache.exists() and not refresh:
        metadata = pd.read_parquet(meta_cache)
    else:
        metadata = import_aq_meta(SOURCE)
        metadata.to_parquet(meta_cache)

    if aq_cache.exists() and not refresh:
        aq = pd.read_parquet(aq_cache)
    else:
        aq = download_aurn_data(SITE, START_YEAR, END_YEAR, SOURCE)
        aq.columns = [str(c) for c in aq.columns]
        aq.to_parquet(aq_cache)
    aq.columns = [str(c) for c in aq.columns]

    if aq.empty:
        sys.exit('No AURN data available; cannot build figures.')
    print(f'  {len(aq):,} hourly records, '
          f'{aq["date_time"].min():%Y-%m-%d} to {aq["date_time"].max():%Y-%m-%d}')
    return metadata, aq


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true',
                        help='re-download instead of using the cache')
    args = parser.parse_args()

    try:
        import kaleido  # noqa: F401
    except ImportError:
        sys.exit(
            "Static image export needs kaleido. Install the docs extra:\n"
            "    pip install -e '.[calendar,docs]'"
        )

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f'PythonAQ {PythonAQ.__version__} - building README figures')
    print(f'output -> {IMAGE_DIR}\n')

    metadata, aq = load_data(refresh=args.refresh)
    print()

    # --- Directional analysis -------------------------------------------------
    save(polar_plot(aq, conc_col='NO2', min_count=10, resolution=400,
                    title='NO2 by wind speed and direction, Leeds Centre 2022-2025'),
         'polar_plot', width=760, height=700)

    save(polar_plot(aq, conc_col='NO2', min_count=10, render='contour',
                    title='The same surface as filled contour bands'),
         'polar_plot_contour', width=760, height=700)

    save(polar_plot(aq, conc_col='NO2', render='tile', ws_limit='max',
                    ws_bins=60, exclude_missing=False,
                    title="render='tile': one flat-filled polygon per bin"),
         'polar_plot_tile', width=760, height=700)

    fig, _ = wind_rose(aq, group_by='none', title='Wind rose, Leeds Centre')
    save(fig, 'wind_rose', width=820, height=680)

    fig, _ = pollutant_rose(aq, pollutant='NO2', title='NO2 pollutant rose')
    save(fig, 'pollutant_rose', width=820, height=680)

    fig, _ = percentile_rose(aq, 'NO2', percentile=[25, 50, 75, 90, 95],
                             title='NO2 percentiles by wind direction')
    save(fig, 'percentile_rose', width=760, height=700)

    save(percentile_rose(aq, 'NO2', statistic='cpf', percentile=95,
                         title='CPF: how often NO2 is in its top 5%')[0],
         'cpf_rose', width=760, height=700)

    save(polar_annulus(aq, 'NO2', period='hour')[0],
         'polar_annulus', width=760, height=740)

    early = aq[aq['date_time'] < '2024-01-01']
    late = aq[aq['date_time'] >= '2024-01-01']
    save(polar_diff(early, late, conc_col='NO2', min_count=10, resolution=400,
                    title='NO2 change, 2024-25 against 2022-23'),
         'polar_diff', width=760, height=700)

    save(dist_plot(aq, [c for c in ('NO2', 'PM2.5', 'O3') if c in aq.columns],
                   title='Distribution of the measured pollutants'),
         'dist_plot', width=900, height=520)

    x, y = ('NOXasNO2', 'NO2') if 'NOXasNO2' in aq.columns else ('NO', 'NO2')
    save(linear_relation(aq, x=x, y=y, period='month',
                         title=f'{y} against {x}, monthly slope')[0],
         'linear_relation', width=940, height=500)

    predictors = [c for c in (x, 'ws', 'air_temp') if c in aq.columns][:3]
    save(run_regression(aq, y=y, x=predictors, window=336, step=72,
                        title=f'Rolling regression of {y}, two-week window')[0],
         'run_regression', width=940, height=600)

    smoothed = rolling_quantile(aq[['date_time', 'NO2']], 'NO2', width=24,
                                quantile=0.5, new_name='rolling median (24 h)')
    smoothed = kz_filter(smoothed, 'NO2', width=24, iterations=3,
                         new_name='KZ(24, 3)')
    smoothed = whittaker_smooth(smoothed, 'NO2', lam=1e6, new_name='Whittaker')
    smoothed = gaussian_smooth(smoothed, 'NO2', sigma=12, new_name='Gaussian')
    window = smoothed[(smoothed['date_time'] >= '2023-01-01')
                      & (smoothed['date_time'] < '2023-03-01')]
    save(time_plot(window, columns_to_plot=['NO2', 'rolling median (24 h)',
                                            'KZ(24, 3)', 'Whittaker',
                                            'Gaussian'],
                   title='The four smoothers on two months of NO2'),
         'smoothers', width=980, height=520)

    save(polar_frequency_plot(aq, separate_by_year=False,
                              title='Wind speed and direction frequency'),
         'polar_frequency', width=760, height=700)

    # Conditioning: the same plot, once per season, on shared scales.
    save(polar_plot(aq, conc_col='NO2', type='season', min_count=8,
                    resolution=260, ws_limit=12, panel_height=430,
                    title='NO2 by season'),
         'polar_plot_by_season', width=900, height=880, scale=1.2)

    save(polar_cluster(aq, feature_cols=['NO2', 'PM10', 'O3'], n_clusters=6),
         'polar_cluster', width=800, height=700)

    # --- Time series and trends ----------------------------------------------
    fig, _ = time_variation(aq, ['NO2', 'O3'], n_boot=200, random_state=42,
                            title='Temporal variation')
    save(fig, 'time_variation', width=1000, height=700)

    save(theil_sen_plot(aq, pollutant_col='NO2', agg_freq='ME',
                        title='Theil-Sen trend in monthly mean NO2'),
         'theil_sen', width=900, height=520)

    save(smooth_trend_plot(aq, pollutant_col='NO2', avg_freq='MS',
                           title='Smooth (GAM) trend in monthly mean NO2'),
         'smooth_trend', width=900, height=520)

    save(time_plot(aq, columns_to_plot=['NO2', 'PM10', 'O3'],
                   averaging_period='ME', group_data=True,
                   title='Monthly means'),
         'time_plot', width=900, height=520)

    fig, levels = trend_level(aq, 'NO2', x='month', y='hour', type='year',
                              title='NO2 by month, hour and year')
    save(fig, 'trend_level', width=1000, height=620)

    # --- Distributions and relationships -------------------------------------
    fig, _ = summary_plot(aq[['date_time', 'NO2', 'PM10', 'O3']])
    save(fig, 'summary_plot', width=950, height=620)

    fig, fit = scatter_plot(aq, 'NOXasNO2', 'NO2', linear=True, smooth=True,
                            title='NO2 against NOx')
    save(fig, 'scatter_plot', width=850, height=620)

    fig, corr = corr_plot(
        aq, ['NO', 'NO2', 'NOXasNO2', 'O3', 'PM10', 'PM2.5', 'SO2', 'CO', 'ws', 'temp'],
        cluster=True, title='Correlation between pollutants',
    )
    save(fig, 'corr_plot', width=760, height=700)

    # Rendered at scale 1: this is the only figure made of photographic map
    # tiles, so it cannot be palette-reduced and would otherwise dominate the
    # repository size.
    save(map_sites(metadata, sites=[SITE, 'LED6']), 'map_sites',
         width=820, height=520, scale=1.0)

    # --- Model evaluation -----------------------------------------------------
    # No model output ships with the package, so a persistence forecast and a
    # deliberately damped variant stand in: enough to show what the plots say.
    evaluation = aq[['date_time', 'NO2']].dropna().copy()
    evaluation['persistence'] = evaluation['NO2'].shift(24)
    evaluation['damped'] = evaluation['NO2'] * 0.6 + 8
    evaluation = evaluation.dropna()

    met = [c for c in ('ws', 'air_temp') if c in aq.columns]
    save(conditional_eval(evaluation.merge(aq[['date_time', *met]],
                                           on='date_time', how='left'),
                          obs='NO2', mod='damped', variables=met,
                          title='Where the damped model goes wrong, and with what')[0],
         'conditional_eval', width=940, height=760)

    save(conditional_quantile(evaluation, obs='NO2', mod='damped',
                              title='Conditional quantiles: a damped model')[0],
         'conditional_quantile', width=800, height=700)

    save(taylor_diagram(evaluation, obs='NO2', mod=['persistence', 'damped'],
                        title='Taylor diagram: two stand-in models')[0],
         'taylor_diagram', width=760, height=720)

    # --- Time proportion ------------------------------------------------------
    save(time_prop(aq, 'NO2', 'wd', avg_time='month',
                   title='Monthly NO2 split by wind sector')[0],
         'time_prop', width=1000, height=560)

    # --- calendar needs the optional plotly-calplot ---------------------------
    try:
        save(PythonAQ.calendar(aq, value_column='PM10', date_column='date_time'),
             'calendar', width=1000, height=760)
    except Exception as exc:
        print(f'  calendar.png skipped: {type(exc).__name__}: {str(exc)[:70]}')

    total = sum(kb for _, kb in _written)
    print(f'\n  {len(_written)} figures, {total} KB total')

    # Print a couple of numbers the README quotes, so they can be kept honest.
    print('\nFigures reference these values:')
    hourly = aq.groupby(aq['date_time'].dt.hour)['NO2'].mean()
    print(f'  NO2 peak hour        : {int(hourly.idxmax())}:00')
    print(f'  NO2 / NOx slope, R2  : {fit["slope"].iloc[0]:.3f}, '
          f'{fit["r_squared"].iloc[0]:.3f}')
    print(f'  strongest correlation: PM10/PM2.5 r = {corr.loc["PM10", "PM2.5"]:.3f}')
    print(f'  O3 vs NO2            : r = {corr.loc["O3", "NO2"]:.3f}')
    print('\nAnnual summary (NO2):')
    print(aq_stats(aq, 'NO2', data_thresh=75).round(1).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
