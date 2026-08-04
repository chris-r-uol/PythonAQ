# PythonAQ - Air Quality Data Analysis and Visualization Toolkit
This repository provides a comprehensive set of Python tools for downloading, processing, and visualising air quality and meteorological data. The toolkit is designed for researchers, scientists, and environmental analysts who need to work with air quality datasets from various sources, such as AURN, NOAA, and others.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
    - [Data Retrieval Functions](#data-retrieval-functions)
        - [import_aq_meta](#import_aq_meta)
        - [download_aurn_data](#download_aurn_data)
        - [download_noaa_data](#download_noaa_data)
    - [Data Parsing Functions](#data-parsing-functions)
        - [parse_noaa_data](#parse_noaa_data)
    - [Visualisation Functions](#visualisation-functions)
        - [calendar](#calendar)
        - [map_sites](#map_sites)
        - [polar_cluster](#polar_cluster)
        - [polar_frequency_plot](#polar_frequency_plot)
        - [polar_plot](#polar_plot)
        - [pollutant_rose](#pollutant_rose)
        - [smooth_trend_plot](#smooth_trend_plot)
        - [summary_plot](#summary_plot)
        - [theil_sen_plot](#theil_sen_plot)
        - [time_plot](#time_plot)
        - [wind_rose](#wind_rose)
        - [percentile_rose](#percentile_rose)
        - [time_variation](#time_variation)
        - [trend_level](#trend_level)
        - [scatter_plot](#scatter_plot)
        - [corr_plot](#corr_plot)
    - [Statistics](#statistics)
        - [mod_stats](#mod_stats)
        - [aq_stats](#aq_stats)
    - [Utilities](#utilities)
        - [time_average](#time_average)
        - [select_by_date](#select_by_date)
        - [rolling_mean](#rolling_mean)
        - [cut_data](#cut_data)
        - [calc_percentile](#calc_percentile)
        - [deseason_data](#deseason_data)
        - [get_period](#get_period)
        - [e_sat and rh](#e_sat-and-rh)
- [Relationship to openair](#relationship-to-openair)
- [Worked example](#worked-example)
- [Development](#development)
- [Contributing](#contributing)
- [Licence](#licence)

## Features
- **Data Retrieval**: Functions to download air quality and meteorological data from various sources.
- **Data Parsing**: Utilities to parse and clean raw data into usable formats.
- **Visualisation**: Advanced plotting functions using Plotly for interactive data visualisation.
- **Statistical Analysis**: Tools for performing regression analysis and clustering.
- **Meteorological Calculations**: Functions to calculate relative humidity and saturation vapour pressure.
- **Model Evaluation**: Standard statistics for comparing modelled against observed concentrations.

## Installation
Clone the repository and install the package:

```bash
git clone https://github.com/chris-r-uol/PythonAQ.git
cd PythonAQ
pip install -e .
```

This installs `PythonAQ` and its runtime dependencies, so the package is
importable from anywhere:

```python
from PythonAQ import import_aq_meta, wind_rose
```

### Optional extras
Some features carry dependencies that are deliberately kept out of the core
install:

```bash
pip install -e '.[calendar]'  # the calendar plot
pip install -e '.[app]'       # the Streamlit demo application
pip install -e '.[dev]'       # test and lint tooling
```

> **Note on the `calendar` extra:** it relies on `plotly-calplot`, which pins
> `numpy<2` and `plotly<6`, and is not compatible with pandas 3. Installing it
> will constrain the rest of your environment, which is why `calendar` is the
> only function not available from a plain `pip install -e .`.

## Dependencies
- Python 3.9 or higher
- numpy, pandas, scipy
- plotly
- requests, rdata
- scikit-learn
- statsmodels
- pymannkendall
- pygam

Optional: `plotly-calplot` (calendar plot), `streamlit` (demo app),
`pytest`/`flake8` (development).

## Web Application Demo
A web application written using [Streamlit](https://streamlit.io) is available
as `app.py` in the repository root. Install the `app` extra, then run it from
the repository root:

```bash
pip install -e '.[app,calendar]'
streamlit run app.py
```

The web app has pre-loaded the current visualisation functions and
implementations can be used elsewhere.

## Relationship to openair

PythonAQ follows the R [openair](https://github.com/openair-project/openair)
package, using Python naming conventions (`timeVariation` becomes
`time_variation`). Where a statistic has a precise definition, such as the
`modStats` metrics, the formulas follow the openair R source exactly.

| openair | PythonAQ | | openair | PythonAQ |
|---|---|---|---|---|
| `importAURN` etc. | `download_aurn_data` | | `timeAverage` | `time_average` |
| `importMeta` | `import_aq_meta` | | `selectByDate` | `select_by_date` |
| `windRose` | `wind_rose` | | `rollingMean` | `rolling_mean` |
| `pollutionRose` | `pollutant_rose` | | `cutData` | `cut_data` |
| `percentileRose` | `percentile_rose` | | `calcPercentile` | `calc_percentile` |
| `polarPlot` | `polar_plot` | | `modStats` | `mod_stats` |
| `polarFreq` | `polar_frequency_plot` | | `aqStats` | `aq_stats` |
| `polarCluster` | `polar_cluster` | | `corPlot` | `corr_plot` |
| `timePlot` | `time_plot` | | `scatterPlot` | `scatter_plot` |
| `timeVariation` | `time_variation` | | `trendLevel` | `trend_level` |
| `TheilSen` | `theil_sen_plot` | | `calendarPlot` | `calendar` |
| `smoothTrend` | `smooth_trend_plot` | | | |

Not yet ported: `polarAnnulus`, `polarDiff`, `timeProp`, `TaylorDiagram`,
`conditionalQuantile`, `conditionalEval`, `distPlot`, the trajectory functions
(`trajPlot`, `trajLevel`, `trajCluster`), and the remaining smoothers
(`GaussianSmooth`, `WhittakerSmooth`, `kzFilter`).

Two deliberate differences from openair:
- `time_average` averages wind direction as a vector but keeps the **scalar**
  mean wind speed. Pass `vector_ws=True` for the vector magnitude instead.
- `cut_data(type='daylight')` uses a fixed 07:00-19:00 window rather than
  openair's solar elevation calculation.

## Worked example

`examples/demo_leeds.py` runs every public function against real data for
Leeds Centre (AURN site `LEED`), 2022-2025, and writes each figure to
`examples/output/` as a standalone HTML file:

```bash
pip install -e '.[calendar]'
python examples/demo_leeds.py
```

It downloads ~35,000 hourly records from DEFRA plus meteorology from the
nearest NOAA station, caches them under `examples/output/cache/` so re-runs are
instant, and checks its own coverage — exiting non-zero if any public function
goes unexercised. Pass `--refresh` to re-download.

## Development

Run the test suite with:

```bash
pip install -e '.[dev]'
pytest
```

The tests use synthetic data throughout and never touch the network, so they
run offline and deterministically.

## Usage

### Data Retrieval Functions

#### import_aq_meta
Fetches metadata for air quality monitoring sites from specified sources.

```python
import pandas as pd
from PythonAQ import import_aq_meta

metadata_df = import_aq_meta(source='aurn')
```
**Parameters:**
- `source` (str): Source identifier. Options are "aurn", "saqn", "aqe", "waqn", "ni".

**Returns:**
- `pd.DataFrame`: DataFrame containing site metadata.

#### download_aurn_data
Downloads and processes air quality data from AURN for a specified site and year range.

```python
from PythonAQ import download_aurn_data

data_df = download_aurn_data(site='LEED', start_year=2020, end_year=2021, source='aurn')
```
**Parameters:**
- `site` (str): Site identifier.
- `start_year` (int): Starting year.
- `end_year` (int): Ending year.
- `source` (str): Source identifier.

**Returns:**
- `pd.DataFrame`: Combined DataFrame with data from all requested years.

#### download_noaa_data
Downloads and processes meteorological data from NOAA for a specified station and year range.

```python
from PythonAQ import download_noaa_data

noaa_df = download_noaa_data(station_code='725300-94846', year_start=2020, year_end=2021)
```
**Parameters:**
- `station_code` (str): NOAA station code.
- `year_start` (int): Starting year.
- `year_end` (int): Ending year.

**Returns:**
- `pd.DataFrame`: Combined DataFrame with processed meteorological data.

### Data Parsing Functions

#### parse_noaa_data
Parses raw NOAA data and extracts relevant meteorological parameters.

```python
from PythonAQ import parse_noaa_data

parsed_df = parse_noaa_data(raw_noaa_df)
```
**Parameters:**
- `data` (pd.DataFrame): Raw NOAA data DataFrame.

**Returns:**
- `pd.DataFrame`: Processed DataFrame with meteorological parameters.

### Visualisation Functions

#### calendar
Creates a calendar heatmap visualisation of the provided data.

```python
from PythonAQ import calendar

fig = calendar(data_df, value_column='PM10')
fig.show()
```
**Parameters:**
- `data` (pd.DataFrame): Data containing the values to plot.
- `value_column` (str): Column name for the values to aggregate and plot.

**Returns:**
- `go.Figure`: Plotly Figure object of the calendar heatmap.

#### map_sites
Plots the locations of air quality monitoring sites on a map.

```python
from PythonAQ import map_sites

fig = map_sites(metadata_df, sites=['LEED', 'LED6'])
fig.show()
```
**Parameters:**
- `data` (pd.DataFrame): Metadata with columns site_id, latitude, and longitude.
- `sites` (list): List of site identifiers to map.

**Returns:**
- `go.Figure`: Plotly Figure object showing site locations.

#### polar_cluster
Creates a polar plot with clustering based on specified features.

```python
from PythonAQ import polar_cluster

fig = polar_cluster(data_df, feature_cols=['PM10', 'NO2'])
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Data containing wind data and features.
- `feature_cols` (list): Columns to use for clustering.

**Returns:**
- `go.Figure`: Polar cluster plot.

#### polar_frequency_plot
Creates a polar frequency plot of wind speed and direction distributions.

```python
from PythonAQ import polar_frequency_plot

fig = polar_frequency_plot(data_df, separate_by_year=True)
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Data containing wind data.
- `separate_by_year` (bool): Whether to separate data by year.

**Returns:**
- `go.Figure`: Polar frequency plot.

#### polar_plot
Generates a polar plot of pollutant concentrations varying with wind speed and
direction, smoothed with a tensor-product GAM.

```python
from PythonAQ import polar_plot

fig = polar_plot(data_df, conc_col='NO2')
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Data containing wind and concentration data.
- `conc_col` (str): Column name for concentration.
- `render` (str): `'raster'` (default) draws one continuous surface,
  `'contour'` draws filled contour bands, `'tile'` is the original
  one-polygon-per-bin rendering.
- `resolution` (int): Grid points per axis for the predicted surface, default
  300. Higher is smoother and slower.
- `ws_limit` (str or float): Radial extent. `'auto'` (default) uses the 99th
  percentile of wind speed, `'max'` the full observed range, or give a number.
- `min_count` (int): Minimum observations per bin, and the local density
  required for a grid cell to be drawn.
- `exclude_missing` (bool), `exclude_distance` (float or None): Blank areas of
  the grid too far from any observation, rather than extrapolating into empty
  sectors.
- `uncertainty` (float or None): Confidence width, e.g. `0.95`, for blanking
  regions where the prediction interval is wider than the prediction itself.
- `upper_limit` (float or None): Blank predictions above this concentration.

**Returns:**
- `go.Figure`: Polar plot of concentration.

> **On smoothness:** the surface is predicted onto a regular Cartesian grid in
> wind-component space and drawn as a single interpolated raster, which is what
> openair does via lattice's `levelplot`. Drawing one flat-filled polygon per
> bin — the old behaviour, still available as `render='tile'` — looks blocky
> however smooth the underlying fit is, because nothing interpolates between
> neighbouring cells.

#### pollutant_rose
Creates a pollutant rose plot displaying pollutant concentrations by wind direction.

```python
from PythonAQ import pollutant_rose

fig, summary_df = pollutant_rose(data_df, pollutant='NO2', group_by='year')
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Data containing wind and pollutant data.
- `pollutant` (str): Column name for pollutant concentration.

**Returns:**
- `go.Figure`: Pollutant rose figure.
- `pd.DataFrame`: Summary statistics.

#### smooth_trend_plot
Fits a non-parametric smooth trend (a GAM) to averaged time series data, with a
confidence band and a Mann-Kendall significance annotation.

```python
from PythonAQ import smooth_trend_plot

fig = smooth_trend_plot(data_df, pollutant_col='NO2', avg_freq='MS')
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Input data.
- `date_col` (str): Datetime column, defaults to `'date_time'`.
- `pollutant_col` (str): Column for analysis.
- `avg_freq` (str): Averaging frequency, defaults to `'MS'` (month start).
- `deseason` (bool): Whether to remove the seasonal cycle via STL first.
- `alpha` (float): Significance level for the confidence band.

**Returns:**
- `go.Figure`: Smooth trend plot.

#### summary_plot
Generates a comprehensive data summary plot and statistics.

```python
from PythonAQ import summary_plot

fig, summary_df = summary_plot(data_df)
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Input data.

**Returns:**
- `go.Figure`: Data summary plot.
- `pd.DataFrame`: Summary statistics.

#### theil_sen_plot
Performs Theil-Sen regression analysis and plots the time series data.  Deseasoned data can be added to the plot using an additional dataframe and is calculated using the deseason_data function.

```python
from PythonAQ import theil_sen_plot

fig = theil_sen_plot(data_df, pollutant_col='NO2', agg_freq='ME')
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Input data.
- `pollutant_col` (str): Column for analysis.
- `deseason_data` (pd.DataFrame): Optional deseasoned data for plotting

**Returns:**
- `go.Figure`: Theil-Sen regression plot.

#### time_plot
Plots time series data with options for grouping, stacking, and normalisation.

```python
from PythonAQ import time_plot

fig = time_plot(data_df, columns_to_plot=['NO2', 'PM10'], group_data=True)
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Input data.
- `columns_to_plot` (list): Columns to plot.

**Returns:**
- `go.Figure`: Time series plot.

#### wind_rose
Generates a wind rose plot displaying wind speed and direction distributions.

```python
from PythonAQ import wind_rose

fig, summary_df = wind_rose(data_df, group_by='year')
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Data containing wind data.
- `group_by` (str): Grouping option ('none', 'year', or 'quartile').

**Returns:**
- `go.Figure`: Wind rose figure.
- `pd.DataFrame`: Summary statistics.

#### percentile_rose
Shows how the *distribution* of a pollutant, rather than just its mean, varies
with wind direction. Port of openair's `percentileRose`.

```python
from PythonAQ import percentile_rose

fig, summary_df = percentile_rose(data_df, 'NO2', percentile=[25, 50, 75, 95])
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Data containing wind direction and the pollutant.
- `pollutant` (str): Column to summarise.
- `percentile` (sequence): Percentiles to draw, e.g. `[25, 50, 75, 90, 95]`.
- `direction_bins` (int): Number of wind direction sectors, default 36.
- `fill` (bool), `mean_line` (bool): Styling toggles.

**Returns:**
- `go.Figure`: Percentile rose.
- `pd.DataFrame`: Percentile value per direction bin.

#### time_variation
The four-panel temporal summary: hour of day split by weekday, day of week,
hour of day, and month of year, each with bootstrap confidence intervals in the
mean. Port of openair's `timeVariation`.

```python
from PythonAQ import time_variation

fig, summary_df = time_variation(data_df, ['NO2', 'PM10'], random_state=42)
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Input data.
- `pollutant` (str or list): One or more columns to plot.
- `statistic` (str): `'mean'` or `'median'`.
- `conf_int` (float): Confidence level, default `0.95`.
- `n_boot` (int): Bootstrap replicates (openair's `B`), default 100.
- `normalise` (bool): Divide each series by its mean, to compare shapes across
  variables on different scales.
- `ci` (bool): Whether to draw the confidence bands.
- `random_state` (int or None): Seed, for reproducible intervals.

**Returns:**
- `go.Figure`: Four-panel figure.
- `pd.DataFrame`: Long-format table of every plotted statistic.

#### trend_level
Heat map of a pollutant over two time dimensions, optionally split into panels.
Port of openair's `trendLevel`.

```python
from PythonAQ import trend_level

fig, summary_df = trend_level(data_df, 'NO2', x='month', y='hour', type='year')
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Input data.
- `pollutant` (str): Column to aggregate.
- `x`, `y` (str): Axis variables, from `'hour'`, `'weekday'`, `'month'`,
  `'year'`, `'season'`, `'monthyear'`, `'wd'`, or any existing column.
- `type` (str or None): Variable to split into panels; `None` for a single panel.
- `statistic` (str): `'mean'`, `'median'`, `'max'`, `'min'` or `'frequency'`.

**Returns:**
- `go.Figure`: Heat map.
- `pd.DataFrame`: Long-format aggregated values.

#### scatter_plot
Flexible scatter plots with optional linear and LOWESS fits. Port of openair's
`scatterPlot`.

```python
from PythonAQ import scatter_plot

fig, fit_df = scatter_plot(data_df, 'NOx', 'NO2', linear=True, smooth=True)
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Input data.
- `x`, `y` (str): Column names for the axes.
- `method` (str): `'scatter'`, `'hexbin'` or `'density'`.
- `colour_by` (str or None): Column used to colour points.
- `linear` (bool): Add a least-squares line, annotated with slope and R².
- `smooth` (bool): Add a LOWESS smooth line.
- `one_to_one` (bool): Draw the 1:1 line, for model evaluation.

**Returns:**
- `go.Figure`: Scatter plot.
- `pd.DataFrame`: Fit statistics; empty if no fit was requested.

#### corr_plot
Correlation matrix between pollutants, optionally ordered by hierarchical
clustering so related species sit together. Port of openair's `corPlot`.

```python
from PythonAQ import corr_plot

fig, corr_df = corr_plot(data_df, ['NO2', 'NOx', 'O3', 'PM10'])
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Input data.
- `pollutants` (list or None): Columns to correlate; defaults to all numeric.
- `method` (str): `'pearson'`, `'spearman'` or `'kendall'`.
- `cluster` (bool): Order variables by hierarchical clustering.
- `annotate` (bool): Print the coefficient in each cell.

**Returns:**
- `go.Figure`: Correlation heat map.
- `pd.DataFrame`: The correlation matrix, in plotted order.

### Statistics

#### mod_stats
Standard model evaluation statistics. Port of openair's `modStats`, following
the R source formulas exactly.

```python
from PythonAQ import mod_stats

stats_df = mod_stats(data_df, mod='model', obs='observed')
```
**Parameters:**
- `df` (pd.DataFrame): Data containing the modelled and observed columns.
- `mod`, `obs` (str): Modelled and observed column names.
- `group_by` (str, list or None): Column(s) to compute statistics within.

**Returns:**
- `pd.DataFrame`: One row per group with `n`, `FAC2`, `MB`, `MGE`, `NMB`,
  `NMGE`, `RMSE`, `r`, `P`, `COE` and `IOA`.

Notes on interpretation:
- **FAC2** — fraction of predictions within a factor of two, i.e. `0.5 <= mod/obs <= 2`.
- **COE** — 1 for a perfect model, 0 when no better than the observed mean,
  negative when worse.
- **IOA** — spans -1 to +1; piecewise, per Willmott et al. (2011).

#### aq_stats
Annual air quality summary statistics, including exceedance counts against the
UK objectives. Port of openair's `aqStats`.

```python
from PythonAQ import aq_stats

summary_df = aq_stats(data_df, 'NO2')
```
**Parameters:**
- `df` (pd.DataFrame): Input data, at hourly resolution.
- `pollutant` (str): Column to summarise.
- `data_thresh` (float): Minimum data capture percentage per year.
- `percentile` (sequence): Additional percentiles to report.
- `transpose` (bool): Return statistics as rows rather than columns.

**Returns:**
- `pd.DataFrame`: One row per year with data capture, mean, min, max, median,
  requested percentiles, maximum daily and rolling 8-hour means, and
  exceedance counts where an objective is defined for the pollutant.

### Utilities
The utilities functions are various helper functions to aid the rest of the functionality.

#### time_average
Averages a time series over a period, honouring a data-capture threshold.
Wind direction is averaged as a **vector**, so northerly winds either side of
360° do not average to south. Port of openair's `timeAverage`.

```python
from PythonAQ import time_average

daily = time_average(data_df, avg_time='day', data_thresh=75)
```
**Parameters:**
- `df` (pd.DataFrame): Input data.
- `avg_time` (str): `'hour'`, `'day'`, `'3 day'`, `'month'`, `'year'`, or any
  pandas offset alias.
- `data_thresh` (float): Minimum percentage of data required in a period;
  periods below this become NaN.
- `statistic` (str): `'mean'`, `'median'`, `'max'`, `'min'`, `'sum'`, `'sd'`,
  `'frequency'`, `'data.cap'` or `'percentile'`.
- `percentile` (float): Required when `statistic='percentile'`.

**Returns:**
- `pd.DataFrame`: Averaged data.

#### select_by_date
Subsets a DataFrame by date components, combining all criteria with AND.
Port of openair's `selectByDate`.

```python
from PythonAQ import select_by_date

subset = select_by_date(data_df, year=2021, month='June', day=['Sat', 'Sun'])
```
**Parameters:**
- `df` (pd.DataFrame): Input data.
- `start`, `end` (str or datetime): Inclusive date range bounds. A bare end
  date includes the whole of that day.
- `year`, `month`, `day`, `hour` (int, str or sequence): Components to keep.
  `month` accepts names or numbers; `day` accepts weekday names or month-days.
- `season` (str or sequence): e.g. `'summer'`.

**Returns:**
- `pd.DataFrame`: The matching subset.

#### rolling_mean
Rolling mean with a data-capture threshold. The default 8-point window matches
the running 8-hour mean used for ozone. Port of openair's `rollingMean`.

```python
from PythonAQ import rolling_mean

result = rolling_mean(data_df, 'O3', width=8, data_thresh=75)
```
**Parameters:**
- `df` (pd.DataFrame): Input data.
- `pollutant` (str): Column to smooth.
- `width` (int): Window width in observations.
- `data_thresh` (float): Minimum percentage of valid values in a window.
- `align` (str): `'centre'`, `'left'` or `'right'`.

**Returns:**
- `pd.DataFrame`: Input data with the rolling mean column appended.

#### cut_data
Adds a conditioning column for splitting data. Port of openair's `cutData`.

```python
from PythonAQ import cut_data

result = cut_data(data_df, type='season')
```
**Parameters:**
- `df` (pd.DataFrame): Input data.
- `type` (str): `'year'`, `'month'`, `'monthyear'`, `'season'`, `'seasonyear'`,
  `'weekday'`, `'weekend'`, `'hour'`, `'daylight'`, `'wd'`, or the name of a
  numeric column (split into `n_levels` quantiles).
- `hemisphere` (str): `'northern'` or `'southern'`, for season definitions.

**Returns:**
- `pd.DataFrame`: Input data with a categorical column named `type` added.

#### calc_percentile
Calculates percentiles of a pollutant over a period. Port of openair's
`calcPercentile`.

```python
from PythonAQ import calc_percentile

result = calc_percentile(data_df, 'NO2', percentile=[25, 50, 75, 95], avg_time='month')
```
**Parameters:**
- `df` (pd.DataFrame): Input data.
- `pollutant` (str): Column to summarise.
- `percentile` (sequence): Percentiles to compute.
- `avg_time` (str): Averaging period, as for `time_average`.

**Returns:**
- `pd.DataFrame`: One column per requested percentile, named `percentile.N`.

#### deseason_data
A function to deseason data.

```python
from PythonAQ import deseason_data, get_period

ds = deseason_data(
    data=df,
    pollutant_column='NO',
    interval='7D',
    period=get_period('7D'),
    method='additive',
    date_column='date_time',
)
```

**deseason_data Parameters**
- `data` (pd.DataFrame): Data frame containing the data to be deseasoned.
- `pollutant_column` (str): Column heading for the pollutant to be analysed
- `interval` (str): The time interval to be averaged, e.g. `'h'`, `'D'`, `'7D'`, `'ME'`, `'QE'`, `'YE'`.
- `period` (int): The period for the deseasoning algorith.  This can be solved in terms of the interval by using the get_period() function
- `method` (str): The method for performing the deseasoning
- `date_column` (str): The location of the date and time information in the dataset, defaults to 'date_time'.

**Returns:**
- `pd.DataFrame`: Data frame containing deseasoned data in the column deseasoned_{pollutant_column}

#### get_period
A function to convert the pandas time series strings into appropriate values for the deseasoning algorithm

```python
from PythonAQ import get_period

period = get_period('7D')
```

**get_period Parameters**
- `interval` (str): The interval to be conveted into 

**Returns:**
- `int`: Number of periods in one seasonal cycle (e.g., a year).

#### e_sat and rh
Functions to calculate saturation vapour pressure and relative humidity.

```python
from PythonAQ import e_sat, rh

e_saturation = e_sat(25)  # Temperature in Celsius
relative_humidity = rh(25, 20)
```
**e_sat Parameters:**
- `T_obs` (float or array-like): Temperature in Celsius.

**e_sat Returns:**
- `float or ndarray`: Saturation vapour pressure in millibars.

**rh Parameters:**
- `T` (float or array-like): Air temperature in Celsius.
- `T_d` (float or array-like): Dew point temperature in Celsius.

**rh Returns:**
- `float or ndarray`: Relative humidity as a percentage.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes or additions.

## Licence
This project is licensed under the MIT Licence.