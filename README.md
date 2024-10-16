Air Quality Data Analysis and Visualization Toolkit
This repository provides a comprehensive set of Python tools for downloading, processing, and visualizing air quality and meteorological data. The toolkit is designed for researchers, scientists, and environmental analysts who need to work with air quality datasets from various sources, such as AURN, NOAA, and others.

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
    - [Visualization Functions](#visualization-functions)
        - [calendar](#calendar)
        - [map_sites](#map_sites)
        - [polar_cluster](#polar_cluster)
        - [polar_frequency_plot](#polar_frequency_plot)
        - [polar_plot](#polar_plot)
        - [pollutant_rose](#pollutant_rose)
        - [summary_plot](#summary_plot)
        - [theil_sen_plot](#theil_sen_plot)
        - [time_plot](#time_plot)
        - [wind_rose](#wind_rose)
    - [Utilities](#utilities)
        - [e_sat and rh](#e_sat-and-rh)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Data Retrieval**: Functions to download air quality and meteorological data from various sources.
- **Data Parsing**: Utilities to parse and clean raw data into usable formats.
- **Visualization**: Advanced plotting functions using Plotly for interactive data visualization.
- **Statistical Analysis**: Tools for performing regression analysis and clustering.
- **Meteorological Calculations**: Functions to calculate relative humidity and saturation vapor pressure.

## Installation
To use this toolkit, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/air-quality-toolkit.git
cd air-quality-toolkit
pip install -r requirements.txt
```

## Dependencies
- Python 3.7 or higher
- requests
- rdata
- pandas
- numpy
- plotly
- pygam
- scikit-learn
- streamlit (for testing purposes)

## Usage

### Data Retrieval Functions

#### import_aq_meta
Fetches metadata for air quality monitoring sites from specified sources.

```python
import pandas as pd
from your_module import import_aq_meta

metadata_df = import_aq_meta(source='aurn')
```
**Parameters:**
- `source` (str): Source identifier. Options are "aurn", "saqn", "aqe", "waqn", "ni".

**Returns:**
- `pd.DataFrame`: DataFrame containing site metadata.

#### download_aurn_data
Downloads and processes air quality data from AURN for a specified site and year range.

```python
from your_module import download_aurn_data

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
from your_module import download_noaa_data

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
from your_module import parse_noaa_data

parsed_df = parse_noaa_data(raw_noaa_df)
```
**Parameters:**
- `data` (pd.DataFrame): Raw NOAA data DataFrame.

**Returns:**
- `pd.DataFrame`: Processed DataFrame with meteorological parameters.

### Visualization Functions

#### calendar
Creates a calendar heatmap visualization of the provided data.

```python
from your_module import calendar

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
from your_module import map_sites

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
from your_module import polar_cluster

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
from your_module import polar_frequency_plot

fig = polar_frequency_plot(data_df, separate_by_year=True)
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Data containing wind data.
- `separate_by_year` (bool): Whether to separate data by year.

**Returns:**
- `go.Figure`: Polar frequency plot.

#### polar_plot
Generates a polar plot of pollutant concentrations varying with wind speed and direction.

```python
from your_module import polar_plot

fig = polar_plot(data_df, conc_col='NO2')
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Data containing wind and concentration data.
- `conc_col` (str): Column name for concentration.

**Returns:**
- `go.Figure`: Polar plot of concentration.

#### pollutant_rose
Creates a pollutant rose plot displaying pollutant concentrations by wind direction.

```python
from your_module import pollutant_rose

fig, summary_df = pollutant_rose(data_df, pollutant='NO2', group_by='year')
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Data containing wind and pollutant data.
- `pollutant` (str): Column name for pollutant concentration.

**Returns:**
- `go.Figure`: Pollutant rose figure.
- `pd.DataFrame`: Summary statistics.

#### summary_plot
Generates a comprehensive data summary plot and statistics.

```python
from your_module import summary_plot

fig, summary_df = summary_plot(data_df)
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Input data.

**Returns:**
- `go.Figure`: Data summary plot.
- `pd.DataFrame`: Summary statistics.

#### theil_sen_plot
Performs Theil-Sen regression analysis and plots the time series data.

```python
from your_module import theil_sen_plot

fig = theil_sen_plot(data_df, pollutant_col='NO2', agg_freq='M')
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Input data.
- `pollutant_col` (str): Column for analysis.

**Returns:**
- `go.Figure`: Theil-Sen regression plot.

#### time_plot
Plots time series data with options for grouping, stacking, and normalization.

```python
from your_module import time_plot

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
from your_module import wind_rose

fig, summary_df = wind_rose(data_df, group_by='year')
fig.show()
```
**Parameters:**
- `df` (pd.DataFrame): Data containing wind data.
- `group_by` (str): Grouping option ('none', 'year', or 'quartile').

**Returns:**
- `go.Figure`: Wind rose figure.
- `pd.DataFrame`: Summary statistics.

### Utilities

#### e_sat and rh
Functions to calculate saturation vapor pressure and relative humidity.

```python
from your_module import e_sat, rh

e_saturation = e_sat(25)  # Temperature in Celsius
relative_humidity = rh(25, 20)
```
**e_sat Parameters:**
- `T_obs` (float or array-like): Temperature in Celsius.

**e_sat Returns:**
- `float or ndarray`: Saturation vapor pressure in millibars.

**rh Parameters:**
- `T` (float or array-like): Air temperature in Celsius.
- `T_d` (float or array-like): Dew point temperature in Celsius.

**rh Returns:**
- `float or ndarray`: Relative humidity as a percentage.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes or additions.

## License
This project is licensed under the MIT License.