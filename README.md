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
        - [summary_plot](#summary_plot)
        - [theil_sen_plot](#theil_sen_plot)
        - [time_plot](#time_plot)
        - [wind_rose](#wind_rose)
    - [Utilities](#utilities)
        - [deseason_data](#deseason_data)
        - [get_period](#get_period)
        - [e_sat and rh](#e_sat-and-rh)
- [Contributing](#contributing)
- [Licence](#licence)

## Features
- **Data Retrieval**: Functions to download air quality and meteorological data from various sources.
- **Data Parsing**: Utilities to parse and clean raw data into usable formats.
- **Visualisation**: Advanced plotting functions using Plotly for interactive data visualisation.
- **Statistical Analysis**: Tools for performing regression analysis and clustering.
- **Meteorological Calculations**: Functions to calculate relative humidity and saturation vapour pressure.

## Installation
To use this toolkit, clone the repository and install the required dependencies:

```bash
git clone https://github.com/chris-r-uol/PythonAQ.git
cd PythonAQ
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
- streamlit (for testing and web-application purposes)

## Web Application Demo
A web application written using Streamlit (https://streamlit.io) is available as app.py. To run this application, first ensure Streamlit is installed and you are in the appropriate directory, then use the following commands to execute the programme.

```bash
streamlit run app.py
```

The web app has pre-loaded the current visualisation functions and implementations can be used elsewhere.

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
Generates a polar plot of pollutant concentrations varying with wind speed and direction.

```python
from PythonAQ import polar_plot

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

fig = theil_sen_plot(data_df, pollutant_col='NO2', agg_freq='M')
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

### Utilities
The utilities functions are various helper functions to aid the rest of the functionality.

#### deseason_data
A function to deseason data.

```python
from PythonAQ import deseason_data
ds = deseason_data(data=df, pollutant_column='NO', interval='7D', period=utilities.get_period('7D') , method='additive', date_column='date_time')
```

**deseason_data Parameters**
- `data` (pd.DataFrame): Data frame containing the data to be deseasoned.
- `pollutant_column` (str): Column heading for the pollutant to be analysed
- `interval` (str): The time interval to be averaged, can be H, D, M, Q, Y/A.
- `period` (int): The period for the deseasoning algorith.  This can be solved in terms of the interval by using the utilities.get_period() function
- `method` (str): The method for performing the deseasoning
- `date_column` (str): The location of the date and time information in the dataset, defaults to 'date_time'.

**Returns:**
- `pd.DataFrame`: Data frame containing deseasoned data in the column deseasoned_{pollutant_column}

#### get_period
A function to convert the pandas time series strings into appropriate values for the deseasoning algorithm

```python
from PythonAQ import get_period('7D')
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