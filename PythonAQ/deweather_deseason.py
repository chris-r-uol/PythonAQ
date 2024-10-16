import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st

def deseason_data(data, pollutant_column, interval, period, method='additive', date_column='date_time'):
    """
    Removes seasonal components from time series data using seasonal decomposition.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the data with a datetime column.
    - pollutant_column (str): Name of the column with the pollutant data.
    - interval (str): Resampling interval (e.g., 'D' for daily).
    - period (int): Period for seasonal decomposition.
    - method (str): Type of seasonal component ('additive' or 'multiplicative'). Default is 'additive'.
    - date_column (str): Name of the datetime column. Default is 'date_time'.

    Returns:
    - pd.DataFrame: DataFrame with the deseasoned data added as a new column.
    """

    # Copy data to avoid modifying the original DataFrame
    df = data.copy()

    # Ensure date_column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Set date_column as the index
    df.set_index(date_column, inplace=True)

    # Resample the data
    # Ensure pollutant_column is numeric
    df[pollutant_column] = pd.to_numeric(df[pollutant_column], errors='coerce')
    
    # Resample the data for the pollutant_column only
    resampled_df = df[[pollutant_column]].resample(interval).mean()


    # Interpolate missing values in the pollutant column
    resampled_df[pollutant_column].interpolate(method='linear', inplace=True)

    # Check if data length is sufficient for seasonal decomposition
    required_length = 2 * period
    if len(resampled_df) < required_length:
        raise ValueError(f"Data length must be at least two times the period ({required_length} observations).")

    # Perform seasonal decomposition
    decomposition = seasonal_decompose(resampled_df[pollutant_column], model=method, period=period)

    # Remove the seasonal component
    if method == 'additive':
        deseasoned = resampled_df[pollutant_column] - decomposition.seasonal
    elif method == 'multiplicative':
        deseasoned = resampled_df[pollutant_column] / decomposition.seasonal.replace(0, pd.NA)
        deseasoned = deseasoned.fillna(method='bfill').fillna(method='ffill')
    else:
        raise ValueError("Method must be 'additive' or 'multiplicative'.")

    # Add the deseasoned data to the DataFrame
    resampled_df[f'deseasoned_{pollutant_column}'] = deseasoned

    return resampled_df
