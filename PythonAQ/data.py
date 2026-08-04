import requests
import rdata
import pandas as pd
import logging 
import numpy as np

from io import BytesIO, StringIO
#from datetime import datetime
from .utilities import rh


def get_r_data(url):
    """
    Fetches and converts RData from a given URL.
    This function sends a GET request to the specified URL to retrieve RData.
    It then parses and converts the RData into a Python-readable format.
    Args:
        url (str): The URL from which to fetch the RData.
    Returns:
        object: The converted RData.
    Raises:
        requests.exceptions.RequestException: If the request fails or the status code is not 200.
    """
    try:
        response = requests.get(url, timeout=10)  # Timeout added for robustness
        response.raise_for_status()  # Raise error if the status code is not 200

        # Load the RData into memory
        rdata_in_memory = BytesIO(response.content)
        parsed = rdata.parser.parse_file(rdata_in_memory)
        converted = rdata.conversion.convert(parsed)

        return converted

    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, 
            requests.exceptions.Timeout) as network_err:
        logging.error(f"Network error for URL {url}: {network_err}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Failed to retrieve data from URL {url}: {req_err}")
    except Exception as e:
        logging.error(f"An unexpected error occurred for URL {url}: {e}")
    
    return None

def import_aq_meta(source: str) -> pd.DataFrame:
    """
    Imports air quality metadata from a specified source.
    
    This function fetches metadata from a given source URL, processes it,
    and returns a pandas DataFrame containing the metadata.
    
    Parameters:
    -----------
    source : str
        The source identifier for the metadata. Valid options are:
        "aurn", "saqn", "aqe", "waqn", "ni".
        
    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the metadata, with duplicates removed.
        
    Raises:
    -------
    ValueError
        If the provided source is not one of the valid options.
    """
    source_dict = {
        "aurn": "http://uk-air.defra.gov.uk/openair/R_data/AURN_metadata.RData",
        "saqn": "https://www.scottishairquality.scot/openair/R_data/SCOT_metadata.RData",
        "aqe": "https://airqualityengland.co.uk/assets/openair/R_data/AQE_metadata.RData",
        "waqn": "https://airquality.gov.wales/sites/default/files/openair/R_data/WAQ_metadata.RData",
        "ni": "https://www.airqualityni.co.uk/openair/R_data/NI_metadata.RData"
    }

    key_dict = {
        "aurn": "AURN_metadata",
        "saqn": "meta",
        "aqe": "metadata",
        "waqn": "metadata",
        "ni": "metadata"

    }
    
    if source not in source_dict:
        raise ValueError(f"Invalid source '{source}'. Valid options are: {', '.join(source_dict.keys())}")
    
    try:
        df = get_r_data(source_dict[source])
        df = df[key_dict[source]]
        if df is not None:
            df = df.drop_duplicates(subset=['site_id'])
            return df
        else:
            logging.error(f"No data returned for source '{source}'")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"An error occurred while importing metadata for source '{source}': {e}")
        return pd.DataFrame()

def download_aurn_data(site: str, start_year: int, end_year: int, source: str) -> pd.DataFrame:
    """
    Downloads AURN data for a specified site and range of years, 
    processes the data, and returns a combined pandas DataFrame.
    
    Parameters:
    -----------
    site : str
        The site identifier for the AURN data.
    start_year : int
        The starting year of the data to download.
    end_year : int
        The ending year of the data to download.
    source : str
        The source identifier for the metadata. Valid options are:
        "aurn", "saqn", "aqe", "waqn", "ni".
        
    Returns:
    --------
    pd.DataFrame
        A combined DataFrame with data from all requested years, or 
        an empty DataFrame if no data is available.
    """

    source_dict = {"aurn":"https://uk-air.defra.gov.uk/openair/R_data/",
                   "saqn":"https://www.scottishairquality.scot/openair/R_data/",
                   "aqe":"https://airqualityengland.co.uk/assets/openair/R_data/",
                   "waqn":"https://airquality.gov.wales/sites/default/files/openair/R_data/",
                   "ni":"https://www.airqualityni.co.uk/openair/R_data/"}

    years = list(range(start_year, end_year + 1))
    #url_stub = 'https://uk-air.defra.gov.uk/openair/R_data/'
    url_stub = source_dict[source]
    all_data = []  # List to store DataFrames

    for y in years:
        site_id = f'{site}_{y}'
        url = f'{url_stub}{site_id}.RData'

        try:
            # Fetch the RData file
            response = requests.get(url, timeout=10)  # Timeout added for robustness
            response.raise_for_status()  # Raise error if the status code is not 200

            # Load the RData into memory
            rdata_in_memory = BytesIO(response.content)
            parsed = rdata.parser.parse_file(rdata_in_memory)
            converted = rdata.conversion.convert(parsed)

            # Process the data
            data = converted.get(site_id)
            if data is not None:
                # Convert the date field from UNIX timestamp
                data['date_time'] = pd.to_datetime(data['date'], unit='s')
                all_data.append(data)
            else:
                logging.warning(f'No data found for site {site_id}')

        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout) as network_err:
            logging.error(f"Network error for {site_id} at {url}: {network_err}")
        except requests.exceptions.RequestException as req_err:
            logging.error(f"Failed to retrieve data for {site_id} at {url}: {req_err}")
        except KeyError:
            logging.error(f"Key {site_id} not found in the parsed data.")
        except Exception as e:
            logging.error(f"An unexpected error occurred for {site_id}: {e}")

    # Concatenate all collected data if any exists
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    else:
        logging.error('No data available for the given site and year range.')
        return pd.DataFrame()
    
def download_noaa_data(station_code: str, year_start: int, year_end: int) -> pd.DataFrame:
    """
    Downloads NOAA data for a specified station and range of years,
    processes the data, and returns a combined pandas DataFrame.

    Parameters:
    -----------
    year_start : int
        The starting year of the data to download.
    year_end : int
        The ending year of the data to download.
    station_code : str
        The station code for the NOAA data.

    Returns:
    --------
    pd.DataFrame
        A combined DataFrame with data from all requested years.
    """
    # Remove any hyphens from the station code
    station_code = station_code.replace('-', '')

    # Validate the year range
    if year_start > year_end:
        raise ValueError("Start year must be less than or equal to end year.")

    # Generate the list of years
    years = list(range(year_start, year_end + 1))

    data_frames = []

    for year in years:
        url = f'https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station_code}.csv'

        try:
            # Fetch the CSV file
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Read the CSV content into a pandas DataFrame
            df_temp = pd.read_csv(StringIO(response.text))

            # Process the data (Assuming parse_noaa_data is defined elsewhere)
            df_temp_processed = parse_noaa_data(df_temp)

            # Append the processed DataFrame to the list
            data_frames.append(df_temp_processed)

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred for year {year}: {http_err}")
            continue
        except requests.exceptions.ConnectionError as conn_err:
            logging.error(f"Connection error occurred for year {year}: {conn_err}")
            continue
        except requests.exceptions.Timeout as timeout_err:
            logging.error(f"Timeout error for year {year}: {timeout_err}")
            continue
        except requests.exceptions.RequestException as req_err:
            logging.error(f"Request exception for year {year}: {req_err}")
            continue
        except pd.errors.EmptyDataError:
            logging.warning(f"No data found for year {year} at {url}")
            continue
        except Exception as e:
            logging.error(f"An unexpected error occurred for year {year}: {e}")
            continue

    # Concatenate all collected data if any exists
    if data_frames:
        combined_data = pd.concat(data_frames, ignore_index=False).reset_index()
        return combined_data
    else:
        logging.error("No data was downloaded for the given station and year range.")
        return pd.DataFrame()

def parse_noaa_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Parses NOAA data and returns a cleaned DataFrame with relevant meteorological parameters.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The raw NOAA data as a pandas DataFrame.
        
    Returns:
    --------
    pd.DataFrame
        A DataFrame containing processed data with columns:
        - 'station'
        - 'date'
        - 'air_temp'
        - 'ws' (wind speed)
        - 'wd' (wind direction)
        - 'dew_point'
        - 'atmospheric_pressure'
    """
    if data.empty:
        logging.warning("parse_noaa_data received an empty DataFrame.")
        return pd.DataFrame()

    site_id = str(data['STATION'].iloc[0]).replace('99999', '-99999')
    try:
        # Initialize a dictionary to hold processed data
        processed_data = {}

        def _field(column, index, missing, scale=1.0):
            """Extract one comma-separated ISD sub-field.

            The NOAA ISD format stores several values per column, each with its
            own 'missing' sentinel and scaling factor. The sentinel must be
            cleared *before* scaling, otherwise the sentinel is silently
            converted into a plausible-looking physical value.
            """
            values = pd.to_numeric(
                data[column].astype(str).str.split(',').str[index], errors='coerce'
            )
            return values.replace(missing, np.nan) / scale

        # Process 'DEW' column if present (tenths of degrees Celsius)
        if 'DEW' in data.columns:
            processed_data['dew_point'] = _field('DEW', 0, 9999, scale=10.0)

        # Process 'SLP' column if present (tenths of hectopascals)
        if 'SLP' in data.columns:
            processed_data['atmospheric_pressure'] = _field('SLP', 0, 99999, scale=10.0)

        # Process 'TMP' column if present (tenths of degrees Celsius)
        if 'TMP' in data.columns:
            processed_data['air_temp'] = _field('TMP', 0, 9999, scale=10.0)

        # Process 'WND' column if present
        if 'WND' in data.columns:
            # Sub-fields: direction, direction quality, type, speed, speed quality
            processed_data['wd'] = _field('WND', 0, 999)               # angular degrees
            processed_data['ws'] = _field('WND', 3, 9999, scale=10.0)  # tenths of m/s

        # Extract station and date information
        processed_data['station'] = data.get('STATION')
        processed_data['date_string'] = data.get('DATE')

        # Convert date strings to datetime objects
        processed_data['date'] = pd.to_datetime(processed_data['date_string'], errors='coerce')
        if processed_data['date'].isnull().any():
            logging.warning("Some dates could not be parsed and will be set as NaT.")

        # Create the final DataFrame
        new_data = pd.DataFrame(processed_data)

        # Set 'date' as the index
        new_data.set_index('date', inplace=True)
        
        # Identify the numeric columns for resampling. Only keep the ones that
        # were actually present in the source file, otherwise a station that
        # does not report (say) SLP would raise a KeyError.
        numeric_cols = [
            col for col in ['air_temp', 'ws', 'wd', 'dew_point', 'atmospheric_pressure']
            if col in new_data.columns
        ]
        # Resample numeric data hourly and compute the mean.
        # Note: 'h' rather than 'H', which pandas 3 removed.
        numeric_data = new_data[numeric_cols].resample('h').mean()
        numeric_data['site'] = site_id

        # Calculate relative humidity
        if 'air_temp' in numeric_data and 'dew_point' in numeric_data:
            air_temp = numeric_data['air_temp']
            dew_point = numeric_data['dew_point']
            numeric_data['relative_humidity'] = rh(air_temp, dew_point)

        return numeric_data

    except Exception as e:
        logging.error(f"An error occurred while parsing NOAA data: {e}")
        
        
        return pd.DataFrame()