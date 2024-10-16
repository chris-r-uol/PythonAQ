import pandas as pd
import plotly.express as px

def map_sites(data, sites=['LEED', 'LED6']):
    '''
    Function for mapping the AURN sites.
    
    Parameters:
    data (pd.DataFrame): Metadata for the site locations with columns 'site_id', 'latitude', and 'longitude'.
    sites (str or list): A single site identifier or a list of site identifiers to be mapped.
    
    Returns:
    fig (plotly.graph_objs._figure.Figure): A Plotly figure object representing the map of selected sites.
    
    Raises:
    ValueError: If input parameters are invalid or required columns are missing.
    '''
    # Validate input types
    if isinstance(sites, str):
        sites = [sites]
    elif isinstance(sites, (list, tuple)):
        if not all(isinstance(site, str) for site in sites):
            raise ValueError("All elements in 'sites' must be strings.")
    else:
        raise ValueError("'sites' parameter must be a string or a list/tuple of strings.")
    
    # Check for required columns in the DataFrame
    required_columns = {'site_id', 'latitude', 'longitude'}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    
    # Filter the data for the specified sites
    filtered_data = data[data['site_id'].isin(sites)]
    
    # Check if any sites were found
    if filtered_data.empty:
        raise ValueError("No sites found with the specified identifier(s).")
    
    # Create the Plotly map
    fig = px.scatter_mapbox(
        filtered_data,
        lat="latitude",
        lon="longitude",
        hover_name="site_id",
        zoom=5,
        height=600,
        width=800,
        mapbox_style="open-street-map",  # Alternative styles: "carto-positron", "stamen-terrain", etc.
        title="Map Showing Location of Sites"
    )
    
    # Customize marker appearance
    fig.update_traces(marker=dict(size=12, color='red', symbol='circle'))
    
    # Update layout for better appearance
    fig.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},
        title_x=0.5
    )
    
    return fig
