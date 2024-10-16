import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from math import ceil

def wind_rose(df, ws_col='ws', wd_col='wd', 
              date_col='date_time',  # Column for date/datetime
              direction_bins=16, speed_bins=None, 
              calms=True, speed_labels=None, 
              direction_labels=None, title='Wind Rose',
              colors=px.colors.sequential.Viridis,
              mode='count',
              group_by='quartile',  # 'none', 'year', or 'quartile'
              quartile_col='NO2',  # Column for quartile grouping
              subplot_cols=3,  # Number of columns in subplots
              fig_width=1200,   # Width of the figure in pixels
              fig_height=800    # Height of the figure in pixels
             ):
    """
    Creates a wind rose plot using Plotly, displaying bars by either percentage or count,
    and optionally grouping by year or quartiles of a specified column.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing wind data.
    - ws_col (str): Column name for wind speed.
    - wd_col (str): Column name for wind direction.
    - date_col (str or None): Column name containing date/datetime information for grouping by year.
    - direction_bins (int): Number of direction bins (e.g., 12 for 30° bins).
    - speed_bins (list or None): List defining wind speed categories. If None, defaults are used.
    - calms (bool): Whether to include calms (ws=0) in the plot.
    - speed_labels (list or None): Labels for wind speed categories. If None, defaults are used.
    - direction_labels (list or None): Labels for wind direction categories. If None, defaults are used.
    - title (str): Title of the plot.
    - colors (list): Plotly color scale for wind speed categories, default is px.colors.sequential.Viridis.
    - mode (str): 'percentage' to display bars as percentages, 'count' to display raw counts.
    - group_by (str): 'none' to plot all data together, 'year' to plot separate wind roses per year,
                      or 'quartile' to plot separate wind roses per quartile of a specified column.
    - quartile_col (str or None): Column name to use for quartile grouping. Required if group_by='quartile'.
    - subplot_cols (int): Number of subplot columns when grouping by year or quartile.
    - fig_width (int): Width of the entire figure in pixels.
    - fig_height (int): Height of the entire figure in pixels.

    Returns:
    - fig (plotly.graph_objects.Figure): The resulting wind rose figure.
    - summary_df (pd.DataFrame): DataFrame containing summary statistics.
    """
    # Validate 'mode' parameter
    if mode not in ['percentage', 'count']:
        raise ValueError("Invalid mode. Choose either 'percentage' or 'count'.")
    
    # Validate 'group_by' parameter
    if group_by not in ['none', 'year', 'quartile']:
        raise ValueError("Invalid group_by option. Choose 'none', 'year', or 'quartile'.")
    
    # Handle grouping logic
    if group_by == 'year':
        if date_col is None:
            raise ValueError("date_col must be provided when group_by='year'.")
        # Ensure date_col is in datetime format
        if not np.issubdtype(df[date_col].dtype, np.datetime64):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        # Drop rows with invalid dates
        df = df.dropna(subset=[date_col])
        # Extract year
        df['year'] = df[date_col].dt.year
        group_column = 'year'
    elif group_by == 'quartile':
        if quartile_col is None:
            raise ValueError("quartile_col must be provided when group_by='quartile'.")
        if not np.issubdtype(df[quartile_col].dtype, np.number):
            raise ValueError("quartile_col must be a numeric column.")
        # Drop rows with missing quartile_col
        df = df.dropna(subset=[quartile_col])
        # Calculate quartiles
        df['quartile'] = pd.qcut(df[quartile_col], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        group_column = 'quartile'
    else:
        group_column = None  # 'none' grouping
    
    # Copy relevant columns, including 'year' or 'quartile' if grouping
    if group_column:
        data = df[[ws_col, wd_col, group_column]].copy()
    else:
        data = df[[ws_col, wd_col]].copy()
    
    # Handle missing or invalid data
    data = data.dropna(subset=[ws_col, wd_col])
    data = data[(data[ws_col] >= 0) & (data[wd_col] >= 0) & (data[wd_col] <= 360)]
    
    # Define default speed bins if not provided
    if speed_bins is None:
        speed_bins = [0, 1, 3, 5, 7, 10, 20]
        speed_labels = ['0-1', '1-3', '3-5', '5-7', '7-10', '10-20']
    else:
        if speed_labels is None:
            # Automatically generate labels if not provided
            speed_labels = [f"{speed_bins[i]}-{speed_bins[i+1]}" for i in range(len(speed_bins)-1)]
    
    # Define direction bins
    bin_size = 360 / direction_bins
    if direction_labels is None:
        direction_labels = [f"{int(bin_size * i)}-{int(bin_size * (i+1))}°" for i in range(direction_bins)]
    
    # Assign wind directions to bins
    data['direction_bin'] = pd.cut(data[wd_col], 
                                   bins=np.linspace(0, 360, direction_bins+1, endpoint=True), 
                                   labels=direction_labels, 
                                   include_lowest=True, right=False)
    
    # Assign wind speeds to bins
    data['speed_bin'] = pd.cut(data[ws_col], 
                               bins=speed_bins, 
                               labels=speed_labels, 
                               include_lowest=True, right=False)
    
    # Handle calms
    if calms:
        calms_df = data[data[ws_col] == 0].copy()
        if not calms_df.empty:
            calms_df['direction_bin'] = 'Calm'
            calms_df['speed_bin'] = '0'
            # Append calms to data using pd.concat
            data = pd.concat([data[data[ws_col] > 0], calms_df], ignore_index=True)
            # Update direction labels to include 'Calm'
            if 'Calm' not in direction_labels:
                direction_labels = ['Calm'] + direction_labels
    else:
        # Exclude calms
        data = data[data[ws_col] > 0]
    
    # Recalculate direction bins if 'Calm' is added
    if calms:
        all_direction_bins = ['Calm'] + direction_labels
    else:
        all_direction_bins = direction_labels
    
    # Correct for direction bin bias
    primary_dirs = []
    for angle in [0, 90, 180, 270]:
        label = f"{angle}-{angle + int(bin_size)}°"
        if label in direction_labels:
            primary_dirs.append(label)
    
    # Calculate adjustment factors
    adjustment_factors = {}
    for dir_label in all_direction_bins:
        if dir_label in primary_dirs:
            adjustment_factors[dir_label] = 0.75
        elif dir_label != 'Calm':
            adjustment_factors[dir_label] = 1.125
        else:
            adjustment_factors[dir_label] = 1  # No adjustment for calms
    
    # Apply adjustment factors
    data['adjustment'] = data['direction_bin'].map(adjustment_factors)
    
    # Calculate frequency, including grouping if applicable
    if group_column:
        freq = data.groupby([group_column, 'direction_bin', 'speed_bin']).size().reset_index(name='count')
        # Apply adjustment
        freq['count'] = freq['count'] * freq['direction_bin'].map(adjustment_factors)
        if mode == 'percentage':
            # Calculate total counts per direction bin for percentage
            total_counts = freq.groupby([group_column, 'direction_bin'])['count'].sum().reset_index(name='total')
            freq = freq.merge(total_counts, on=[group_column, 'direction_bin'])
            freq['value'] = (freq['count'] / freq['total']) * 100
        else:
            # For 'count' mode, use raw counts
            freq['value'] = freq['count']
    else:
        freq = data.groupby(['direction_bin', 'speed_bin']).size().reset_index(name='count')
        # Apply adjustment
        freq['count'] = freq['count'] * freq['direction_bin'].map(adjustment_factors)
        if mode == 'percentage':
            # Calculate total counts per direction bin for percentage
            total_counts = freq.groupby('direction_bin')['count'].sum().reset_index(name='total')
            freq = freq.merge(total_counts, on='direction_bin')
            freq['value'] = (freq['count'] / freq['total']) * 100
        else:
            # For 'count' mode, use raw counts
            freq['value'] = freq['count']
    
    # Pivot the data for plotting
    if group_column:
        pivot = freq.pivot_table(index=[group_column, 'direction_bin'], columns='speed_bin', values='value', fill_value=0).reset_index()
        # Sort direction bins numerically based on starting degree
        if group_by == 'quartile':
            # For quartiles, maintain order Q1 to Q4
            pivot[group_column] = pd.Categorical(pivot[group_column], categories=['Q1', 'Q2', 'Q3', 'Q4'], ordered=True)
            pivot = pivot.sort_values([group_column, 'direction_bin'], key=lambda x: x.map({'Calm': -1} if 'Calm' in x.unique() else {}) if group_by == 'quartile' else 0)
        else:
            pivot['direction_order'] = pivot['direction_bin'].apply(lambda x: 0 if x == 'Calm' else int(x.split('-')[0]))
            pivot.sort_values(['year', 'direction_order'], inplace=True)
            pivot.drop('direction_order', axis=1, inplace=True)
    else:
        pivot = freq.pivot(index='direction_bin', columns='speed_bin', values='value').fillna(0)
        # Sort direction bins numerically based on starting degree
        if calms and 'Calm' in pivot.index:
            calm_row = pivot.loc['Calm']
            pivot = pivot.drop('Calm')
            pivot = pivot.sort_index(key=lambda x: pd.to_numeric(x.str.split('-').str[0], errors='coerce'))
            pivot = pd.concat([pivot, calm_row.to_frame().T])
        else:
            pivot = pivot.sort_index(key=lambda x: pd.to_numeric(x.str.split('-').str[0], errors='coerce'))
    
    # Determine the number of groups for subplots
    if group_column:
        groups = pivot[group_column].unique()
        num_groups = len(groups)
    else:
        groups = [None]
        num_groups = 1
    
    # Create the wind rose plot
    if group_column is None:
        # Single Wind Rose Plot
        fig = go.Figure()
        
        # Ensure enough colors are available
        if len(speed_labels) > len(colors):
            colors = colors * (len(speed_labels) // len(colors) + 1)
        colors = colors[:len(speed_labels)]
        
        # Add traces for each speed bin
        for i, speed in enumerate(speed_labels):
            if speed in pivot.columns:
                fig.add_trace(go.Barpolar(
                    r=pivot[speed],
                    theta=pivot.index.str.extract('(\d+)')[0].astype(float) + bin_size/2,
                    width=bin_size,
                    name=f"{speed} m/s",
                    marker_color=colors[i]
                ))
        
        # Optionally, add calms as a separate trace
        # Uncomment the following lines if you wish to include calms in 'none' group
        # if calms and 'Calm' in pivot.index:
        #     fig.add_trace(go.Barpolar(
        #         r=[pivot.loc['Calm'].values[0]],
        #         theta=[0],
        #         width=[360],
        #         name='Calm',
        #         marker_color='gray'
        #     ))
        
        # Update layout
        radial_title = "Percentage (%)" if mode == 'percentage' else "Count"
        fig.update_layout(
            title=title,
            legend_title="Wind Speed (m/s)",
            template='plotly_white',
            width=fig_width,    # Set figure width
            height=fig_height,  # Set figure height
            polar=dict(
                angularaxis=dict(direction='clockwise'),
                radialaxis=dict(
                    title=radial_title,
                    ticksuffix="%" if mode == 'percentage' else "",
                    angle=45,
                    dtick=10 if mode == 'percentage' else None,
                    showline=False
                )
            )
        )
        
        # Create summary DataFrame
        summary_df = freq.copy()
    else:
        # Multiple Wind Rose Plots (Grouped)
        fig = make_subplots(
            rows=ceil(num_groups / subplot_cols), cols=subplot_cols,
            subplot_titles=[],  # We'll add custom annotations for titles
            specs=[[{"type": "polar"} for _ in range(subplot_cols)] for _ in range(ceil(num_groups / subplot_cols))],
            horizontal_spacing=0.02,  # Reduced spacing for more subplot area
            vertical_spacing=0.02
        )
        
        # Ensure enough colors are available
        if len(speed_labels) > len(colors):
            colors = colors * (len(speed_labels) // len(colors) + 1)
        colors = colors[:len(speed_labels)]
        
        for idx, group in enumerate(groups, start=1):
            # Filter data for the current group
            if group_column == 'quartile':
                group_label = group  # e.g., 'Q1', 'Q2', etc.
                group_data = pivot[pivot[group_column] == group]
            else:
                group_label = group  # e.g., year number
                group_data = pivot[pivot[group_column] == group]
            
            # Extract direction bins for the current group
            direction_bins_group = group_data['direction_bin']
            
            # Add traces for each speed bin
            for i, speed in enumerate(speed_labels):
                if speed in group_data.columns:
                    # Only set name and showlegend for the first group
                    name = f"{speed} m/s" if idx == 1 else None
                    showlegend = (idx == 1)
                    
                    fig.add_trace(go.Barpolar(
                        r=group_data[speed],
                        theta=group_data['direction_bin'].str.extract('(\d+)')[0].astype(float) + bin_size/2,
                        width=bin_size,
                        name=name,
                        marker_color=colors[i],
                        showlegend=showlegend
                    ), row=(idx-1)//subplot_cols + 1, col=(idx-1)%subplot_cols + 1)
            
            # Optionally, add calms as a separate trace
            # Uncomment the following lines if you wish to include calms in 'quartile' group
            # if calms and 'Calm' in group_data['direction_bin'].values:
            #     calm_value = group_data[group_data['direction_bin'] == 'Calm']['value'].sum()
            #     fig.add_trace(go.Barpolar(
            #         r=[calm_value],
            #         theta=[0],
            #         width=[360],
            #         name='Calm' if idx == 1 else None,
            #         marker_color='gray',
            #         showlegend=(idx == 1)
            #     ), row=(idx-1)//subplot_cols + 1, col=(idx-1)%subplot_cols + 1)
        
        # Update layout
        radial_title = "Percentage (%)" if mode == 'percentage' else "Count"
        fig.update_layout(
            title=title,
            legend_title="Wind Speed (m/s)",
            template='plotly_white',
            width=fig_width,    # Set figure width
            height=fig_height   # Set figure height
        )
        
        # Calculate subplot titles' positions and add them as annotations
        for idx, group in enumerate(groups, start=1):
            row = (idx - 1) // subplot_cols + 1
            col = (idx - 1) % subplot_cols + 1
            
            # Calculate the subplot's domain
            subplot_width = 1 / subplot_cols
            subplot_height = 1 / ceil(num_groups / subplot_cols)
            x0 = (col - 1) * subplot_width
            x1 = col * subplot_width
            y0 = 1 - row * subplot_height
            y1 = 1 - (row - 1) * subplot_height
            
            # Position the title at the left side of the subplot
            x_title = x0 + 0.02 * subplot_width  # 2% padding from the left
            y_title = y1 - 0.02 * subplot_height  # 2% padding from the top
            
            # Format the title based on grouping
            if group_by == 'year':
                annotation_text = f"<b>{group}</b>"
            elif group_by == 'quartile':
                annotation_text = f"<b>{group}</b>"
            else:
                annotation_text = f"<b>{group}</b>"
            
            fig.add_annotation(
                text=annotation_text,
                x=x_title,
                y=y_title,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(size=14, color='#666666'),
                xanchor='left',
                yanchor='top'
            )
        
        # Update each subplot's polar layout
        for i in range(1, num_groups + 1):
            fig.update_layout(**{
                f'polar{i}': dict(
                    angularaxis=dict(direction='clockwise'),
                    radialaxis=dict(
                        title=radial_title,
                        ticksuffix="%" if mode == 'percentage' else "",
                        angle=45,
                        dtick=10 if mode == 'percentage' else None,
                        showline=False
                    )
                )
            })
        
        # Adjust the layout for better spacing
        fig.update_layout(
            margin=dict(l=50, r=50, t=100, b=50),  # Increase top margin for the main title
        )
        
        # Create summary DataFrame
        summary_df = freq.copy()
    
    return fig, summary_df
