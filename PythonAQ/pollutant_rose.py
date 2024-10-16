import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from math import ceil
from wind_rose import wind_rose  # Assuming wind_rose is imported as specified

def pollutant_rose(df, pollutant, wd_col='wd', condition_col=None,
                   direction_bins=16, pollutant_bins=None, 
                   calms=True, pollutant_labels=None, 
                   direction_labels=None, title='Pollutant Rose',
                   colors=px.colors.sequential.Jet,
                   mode='count',
                   statistic=None,
                   group_by='none',  # 'none', 'year', 'quartile'
                   subplot_cols=3,    # Number of subplot columns
                   fig_width=1200,    # Figure width in pixels
                   fig_height=800     # Figure height in pixels
                  ):
    """
    Creates a pollutant rose plot using Plotly, displaying pollutant concentrations by wind direction,
    and optionally conditioning on another variable or grouping by year/quartiles.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing wind and pollutant data.
    - pollutant (str): Column name for pollutant concentration to plot.
    - wd_col (str): Column name for wind direction.
    - condition_col (str or None): Column name to condition the pollutant rose on (e.g., another pollutant).
    - direction_bins (int): Number of wind direction bins (e.g., 16 for 22.5° bins).
    - pollutant_bins (list or None): List defining pollutant concentration categories. If None, defaults are used based on quantiles.
    - calms (bool): Whether to include calms (e.g., pollutant concentration = 0) in the plot.
    - pollutant_labels (list or None): Labels for pollutant concentration categories. If None, defaults are used.
    - direction_labels (list or None): Labels for wind direction categories. If None, defaults are used.
    - title (str): Title of the plot.
    - colors (list): Plotly color scale for pollutant categories, default is px.colors.sequential.Viridis.
    - mode (str): 'count' to display raw counts/sums, 'percentage' to display as percentages.
    - statistic (str or None): 'prop.mean' to display proportion contribution to the mean.
    - group_by (str): 'none' to plot all data together, 'year' to plot separate pollutant roses per year,
                      or 'quartile' to plot separate pollutant roses per quartile of the pollutant column.
    - subplot_cols (int): Number of subplot columns when grouping.
    - fig_width (int): Width of the entire figure in pixels.
    - fig_height (int): Height of the entire figure in pixels.
    
    Returns:
    - fig (plotly.graph_objects.Figure): The resulting pollutant rose figure.
    - summary_df (pd.DataFrame): DataFrame containing summary statistics.
    """
    
    # Validate 'mode' parameter
    if mode not in ['percentage', 'count']:
        raise ValueError("Invalid mode. Choose either 'percentage' or 'count'.")
    
    # Validate 'statistic' parameter
    if statistic not in [None, 'prop.mean']:
        raise ValueError("Invalid statistic. Choose either None or 'prop.mean'.")
    
    # Validate 'group_by' parameter
    if group_by not in ['none', 'year', 'quartile']:
        raise ValueError("Invalid group_by option. Choose 'none', 'year', or 'quartile'.")
    
    # If grouping by year, ensure date_col is provided and extract 'year'
    if group_by == 'year':
        if 'date_time' not in df.columns:
            raise ValueError("DataFrame must contain 'date_time' column for year grouping.")
        # Ensure 'date_time' is in datetime format
        if not np.issubdtype(df['date_time'].dtype, np.datetime64):
            df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
        # Drop rows with invalid dates
        df = df.dropna(subset=['date_time'])
        # Extract year
        df['year'] = df['date_time'].dt.year
    
    # If grouping by quartile, ensure pollutant is numeric and calculate quartiles
    if group_by == 'quartile':
        if not np.issubdtype(df[pollutant].dtype, np.number):
            raise ValueError("pollutant_col must be a numeric column for quartile grouping.")
        # Drop rows with missing pollutant values
        df = df.dropna(subset=[pollutant])
        # Calculate quartiles
        df['quartile'] = pd.qcut(df[pollutant], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # If conditioning on another variable, ensure it exists
    if condition_col:
        if condition_col not in df.columns:
            raise ValueError(f"Condition column '{condition_col}' does not exist in the DataFrame.")
        # If grouping by condition_col, ensure it's handled appropriately
        # Here, we'll treat condition_col similar to 'year' or 'quartile'
    
    # Determine the grouping column
    if condition_col:
        group_column = condition_col
    elif group_by == 'year':
        group_column = 'year'
    elif group_by == 'quartile':
        group_column = 'quartile'
    else:
        group_column = None  # 'none' grouping
    
    # Copy relevant columns, including grouping column if applicable
    if group_column:
        data = df[[pollutant, wd_col, group_column]].copy()
    else:
        data = df[[pollutant, wd_col]].copy()
    
    # Handle missing or invalid data
    data = data.dropna(subset=[pollutant, wd_col])
    data = data[(data[pollutant] >= 0) & (data[wd_col] >= 0) & (data[wd_col] <= 360)]
    
    # Define default pollutant bins if not provided
    if pollutant_bins is None:
        # Use quantiles to define sensible bins
        pollutant_bins = list(pd.qcut(data[pollutant], 7, duplicates='drop').cat.categories)
        pollutant_labels = [f"{int(interval.left)} to {int(interval.right)}" for interval in pollutant_bins]
    else:
        if pollutant_labels is None:
            pollutant_labels = [f"{pollutant_bins[i]} to {pollutant_bins[i+1]}" for i in range(len(pollutant_bins)-1)]
    
    # Define direction bins
    bin_size = 360 / direction_bins
    if direction_labels is None:
        direction_labels = [f"{int(bin_size * i)}-{int(bin_size * (i+1))}°" for i in range(direction_bins)]
    
    # Assign wind directions to bins
    data['direction_bin'] = pd.cut(data[wd_col], 
                                   bins=np.linspace(0, 360, direction_bins+1, endpoint=True), 
                                   labels=direction_labels, 
                                   include_lowest=True, right=False)
    
    # Assign pollutant concentrations to bins
    data['pollutant_bin'] = pd.cut(data[pollutant], 
                                    bins=pollutant_bins, 
                                    labels=pollutant_labels, 
                                    include_lowest=True, right=False)
    
    # Handle calms (pollutant concentration = 0)
    if calms:
        calms_df = data[data[pollutant] == 0].copy()
        if not calms_df.empty:
            calms_df['direction_bin'] = 'Calm'
            calms_df['pollutant_bin'] = '0'
            # Append calms to data using pd.concat
            data = pd.concat([data[data[pollutant] > 0], calms_df], ignore_index=True)
            # Update direction labels to include 'Calm' if not already present
            if 'Calm' not in direction_labels:
                direction_labels = ['Calm'] + direction_labels
    else:
        # Exclude calms
        data = data[data[pollutant] > 0]
    
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
    
    # Calculate frequency based on grouping and statistic
    if group_column:
        if statistic == 'prop.mean':
            # Calculate overall mean pollutant
            overall_mean = data[pollutant].mean()
            # Group by and sum pollutant
            freq = data.groupby([group_column, 'direction_bin', 'pollutant_bin']).agg(
                pollutant_sum=(pollutant, 'sum'),
                count=(pollutant, 'count')
            ).reset_index()
            # Calculate expected sum based on mean
            freq['expected_sum'] = overall_mean * freq['count']
            # Calculate proportion of mean
            freq['value'] = (freq['pollutant_sum'] / freq['expected_sum']) * 100
        else:
            # Default statistics: 'count' or 'percentage'
            freq = data.groupby([group_column, 'direction_bin', 'pollutant_bin']).agg(
                count=(pollutant, 'sum')  # Sum pollutant as count
            ).reset_index()
            if mode == 'percentage':
                # Calculate total counts per group and direction bin for percentage
                total_counts = freq.groupby([group_column, 'direction_bin'])['count'].sum().reset_index(name='total')
                freq = freq.merge(total_counts, on=['group_column', 'direction_bin'])
                freq['value'] = (freq['count'] / freq['total']) * 100
            else:
                # For 'count' mode, use sum of pollutant
                freq['value'] = freq['count']
    else:
        if statistic == 'prop.mean':
            # Calculate overall mean pollutant
            overall_mean = data[pollutant].mean()
            # Group by and sum pollutant
            freq = data.groupby(['direction_bin', 'pollutant_bin']).agg(
                pollutant_sum=(pollutant, 'sum'),
                count=(pollutant, 'count')
            ).reset_index()
            # Calculate expected sum based on mean
            freq['expected_sum'] = overall_mean * freq['count']
            # Calculate proportion of mean
            freq['value'] = (freq['pollutant_sum'] / freq['expected_sum']) * 100
        else:
            # Default statistics: 'count' or 'percentage'
            freq = data.groupby(['direction_bin', 'pollutant_bin']).agg(
                count=(pollutant, 'sum')  # Sum pollutant as count
            ).reset_index()
            if mode == 'percentage':
                # Calculate total counts per direction bin for percentage
                total_counts = freq.groupby('direction_bin')['count'].sum().reset_index(name='total')
                freq = freq.merge(total_counts, on='direction_bin')
                freq['value'] = (freq['count'] / freq['total']) * 100
            else:
                # For 'count' mode, use sum of pollutant
                freq['value'] = freq['count']
    
    # Pivot the data for plotting
    if group_column:
        pivot = freq.pivot_table(index=[group_column, 'direction_bin'], 
                                 columns='pollutant_bin', 
                                 values='value', 
                                 fill_value=0).reset_index()
        # Sort direction bins numerically based on starting degree
        pivot['direction_order'] = pivot['direction_bin'].apply(
            lambda x: 0 if x == 'Calm' else int(x.split('-')[0])
        )
        pivot.sort_values([group_column, 'direction_order'], inplace=True)
        pivot.drop('direction_order', axis=1, inplace=True)
    else:
        pivot = freq.pivot(index='direction_bin', columns='pollutant_bin', values='value').fillna(0)
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
    
    # Create the pollutant rose plot
    if group_column is None:
        # Single Pollutant Rose Plot
        fig = go.Figure()
        
        # Ensure enough colors are available
        if len(pollutant_labels) > len(colors):
            colors = colors * (len(pollutant_labels) // len(colors) + 1)
        colors = colors[:len(pollutant_labels)]
        
        # Add traces for each pollutant bin
        for i, bin_label in enumerate(pollutant_labels):
            if bin_label in pivot.columns:
                fig.add_trace(go.Barpolar(
                    r=pivot[bin_label],
                    theta=pivot.index.str.extract('(\d+)')[0].astype(float) + bin_size/2,
                    width=bin_size,
                    name=f"{bin_label}",
                    marker_color=colors[i]
                ))
        
        # Optionally, add calms as a separate trace
        # Uncomment the following lines if you wish to include calms in 'none' group
        # if calms and '0' in pivot.columns:
        #     fig.add_trace(go.Barpolar(
        #         r=pivot['0'],
        #         theta=[0],
        #         width=[360],
        #         name='Calm',
        #         marker_color='gray'
        #     ))
        
        # Update layout
        radial_title = "Percentage (%)" if mode == 'percentage' else "Sum" if statistic == 'prop.mean' else "Sum"
        fig.update_layout(
            title=title,
            legend_title=f"{pollutant} Concentration",
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
        # Multiple Pollutant Rose Plots (Grouped)
        fig = make_subplots(
            rows=ceil(num_groups / subplot_cols), cols=subplot_cols,
            subplot_titles=[],  # We'll add custom annotations for titles
            specs=[[{"type": "polar"} for _ in range(subplot_cols)] for _ in range(ceil(num_groups / subplot_cols))],
            horizontal_spacing=0.02,  # Reduced spacing for more subplot area
            vertical_spacing=0.02
        )
        
        # Ensure enough colors are available
        if len(pollutant_labels) > len(colors):
            colors = colors * (len(pollutant_labels) // len(colors) + 1)
        colors = colors[:len(pollutant_labels)]
        
        for idx, group in enumerate(groups, start=1):
            if group_column == 'quartile':
                group_label = group  # e.g., 'Q1', 'Q2', etc.
                group_data = pivot[pivot[group_column] == group]
            else:
                group_label = group  # e.g., year number
                group_data = pivot[pivot[group_column] == group]
            
            # Add traces for each pollutant bin
            for i, bin_label in enumerate(pollutant_labels):
                if bin_label in group_data.columns:
                    # Only set name and showlegend for the first group
                    name = f"{bin_label}" if idx == 1 else None
                    showlegend = (idx == 1)
                    
                    fig.add_trace(go.Barpolar(
                        r=group_data[bin_label],
                        theta=group_data['direction_bin'].str.extract('(\d+)')[0].astype(float) + bin_size/2,
                        width=bin_size,
                        name=name,
                        marker_color=colors[i],
                        showlegend=showlegend
                    ), row=(idx-1)//subplot_cols + 1, col=(idx-1)%subplot_cols + 1)
            
            # Optionally, add calms as a separate trace
            # Uncomment the following lines if you wish to include calms in 'grouped' condition
            # if calms and '0' in group_data.columns:
            #     fig.add_trace(go.Barpolar(
            #         r=group_data['0'],
            #         theta=[0],
            #         width=[360],
            #         name='Calm' if idx == 1 else None,
            #         marker_color='gray',
            #         showlegend=(idx == 1)
            #     ), row=(idx-1)//subplot_cols + 1, col=(idx-1)%subplot_cols + 1)
        
        # Update layout
        radial_title = "Percentage (%)" if mode == 'percentage' else "Sum" if statistic == 'prop.mean' else "Sum"
        fig.update_layout(
            title=title,
            legend_title=f"{pollutant} Concentration",
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
            elif condition_col:
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
