import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
from plotly.subplots import make_subplots

def polar_frequency_plot(
    df, 
    ws_col='ws', 
    wd_col='wd', 
    date_col='date_time',          # Column name for datetime data
    direction_bins=36, 
    speed_bins=36, 
    color_palette='Spectral_r',
    min_radius=3,         # Offset zero wind speed from the center
    separate_by_year=True, # Enable separation by year
    fig_width=800,   
    fig_height=800,    
    title='Polar Frequency Plot'
):
    """
    Creates a polar frequency plot showing the distribution of wind speeds and directions,
    with options to separate data by year.

    Parameters:
    - df (pd.DataFrame): DataFrame containing wind data.
    - ws_col (str): Column name for wind speed.
    - wd_col (str): Column name for wind direction.
    - date_col (str): Column name for datetime data (required if separate_by_year is True).
    - direction_bins (int): Number of bins for wind direction (default: 36).
    - speed_bins (int): Number of bins for wind speed (default: 36).
    - color_palette (str or list): Plotly color scale for encoding frequency (default: 'Spectral_r').
    - min_radius (float): Minimum radius to offset zero wind speed from the center (default: 0.1).
    - separate_by_year (bool): Whether to separate the data by year (default: False).
    - fig_width (int): Width of the figure in pixels (default: 800).
    - fig_height (int): Height of the figure in pixels (default: 800).
    - title (str): Title of the plot (default: 'Polar Frequency Plot').

    Returns:
    - fig (plotly.graph_objects.Figure): The resulting polar frequency plot.
    """

    # Validate input columns
    if ws_col not in df.columns:
        raise ValueError(f"Wind speed column '{ws_col}' not found in DataFrame.")
    if wd_col not in df.columns:
        raise ValueError(f"Wind direction column '{wd_col}' not found in DataFrame.")
    if separate_by_year:
        if date_col is None:
            raise ValueError("Please provide 'date_col' when 'separate_by_year' is True.")
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame.")

    # Preprocess data
    data = df[[ws_col, wd_col]].copy()
    if separate_by_year:
        data[date_col] = pd.to_datetime(df[date_col])
        data['Year'] = data[date_col].dt.year
        unique_years = sorted(data['Year'].unique())
    else:
        data['Year'] = None  # Placeholder for consistent processing

    # Determine number of subplots
    if separate_by_year:
        num_years = len(unique_years)
        # Adjust figure size based on number of years
        cols = min(3, num_years)
        rows = (num_years + cols - 1) // cols  # Ceiling division
        fig_width_total = fig_width * cols
        fig_height_total = fig_height * rows
        # Create subplot figure
        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{'type': 'scatter'}]*cols]*rows,
            subplot_titles=[str(year) for year in unique_years],
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )
    else:
        fig = make_subplots()
        fig_width_total = fig_width
        fig_height_total = fig_height
        cols = 1
        rows = 1

    # Initialize shapes and annotations lists
    shapes = []
    annotations = []

    # Function to create polar plot for a given dataset
    def create_polar_plot(data_subset, row=1, col=1):
        # Proceed with the same plotting code as before, adjusted for subplot
        # Ensure wind directions are within [0, 360)
        data_subset = data_subset[(data_subset[wd_col] >= 0) & (data_subset[wd_col] < 360)]

        # Define wind speed and direction bins
        speed_min, speed_max = data_subset[ws_col].min(), data_subset[ws_col].max()
        if speed_min == speed_max:
            speed_min = 0  # Avoid zero division if all speeds are the same
        speed_bins_array = np.linspace(speed_min, speed_max, speed_bins + 1)
        direction_bins_array = np.linspace(0, 360, direction_bins + 1)

        # Compute 2D histogram
        counts, _, _ = np.histogram2d(
            data_subset[ws_col],
            data_subset[wd_col],
            bins=[speed_bins_array, direction_bins_array]
        )

        # Normalize counts to [0,1] for color mapping
        counts_min = counts.min()
        counts_max = counts.max()
        if counts_max == counts_min:
            counts_max += 1  # Avoid division by zero
        norm_counts = (counts - counts_min) / (counts_max - counts_min)

        # Get colorscale
        if isinstance(color_palette, str):
            colorscale = pcolors.get_colorscale(color_palette)
        else:
            colorscale = color_palette  # Assume it's a list of colors

        from plotly.colors import sample_colorscale

        # Compute axis names
        axis_suffix = '' if (row == 1 and col == 1) else str((row - 1) * cols + col)
        xref = f'x{axis_suffix}'
        yref = f'y{axis_suffix}'

        # Loop over bins and create shapes
        num_speed_bins = len(speed_bins_array) - 1
        num_direction_bins = len(direction_bins_array) - 1

        for i in range(num_speed_bins):
            r0 = speed_bins_array[i]
            r1 = speed_bins_array[i+1]
            for j in range(num_direction_bins):
                theta0 = direction_bins_array[j]
                theta1 = direction_bins_array[j+1]
                count = counts[i, j]
                norm_count = norm_counts[i, j]
                # Adjust angles to rotate plot so that north is at the top and angles increase clockwise
                theta0_adj = (-theta0 + 90) % 360
                theta1_adj = (-theta1 + 90) % 360
                # Convert to radians
                theta0_rad = np.deg2rad(theta0_adj)
                theta1_rad = np.deg2rad(theta1_adj)
                # Adjust radial positions by min_radius
                r0_adj = r0 + min_radius
                r1_adj = r1 + min_radius
                # Compute corners
                x0 = r0_adj * np.cos(theta0_rad)
                y0 = r0_adj * np.sin(theta0_rad)
                x1 = r0_adj * np.cos(theta1_rad)
                y1 = r0_adj * np.sin(theta1_rad)
                x2 = r1_adj * np.cos(theta1_rad)
                y2 = r1_adj * np.sin(theta1_rad)
                x3 = r1_adj * np.cos(theta0_rad)
                y3 = r1_adj * np.sin(theta0_rad)
                # Create path
                path = f'M {x0},{y0} L {x1},{y1} L {x2},{y2} L {x3},{y3} Z'

                # Determine fill color
                if count == 0:
                    # Set to transparent if count is zero
                    fillcolor = 'rgba(0,0,0,0)'
                else:
                    # Get fill color from colorscale
                    fillcolor = sample_colorscale(colorscale, [norm_count])[0]

                # Create shape
                shape = dict(
                    type='path',
                    path=path,
                    fillcolor=fillcolor,
                    line=dict(width=0),
                    xref=xref,
                    yref=yref,
                )
                # Add shape to shapes list
                shapes.append(shape)

        # Configure axes
        fig.update_xaxes(
            dict(
                visible=False,
                scaleanchor=yref,
                scaleratio=1,
                range=[-(speed_max + min_radius) * 1.1, (speed_max + min_radius) * 1.1],
            ),
            row=row,
            col=col
        )
        fig.update_yaxes(
            dict(
                visible=False,
                range=[-(speed_max + min_radius) * 1.1, (speed_max + min_radius) * 1.1],
            ),
            row=row,
            col=col
        )

        # Add annotations for compass directions
        compass_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        compass_angles = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        # Adjust compass angles to match the rotation and direction
        compass_angles_adj = (-compass_angles + 90) % 360
        compass_angles_rad = np.deg2rad(compass_angles_adj)
        for direction, angle in zip(compass_directions, compass_angles_rad):
            x = (speed_max + min_radius + 0.1 * speed_max) * np.cos(angle)
            y = (speed_max + min_radius + 0.1 * speed_max) * np.sin(angle)
            annotation = dict(
                x=x,
                y=y,
                text=direction,
                showarrow=False,
                font=dict(size=12),
                xref=xref,
                yref=yref,
            )
            annotations.append(annotation)

        # Add radial grid lines (circles)
        radial_grid = np.linspace(speed_min + min_radius, speed_max + min_radius, 5)
        for r in radial_grid:
            shape = dict(
                type="circle",
                xref=xref,
                yref=yref,
                x0=-r,
                y0=-r,
                x1=r,
                y1=r,
                line=dict(color="lightgray", width=1),
            )
            shapes.append(shape)

            # Add annotations for radial grid lines
            annotation = dict(
                x=r * np.cos(np.deg2rad(15)),  # Slightly offset angle for label position
                y=r * np.sin(np.deg2rad(15)),
                text=f"{r - min_radius:.1f}",
                showarrow=False,
                font=dict(size=10),
                xref=xref,
                yref=yref,
                xanchor='left',
                yanchor='bottom',
            )
            annotations.append(annotation)

        # Add angular grid lines (lines from center)
        angular_grid = np.arange(0, 360, 30)
        # Adjust angular grid lines to match the rotation and direction
        angular_grid_adj = (-angular_grid + 90) % 360
        angular_grid_rad = np.deg2rad(angular_grid_adj)
        for theta in angular_grid_rad:
            x0, y0 = (min_radius * np.cos(theta), min_radius * np.sin(theta))
            x1 = (speed_max + min_radius) * np.cos(theta)
            y1 = (speed_max + min_radius) * np.sin(theta)
            shape = dict(
                type="line",
                xref=xref,
                yref=yref,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color="lightgray", width=1),
            )
            shapes.append(shape)

        # Add an empty scatter trace to ensure axes are created
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode='markers',
                marker=dict(size=0, color='rgba(0,0,0,0)'),
            ),
            row=row,
            col=col
        )

    if separate_by_year:
        # Iterate over each year and create a subplot
        for idx, year in enumerate(unique_years):
            row_idx = (idx // cols) + 1
            col_idx = (idx % cols) + 1
            data_subset = data[data['Year'] == year]
            create_polar_plot(data_subset, row=row_idx, col=col_idx)
    else:
        # If not separating by year, plot the entire dataset
        create_polar_plot(data)
        fig.update_layout(title=title)

    # Adjust layout
    fig.update_layout(
        width=fig_width_total,
        height=fig_height_total,
        title_text=title,
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
    )

    # Add colorbar
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale=color_palette,
            cmin=0,
            #cmax=counts_max,
            colorbar=dict(
                title='Frequency',
                thickness=20,
                len=0.75,
                y=0.5,
            ),
            color=[],
            showscale=True
        ),
        showlegend=False
    ))

    return fig
