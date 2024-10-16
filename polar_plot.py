import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
from pygam import LinearGAM, s, te

def polar_plot(
    df,
    ws_col='ws',
    wd_col='wd',
    conc_col='NO2',
    ws_bins=60,
    wd_bins=72,
    color_palette='Spectral_r',
    title='Polar Plot of Concentration',
    vmin=None,
    vmax=None,
    fig_width=800,
    fig_height=800,
    min_count=3,  # Minimum number of observations per bin
    n_splines=10,
    uncertainty=None
):
    """
    Creates a bivariate polar tile plot of concentrations showing how concentrations
    vary with wind speed and wind direction using Plotly shapes.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing wind data and concentrations.
    - ws_col (str): Column name for wind speed.
    - wd_col (str): Column name for wind direction.
    - conc_col (str): Column name for concentration.
    - ws_bins (int): Number of bins for wind speed (default: 60).
    - wd_bins (int): Number of bins for wind direction (default: 72).
    - color_palette (str or list): Plotly color scale (default: 'Spectral_r').
    - title (str): Title of the plot (default: 'Polar Plot of Concentration').
    - vmin, vmax (float): Minimum and maximum values for color scale.
    - fig_width (int): Width of the figure in pixels (default: 800).
    - fig_height (int): Height of the figure in pixels (default: 800).
    - min_count (int): Minimum number of observations per bin to include in the analysis.
    - n_splines (int): Number of splines for the tensor GAM.
    - uncertainty (float or None): Float value less than 1 for handling the level of confidence
    
    Returns:
    - fig (plotly.graph_objects.Figure): The resulting polar tile plot.
    """
    # Validate input columns
    for col in [ws_col, wd_col, conc_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    
    # Drop rows with missing data
    data = df[[ws_col, wd_col, conc_col]].dropna()
    
    # Ensure wind directions are within [0, 360)
    data[wd_col] = data[wd_col] % 360
    
    # Bin the data into wind speed and wind direction bins
    ws_min, ws_max = data[ws_col].min(), data[ws_col].max()
    ws_bins_array = np.linspace(ws_min, ws_max, ws_bins + 1)
    wd_bins_array = np.linspace(0, 360, wd_bins + 1)
    
    # Assign bins to data
    data['ws_bin'] = pd.cut(data[ws_col], bins=ws_bins_array, labels=False, include_lowest=True)
    data['wd_bin'] = pd.cut(data[wd_col], bins=wd_bins_array, labels=False, include_lowest=True)
    
    # Group data by bins and compute statistics
    grouped = data.groupby(['ws_bin', 'wd_bin'])
    binned_data = grouped.agg(
        ws_mean=(ws_col, 'mean'),
        wd_mean=(wd_col, 'mean'),
        conc_mean=(conc_col, 'mean'),
        conc_std=(conc_col, 'std'),
        count=(conc_col, 'count')
    ).reset_index()
    
    # Remove bins with too few data points
    binned_data = binned_data[binned_data['count'] >= min_count]
    
    # Prepare data for GAM
    # Convert wind speed and direction into u and v components
    wd_rad = np.deg2rad(binned_data['wd_mean'])
    u = binned_data['ws_mean'] * np.sin(wd_rad)
    v = binned_data['ws_mean'] * np.cos(wd_rad)
    X = np.column_stack([u, v])
    y = binned_data['conc_mean']
    
    # Fit GAM using the binned data with tensor spline
    gam = LinearGAM(te(0, 1, n_splines=(n_splines, n_splines)))
    gam.gridsearch(X, y)
    
    # Create grid for prediction (matching bin centers)
    ws_centers = (ws_bins_array[:-1] + ws_bins_array[1:]) / 2
    wd_centers = (wd_bins_array[:-1] + wd_bins_array[1:]) / 2
    ws_grid, wd_grid = np.meshgrid(ws_centers, wd_centers)
    wd_grid_rad = np.deg2rad(wd_grid)
    u_pred = ws_grid * np.sin(wd_grid_rad)
    v_pred = ws_grid * np.cos(wd_grid_rad)
    X_pred = np.column_stack([u_pred.ravel(), v_pred.ravel()])
    
    # Predict concentrations over the grid
    Z_pred = gam.predict(X_pred)
    Z_pred = Z_pred.reshape(ws_grid.shape)
    
    # Get prediction intervals
    intervals = gam.prediction_intervals(X_pred, width=0.95)
    Z_pred_lower = intervals[:, 0].reshape(ws_grid.shape)
    Z_pred_upper = intervals[:, 1].reshape(ws_grid.shape)
    
    # Compute prediction error (half-width of the prediction interval)
    Z_error = (Z_pred_upper - Z_pred_lower) / 2
    
    mean_pollutant = data[conc_col].mean()
    sd_pollutant = data[conc_col].std()
    max_allowed = mean_pollutant + (1.2 * sd_pollutant)
    
    # Implement quality control: Discard predictions where error > predicted concentration or predicted concentration > max_allowed
    Z_masked = np.where(
        (Z_error > np.abs(Z_pred)) | (Z_pred > max_allowed),
        np.nan,
        Z_pred
    )
    
    # Set color scale limits
    if vmin is None:
        if np.nanmin(Z_masked) <= 0:
            vmin = 0
        else:
            vmin = np.nanmin(Z_masked)
    if vmax is None:
        vmax = np.nanmax(Z_masked)
    
    # Normalize concentrations for color mapping
    norm_conc = (Z_masked - vmin) / (vmax - vmin)
    norm_conc = np.clip(norm_conc, 0, 1)
    
    # Get colorscale
    if isinstance(color_palette, str):
        colorscale = pcolors.get_colorscale(color_palette)
    else:
        colorscale = color_palette  # Assume it's a list of colors
    
    from plotly.colors import sample_colorscale
    
    # Initialize shapes list
    shapes = []
    
    # Loop over bins and create tiles
    num_ws_bins = len(ws_centers)
    num_wd_bins = len(wd_centers)
    for i in range(num_ws_bins):
        for j in range(num_wd_bins):
            conc_value = Z_masked[j, i]
            if np.isnan(conc_value) or conc_value <= 0:
                continue  # Skip tiles with NaN values
            
            # Get bin edges
            ws0 = ws_bins_array[i]
            ws1 = ws_bins_array[i + 1]
            wd0 = wd_bins_array[j]
            wd1 = wd_bins_array[j + 1]
            
            # Adjust angles to have north at the top and increase clockwise
            theta0_adj = (-wd0 + 90) % 360
            theta1_adj = (-wd1 + 90) % 360
            
            # Convert to radians
            theta0_rad = np.deg2rad(theta0_adj)
            theta1_rad = np.deg2rad(theta1_adj)
            
            # Compute corners of the tile
            r0 = ws0
            r1 = ws1
            x0 = r0 * np.cos(theta0_rad)
            y0 = r0 * np.sin(theta0_rad)
            x1 = r0 * np.cos(theta1_rad)
            y1 = r0 * np.sin(theta1_rad)
            x2 = r1 * np.cos(theta1_rad)
            y2 = r1 * np.sin(theta1_rad)
            x3 = r1 * np.cos(theta0_rad)
            y3 = r1 * np.sin(theta0_rad)
            
            # Get fill color
            norm_value = norm_conc[j, i]
            fillcolor = sample_colorscale(colorscale, [norm_value])[0]
            
            # Create path for the shape
            path = f'M {x0},{y0} L {x1},{y1} L {x2},{y2} L {x3},{y3} Z'
            
            # Create shape dictionary
            shape = dict(
                type='path',
                path=path,
                fillcolor=fillcolor,
                line=dict(width=0.5, color=fillcolor),
                xref='x',
                yref='y'
            )
            shapes.append(shape)
    
    # Create figure
    fig = go.Figure()
    
    # Update layout
    fig.update_layout(
        title=title,
        width=fig_width,
        height=fig_height,
        shapes=shapes,
        xaxis=dict(
            visible=False,
            scaleanchor='y',
            scaleratio=1,
            range=[-ws_max * 1.1, ws_max * 1.1],
        ),
        yaxis=dict(
            visible=False,
            range=[-ws_max * 1.1, ws_max * 1.1],
        ),
        template='plotly_white'
    )
    
    # Add colorbar using a dummy scatter trace
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale=color_palette,
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(
                title=conc_col,
                thickness=20,
                len=0.75,
                y=0.5,
            ),
            color=[],
            showscale=True
        ),
        showlegend=False
    ))
    
    # Adjust compass annotations to only N, E, S, W with black arrows
    compass_directions = ['N', 'E', 'S', 'W']
    compass_angles = np.array([0, 90, 180, 270])
    compass_angles_adj = (-compass_angles + 90) % 360
    compass_angles_rad = np.deg2rad(compass_angles_adj)

    # Define anchors based on direction to position text correctly
    xanchors = {'N': 'center', 'E': 'left', 'S': 'center', 'W': 'right'}
    yanchors = {'N': 'bottom', 'E': 'middle', 'S': 'top', 'W': 'middle'}

    for direction, angle in zip(compass_directions, compass_angles_rad):
        x_edge = (ws_max + 0.1 * ws_max) * np.cos(angle)
        y_edge = (ws_max + 0.1 * ws_max) * np.sin(angle)
        
        # Add arrow pointing from edge to center
        fig.add_annotation(
            x=x_edge,
            y=y_edge,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='black',
            text='',  # No text for arrow annotation
        )
        
        # Add text at the edge
        fig.add_annotation(
            x=x_edge,
            y=y_edge,
            xref='x',
            yref='y',
            text=direction,
            font=dict(size=12, color='black'),
            showarrow=False,
            xanchor=xanchors[direction],
            yanchor=yanchors[direction],
        )
    
    # Create radial grid lines in steps of 5 up to next step after max wind speed
    radial_step = 5
    ws_max_rounded = ((ws_max // radial_step) + 1) * radial_step
    radial_grid = np.arange(0, ws_max_rounded + radial_step, radial_step)
    for r in radial_grid:
        if r == 0:
            continue  # Skip zero radius circle
        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=-r,
            y0=-r,
            x1=r,
            y1=r,
            line=dict(color="lightgray", width=1)
        )
        # Add annotations for radial grid lines
        fig.add_annotation(
            x=r,
            y=0,
            text=f"{r}",
            showarrow=False,
            font=dict(size=10),
            xref="x",
            yref="y",
            xanchor='left',
            yanchor='bottom'
        )
    
    # Add angular grid lines (lines from center)
    angular_grid = np.array([0, 90, 180, 270])  # Only cardinal points
    angular_grid_adj = (-angular_grid + 90) % 360
    angular_grid_rad = np.deg2rad(angular_grid_adj)
    for theta in angular_grid_rad:
        x0, y0 = 0, 0
        x1 = ws_max_rounded * 1.05 * np.cos(theta)
        y1 = ws_max_rounded * 1.05 * np.sin(theta)
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color="lightgray", width=1)
        )
    
    return fig
