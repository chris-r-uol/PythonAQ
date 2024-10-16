import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

def polar_cluster(
    df,
    ws_col='ws',
    wd_col='wd',
    feature_cols=['PM10'],
    n_clusters=8,
    ws_bins=30,
    wd_bins=36,
    color_palette=px.colors.qualitative.Prism,
    title='Polar Cluster Plot',
    fig_width=800,
    fig_height=800,
    min_count=1,  # Minimum number of observations per bin
    seed=42,
    include_ws_wd_in_clustering=True  # New parameter
):
    """
    Creates a polar tile plot where each tile is colored based on cluster assignments
    from K-means clustering on specified features.

    Parameters:
    - df (pd.DataFrame): DataFrame containing wind data and features for clustering.
    - ws_col (str): Column name for wind speed.
    - wd_col (str): Column name for wind direction.
    - feature_cols (list): List of column names to be used for clustering.
    - n_clusters (int): Number of clusters for K-means (default: 5).
    - ws_bins (int): Number of bins for wind speed (default: 60).
    - wd_bins (int): Number of bins for wind direction (default: 72).
    - color_palette (str or list): Categorical color palette.
    - title (str): Title of the plot (default: 'Polar Cluster Plot').
    - fig_width (int): Width of the figure in pixels (default: 800).
    - fig_height (int): Height of the figure in pixels (default: 800).
    - min_count (int): Minimum number of observations per bin to include in the analysis.
    - seed (int): Random seed number for k-means clustering.
    - include_ws_wd_in_clustering (bool): Whether to include ws and wd in clustering.

    Returns:
    - fig (plotly.graph_objects.Figure): The resulting polar cluster plot.
    """
    # Validate input columns
    required_cols = [ws_col, wd_col] + feature_cols
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Drop rows with missing data in required columns
    data = df[required_cols].dropna()

    # Ensure wind directions are within [0, 360)
    data[wd_col] = data[wd_col] % 360

    # Bin the data into wind speed and wind direction bins
    ws_min, ws_max = data[ws_col].min(), data[ws_col].max()
    ws_bins_array = np.linspace(ws_min, ws_max, ws_bins + 1)
    wd_bins_array = np.linspace(0, 360, wd_bins + 1)

    # Assign bins to data
    data['ws_bin'] = pd.cut(data[ws_col], bins=ws_bins_array, labels=False, include_lowest=True)
    data['wd_bin'] = pd.cut(data[wd_col], bins=wd_bins_array, labels=False, include_lowest=True)

    # Group data by bins and compute aggregated features
    grouped = data.groupby(['ws_bin', 'wd_bin'])

    # Create a dictionary of aggregations for feature_cols
    aggregations = {f"{col}_mean": (col, 'mean') for col in feature_cols}

    # Now pass the aggregations to the agg() function
    binned_data = grouped.agg(
        ws_mean=(ws_col, 'mean'),
        wd_mean=(wd_col, 'mean'),
        count=(ws_col, 'count'),
        **aggregations
    ).reset_index()

    # Remove bins with too few data points
    binned_data = binned_data[binned_data['count'] >= min_count]

    # Prepare data for clustering
    feature_mean_cols = [f"{col}_mean" for col in feature_cols]

    if include_ws_wd_in_clustering:
        # Transform wd_mean into sine and cosine components
        wd_mean_rad = np.deg2rad(binned_data['wd_mean'])
        binned_data['wd_sin'] = np.sin(wd_mean_rad)
        binned_data['wd_cos'] = np.cos(wd_mean_rad)
        feature_mean_cols.extend(['ws_mean', 'wd_sin', 'wd_cos'])
    else:
        pass  # Do nothing

    feature_data = binned_data[feature_mean_cols].values

    # Standardize features
    scaler = StandardScaler()
    feature_data_scaled = scaler.fit_transform(feature_data)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    clusters = kmeans.fit_predict(feature_data_scaled)
    binned_data['cluster'] = clusters

    # Map clusters to colors
    colors = color_palette

    if len(colors) < n_clusters:
        raise ValueError("Not enough colors in the palette for the number of clusters.")

    cluster_colors = {cluster: colors[i % len(colors)] for i, cluster in enumerate(np.unique(clusters))}

    # Initialize shapes list
    shapes = []

    # Loop over bins and create tiles
    for idx, row in binned_data.iterrows():
        cluster = row['cluster']
        fillcolor = cluster_colors[cluster]

        # Get bin edges
        ws_bin = int(row['ws_bin'])
        wd_bin = int(row['wd_bin'])

        ws0 = ws_bins_array[ws_bin]
        ws1 = ws_bins_array[ws_bin + 1]
        wd0 = wd_bins_array[wd_bin]
        wd1 = wd_bins_array[wd_bin + 1]

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

        # Create path for the shape
        path = f'M {x0},{y0} L {x1},{y1} L {x2},{y2} L {x3},{y3} Z'

        # Create shape dictionary
        shape = dict(
            type='path',
            path=path,
            fillcolor=fillcolor,
            line=dict(width=0.5, color=fillcolor),
            xref='x',
            yref='y',
            layer='below'
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

    # Add legend
    legend_items = []
    for cluster, color in cluster_colors.items():
        legend_items.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                legendgroup=str(cluster),
                showlegend=True,
                name=f'Cluster {cluster}'
            )
        )
    fig.add_traces(legend_items)

    # Adjust compass annotations to only N, E, S, W with black arrows
    compass_directions = ['N', 'E', 'S', 'W']
    compass_angles = np.array([0, 90, 180, 270])
    compass_angles_adj = (-compass_angles + 90) % 360
    compass_angles_rad = np.deg2rad(compass_angles_adj)

    # Define anchors and shifts based on direction to position text correctly
    xanchors = {'N': 'center', 'E': 'left', 'S': 'center', 'W': 'right'}
    yanchors = {'N': 'bottom', 'E': 'middle', 'S': 'top', 'W': 'middle'}
    x_shifts = {'N': 0, 'E': 5, 'S': 0, 'W': -5}
    y_shifts = {'N': 5, 'E': 0, 'S': -5, 'W': 0}

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
            xshift=x_shifts[direction],
            yshift=y_shifts[direction],
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
