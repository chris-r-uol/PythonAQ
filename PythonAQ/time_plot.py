import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st

def calculate_arrow_positions(x, y, ws, wd, x_scale, y_scale):
    """
    Calculate the start and end positions of an arrow representing wind speed and direction,
    centered at (x, y).

    Parameters:
    - x (float): x-coordinate (e.g., date as a numeric value)
    - y (float): y-coordinate (e.g., concentration level)
    - ws (float): wind speed in m/s
    - wd (float): wind direction in degrees (meteorological convention)
    - x_scale (float): scaling factor for x displacement
    - y_scale (float): scaling factor for y displacement

    Returns:
    - xi (float): x-coordinate of arrow start
    - yi (float): y-coordinate of arrow start
    - xf (float): x-coordinate of arrow end
    - yf (float): y-coordinate of arrow end
    """

    # Convert wind direction to radians and adjust to mathematical convention
    wd_math = (wd + 180) % 360
    wd_rad = np.deg2rad(wd_math)

    # Calculate displacement components
    dx = ws * np.cos(wd_rad) * x_scale
    dy = ws * np.sin(wd_rad) * y_scale

    # Calculate arrow start and end positions, centered at (x, y)
    xi = x - dx / 2
    xf = x + dx / 2
    yi = y - dy / 2
    yf = y + dy / 2

    return xi, yi, xf, yf


def time_plot(
    df,
    date_col='date_time',
    columns_to_plot=None,
    group_data=False,
    stack_data=False,
    normalize=False,
    normalize_date=None,
    wind_flow={'activate': False},
    averaging_period=None,
    title='Time Series Plot',
    yaxis_title='Value',
    width=1000,
    height=600,
    color_palette=None
):
    # Ensure date_col is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Sort the DataFrame by date
    df.sort_values(by=date_col, inplace=True)

    # Set default columns to plot
    if columns_to_plot is None:
        columns_to_plot = [col for col in df.columns if col != date_col]

    # Check if wind_flow is activated and set up wind columns
    wind_flow_activated = wind_flow and wind_flow.get('activate', False)

    #if wind_flow_activated:
    #    ws_col = wind_flow.get('ws_col', 'ws')
    #    wd_col = wind_flow.get('wd_col', 'wd')
    #    arrow_color = wind_flow.get('color', 'red')
    #    #arrow_scale = wind_flow.get('scale', 0.01)
    #    x_scale = wind_flow.get('scale', 1)
    #    y_scale = wind_flow.get('scale', 1)#


        # Ensure wind columns are present
    #    if ws_col not in df.columns or wd_col not in df.columns:
    #        raise ValueError("Wind speed and direction columns not found in DataFrame.")

        # Convert wind direction to radians
    #    df['wd_rad'] = np.deg2rad(df[wd_col])

        # Calculate u and v components using meteorological convention
    #    df['u'] = -df[ws_col] * np.sin(df['wd_rad'])
    #    df['v'] = -df[ws_col] * np.cos(df['wd_rad'])

    # Resample data if averaging_period is specified
    if averaging_period is not None:
        # Set date_col as index for resampling
        df.set_index(date_col, inplace=True)

        # List of columns to resample
        resample_cols = columns_to_plot.copy()

        # Include wind components if wind_flow is activated
        if wind_flow_activated:
            resample_cols.extend(['u', 'v'])
        # Resample and aggregate data
        df_resampled = df[resample_cols].resample(averaging_period).mean()

        # Reset index to bring date_col back as a column
        df_resampled.reset_index(inplace=True)

        if wind_flow_activated:
            # Recompute wind speed and direction from resampled u and v
            df_resampled['ws'] = np.sqrt(df_resampled['u']**2 + df_resampled['v']**2)
            df_resampled['wd_rad'] = np.arctan2(-df_resampled['u'], -df_resampled['v'])  # Note negative signs
            df_resampled['wd'] = (np.degrees(df_resampled['wd_rad'])) % 360

        # Update df
        df = df_resampled

        # Update date_col
        date_col = df.columns[0]
    else:
        if wind_flow_activated:
            # Wind components are already calculated
            pass

    # Normalize data if required
    if normalize:
        if normalize_date is None:
            normalize_date = df[date_col].iloc[0]
        else:
            normalize_date = pd.to_datetime(normalize_date)

        # Get normalization values
        norm_values = df[df[date_col] == normalize_date][columns_to_plot]
        if norm_values.empty:
            raise ValueError(f"No data found for normalization date {normalize_date}")
        norm_values = norm_values.iloc[0]

        # Normalize columns
        for col in columns_to_plot:
            df[col] = df[col] / norm_values[col]

    # Create list of years if stacking
    if stack_data:
        df['year'] = df[date_col].dt.year
        years = df['year'].unique()
        # Save original dates
        df['original_date'] = df[date_col]
        # Create a new column 'date_no_year' with year set to 2000
        df['date_no_year'] = df[date_col].apply(lambda x: x.replace(year=2000))
    else:
        years = [None]
        df['original_date'] = df[date_col]
        df['date_no_year'] = df[date_col]

    # Add numeric date for calculations (seconds since reference date)
    reference_date = pd.Timestamp('2000-01-01')
    df['date_numeric'] = (df['date_no_year'] - reference_date).dt.total_seconds()

    # Determine number of rows and columns for subplots
    nrows = len(years)
    if group_data:
        ncols = 1
    else:
        ncols = len(columns_to_plot)

    # Create subplots
    specs = [[{} for _ in range(ncols)] for _ in range(nrows)]
    subplot_titles = []

    for year in years:
        if group_data:
            title_text = str(year) if year is not None else ''
            subplot_titles.append(title_text)
        else:
            for col in columns_to_plot:
                title = f"{col}" + (f" ({year})" if year is not None else '')
                subplot_titles.append(title)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles
    )

    # Assign consistent colors to variables
    if color_palette is None:
        color_palette = px.colors.qualitative.Plotly  # Default color palette
    variable_colors = {}
    num_colors = len(color_palette)
    for i, var in enumerate(columns_to_plot):
        variable_colors[var] = color_palette[i % num_colors]

    # Plot data
    for i, year in enumerate(years):
        df_year = df[df['year'] == year] if year is not None else df
        row_num = i + 1

        if group_data:
            col_num = 1
            for trace_name in columns_to_plot:
                fig.add_trace(
                    go.Scatter(
                        x=df_year['date_no_year'],
                        y=df_year[trace_name],
                        mode='lines',
                        name=trace_name if row_num == 1 else trace_name + f" ({year})",
                        showlegend=(row_num == 1),
                        hovertext=df_year['original_date'],
                        hovertemplate="%{hovertext|%Y-%m-%d %H:%M}<br>%{y}<extra>%{fullData.name}</extra>",
                        line=dict(color=variable_colors[trace_name]),
                    ),
                    row=row_num,
                    col=col_num
                )
        else:
            for j, trace_name in enumerate(columns_to_plot):
                col_num = j + 1
                fig.add_trace(
                    go.Scatter(
                        x=df_year['date_no_year'],
                        y=df_year[trace_name],
                        mode='lines',
                        name=trace_name if row_num == 1 else trace_name + f" ({year})",
                        showlegend=(row_num == 1),
                        hovertext=df_year['original_date'],
                        hovertemplate="%{hovertext|%Y-%m-%d %H:%M}<br>%{y}<extra>%{fullData.name}</extra>",
                        line=dict(color=variable_colors[trace_name]),
                    ),
                    row=row_num,
                    col=col_num
                )

    # Update layout
    fig.update_layout(
        title=title,
        height=height * nrows,
        width=width,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )

    # Update y-axes titles
    for i in range(nrows):
        row_num = i + 1
        if group_data:
            fig.update_yaxes(title_text=yaxis_title, row=row_num, col=1)
        else:
            for j in range(ncols):
                col_num = j + 1
                fig.update_yaxes(title_text=columns_to_plot[j], row=row_num, col=col_num)

    # Update x-axes
    if stack_data:
        # Determine x-axis range
        xaxis_start = pd.Timestamp('2000-01-01')
        xaxis_end = pd.Timestamp('2000-12-31')

        # Update x-axes
        for i in range(nrows):
            row_num = i + 1
            for col_num in range(1, ncols + 1):
                fig.update_xaxes(
                    range=[xaxis_start, xaxis_end],
                    row=row_num,
                    col=col_num,
                    dtick="M1",
                    tickformat="%b",
                )
    else:
        # Do not adjust x-axis range, let Plotly auto-scale
        pass

    # Add wind flow arrows if required
    if wind_flow_activated:
        print('Wind flow not currently available.  This is on the to-do list so please be patient, or contribute if you like.')
        # Scale arrows
        #df['u_scaled'] = df['u'] * x_scale #arrow_scale
        #df['v_scaled'] = df['v'] * y_scale #arrow_scale

        # Iterate over the data points
        #for idx, row_data in df.iterrows():
        #    x_numeric = row_data['date_numeric']
        #    x_original = row_data['original_date']
        #    if stack_data:
        #        years_idx = np.where(years == row_data['year'])[0][0]
        #    else:
        #        years_idx = 0
        #    subplot_row = years_idx + 1#

        #    if group_data:
        #        cols = [1]
        #    else:
        #        cols = list(range(1, ncols + 1))#

        #    for col_num in cols:
        #        y = None
        #        if group_data:
        #            y_vals = [row_data[col] for col in columns_to_plot]
        #            y = np.nanmean(y_vals)
        #        else:
        #            col_name = columns_to_plot[col_num - 1]
        #            y = row_data[col_name]

        #        ws = row_data[ws_col]
        #        wd = row_data[wd_col]

                # Check for NaN values
        #        if pd.notnull(x_numeric) and pd.notnull(y) and pd.notnull(ws) and pd.notnull(wd):
        #            xi, yi, xf, yf = calculate_arrow_positions(x_numeric, y, ws, wd, x_scale, y_scale)
        #            print(f'Calculating position of arrow for {x_numeric}, {y} and ws/wd = {ws}, {wd} as:    {xi}, {xf}, {yi}, {yf}')
        #            # Convert numeric x positions back to datetime
        #            xi_datetime = reference_date + pd.Timedelta(seconds=xi)
        #            xf_datetime = reference_date + pd.Timedelta(seconds=xf)

        #            # Calculate the correct axis references
        #            axes_counter = (subplot_row - 1) * ncols + col_num
        #            if axes_counter == 1:
        #                xref = 'x'
        #                yref = 'y'
        #            else:
        #                xref = f'x{axes_counter}'
        #                yref = f'y{axes_counter}'

                    # Add the annotation
        #            fig.add_annotation(
        #                x=xf_datetime,
        #                y=yf,
        #                ax=xi_datetime,
        #                ay=yi,
        #                xref=xref,
        #                yref=yref,
        #                axref=xref,
        #                ayref=yref,
        #                showarrow=True,
        #                arrowhead=2,
        #                arrowsize=1,
        #                arrowwidth=1,
        #                arrowcolor=arrow_color,
        #                opacity=0.7,
        #                standoff=0,
        #                hovertext=(
        #                    f"Date: {x_original:%Y-%m-%d %H:%M}<br>"
        #                    f"Wind Speed: {ws:.2f}<br>"
        #                    f"Wind Direction: {wd:.1f}"
        #                ),
        #                hoverlabel=dict(bgcolor='white'),
        #            )
        #        else:
        #            # Skip the data point if any of the values are NaN
        #            continue

    return fig
