"""Taylor diagrams. Port of openair's ``TaylorDiagram``.

Puts three statistics on one plot by exploiting a geometric identity: if the
radius is the standard deviation and the angle is the inverse cosine of the
correlation, then the distance between a model's point and the observation's
point *is* the centred RMS error. Comparing several models becomes a matter of
seeing which point sits closest to the reference.
"""

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go

from .text import quick_text

__all__ = ['taylor_diagram']


def taylor_diagram(df, obs='obs', mod='mod', group=None, normalise=True,
                   colours=None, title=None, annotate=True,
                   width=780, height=720):
    """Compare one or more models against observations on a Taylor diagram.

    Parameters:
    - df (pd.DataFrame): Data containing the observed and modelled columns.
    - obs (str): Observed column.
    - mod (str or list): One modelled column, or several to compare.
    - group (str or None): Column splitting the comparison, e.g. 'season' or
      'site'. Each level becomes a separate point per model.
    - normalise (bool): Divide standard deviations by the observed standard
      deviation, so that series on different scales share one diagram. Required
      if `group` spans regimes with very different variability.
    - colours (list or None): Point colours.
    - title (str or None): Plot title.
    - annotate (bool): Label each point.
    - width, height (int): Figure size in pixels.

    Returns:
    - fig (go.Figure): The Taylor diagram.
    - summary (pd.DataFrame): sd, correlation and centred RMS error per point.

    Notes:
    - The reference point sits on the horizontal axis at the observed standard
      deviation, which is 1 when normalised. A model is better the closer it
      lies to that point.
    - Correlation is the angle: the horizontal axis is r = 1, the vertical is
      r = 0. Negative correlations cannot be drawn on the usual quarter circle
      and are reported in the summary but omitted from the plot.
    """
    if obs not in df.columns:
        raise ValueError(f"Column '{obs}' not found in the DataFrame.")
    models = [mod] if isinstance(mod, str) else list(mod)
    missing = [m for m in models if m not in df.columns]
    if missing:
        raise ValueError(f'Modelled column(s) not found: {missing}')
    if group is not None and group not in df.columns:
        raise ValueError(f"Grouping column '{group}' not found in the DataFrame.")

    rows = []
    groups = ([(None, df)] if group is None
              else list(df.groupby(group, observed=True, sort=True)))
    for level, frame in groups:
        for model in models:
            pair = frame[[obs, model]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 3:
                continue
            observed = pair[obs].to_numpy(dtype=float)
            modelled = pair[model].to_numpy(dtype=float)
            sd_obs = observed.std(ddof=1)
            sd_mod = modelled.std(ddof=1)
            if sd_obs == 0 or sd_mod == 0:
                continue
            correlation = float(np.corrcoef(observed, modelled)[0, 1])
            # Centred RMS error: the RMS of the differences after removing
            # each series' own mean, so it measures pattern rather than bias.
            #
            # Normalised by n - 1 to match the ddof=1 standard deviations
            # above. The diagram only works because E^2 = sd_f^2 + sd_r^2 -
            # 2 sd_f sd_r R, which makes the distance from the reference point
            # equal the centred RMSE; mixing n and n - 1 between the two terms
            # breaks that identity by a factor of order 1/n.
            centred = float(np.sqrt(np.sum(
                ((modelled - modelled.mean()) - (observed - observed.mean())) ** 2
            ) / (len(pair) - 1)))
            rows.append({
                'group': level, 'model': model, 'n': len(pair),
                'sd_obs': sd_obs, 'sd_mod': sd_mod, 'r': correlation,
                'centred_rmse': centred,
                'bias': float(modelled.mean() - observed.mean()),
            })

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError('No group had enough complete pairs to compare.')

    if normalise:
        summary['sd_plot'] = summary['sd_mod'] / summary['sd_obs']
        summary['rmse_plot'] = summary['centred_rmse'] / summary['sd_obs']
        reference = 1.0
    else:
        summary['sd_plot'] = summary['sd_mod']
        summary['rmse_plot'] = summary['centred_rmse']
        reference = float(summary['sd_obs'].iloc[0])

    drawable = summary[summary['r'] > 0]
    limit = max(float(drawable['sd_plot'].max()) if not drawable.empty else 0.0,
                reference) * 1.25

    fig = go.Figure()
    _add_taylor_axes(fig, reference, limit, normalise)

    colours = colours or pcolors.qualitative.Plotly
    for index, row in enumerate(drawable.itertuples()):
        angle = np.arccos(np.clip(row.r, -1.0, 1.0))
        x, y = row.sd_plot * np.cos(angle), row.sd_plot * np.sin(angle)
        label = (f'{row.model}' if row.group is None
                 else f'{row.model} - {row.group}')
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text' if annotate else 'markers',
            marker=dict(size=12, color=colours[index % len(colours)],
                        line=dict(width=1, color='white')),
            text=[label] if annotate else None, textposition='top center',
            textfont=dict(size=10),
            name=label,
            hovertemplate=(f'{label}<br>r = {row.r:.3f}<br>'
                           f'sd = {row.sd_plot:.3f}<br>'
                           f'centred RMSE = {row.rmse_plot:.3f}<extra></extra>'),
        ))

    fig.update_layout(
        title=title or f'Taylor diagram: {quick_text(obs)}',
        template='plotly_white', width=width, height=height,
        xaxis=dict(range=[0, limit * 1.14], scaleanchor='y', scaleratio=1,
                   title='standard deviation' + (' (normalised)' if normalise else ''),
                   showgrid=False, zeroline=False, constrain='domain'),
        yaxis=dict(range=[0, limit * 1.14], showticklabels=False,
                   showgrid=False, zeroline=False, constrain='domain'),
        showlegend=False,
    )
    return fig, summary


def _add_taylor_axes(fig, reference, limit, normalise):
    """Draw the standard-deviation arcs, correlation rays and RMSE contours."""
    angles = np.linspace(0, np.pi / 2, 200)

    # Arcs of constant standard deviation.
    step = _nice_step(limit)
    for radius in np.arange(step, limit, step):
        fig.add_trace(go.Scatter(
            x=radius * np.cos(angles), y=radius * np.sin(angles), mode='lines',
            line=dict(color='rgba(140,140,140,0.35)', width=1),
            hoverinfo='skip', showlegend=False,
        ))

    # Rays of constant correlation, which is what the angle encodes.
    for r in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99):
        angle = np.arccos(r)
        fig.add_trace(go.Scatter(
            x=[0, limit * np.cos(angle)], y=[0, limit * np.sin(angle)],
            mode='lines', line=dict(color='rgba(140,140,140,0.3)', width=1,
                                    dash='dot'),
            hoverinfo='skip', showlegend=False,
        ))
        fig.add_annotation(
            x=limit * 1.04 * np.cos(angle), y=limit * 1.04 * np.sin(angle),
            text=f'{r:g}', showarrow=False, font=dict(size=9, color='#555'),
        )
    fig.add_annotation(
        x=limit * 0.78 * np.cos(np.pi / 4), y=limit * 0.78 * np.sin(np.pi / 4),
        text='correlation', showarrow=False, textangle=-45,
        font=dict(size=11, color='#555'),
    )

    # Semicircles of constant centred RMS error, centred on the reference.
    for radius in np.arange(step, limit, step):
        arc = np.linspace(0, np.pi, 200)
        x = reference + radius * np.cos(arc)
        y = radius * np.sin(arc)
        inside = (x >= 0) & (np.hypot(x, y) <= limit)
        if not inside.any():
            continue
        fig.add_trace(go.Scatter(
            x=x[inside], y=y[inside], mode='lines',
            line=dict(color='rgba(70,160,70,0.35)', width=1, dash='dash'),
            hoverinfo='skip', showlegend=False,
        ))

    # The observations themselves: a perfect model would sit exactly here.
    fig.add_trace(go.Scatter(
        x=[reference], y=[0], mode='markers',
        marker=dict(size=14, color='black', symbol='star'),
        hoverinfo='skip', showlegend=False,
    ))
    # Labelled with an annotation rather than marker text: the point sits on
    # y = 0, so text below it would fall outside the axis range and clip.
    fig.add_annotation(
        x=reference, y=0, text='observed', showarrow=False,
        yshift=-14, font=dict(size=10, color='#333'),
        bgcolor='rgba(255,255,255,0.75)',
    )


def _nice_step(limit):
    """A round gridline spacing giving roughly four or five arcs."""
    raw = limit / 4.0
    magnitude = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1.0
    for multiple in (1, 2, 2.5, 5, 10):
        if raw <= multiple * magnitude:
            return multiple * magnitude
    return magnitude
