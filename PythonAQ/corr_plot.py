"""Correlation matrices. Port of openair's ``corPlot``."""

import numpy as np
import plotly.graph_objects as go
from .faceting import conditionable

__all__ = ['corr_plot']


def _cluster_order(corr):
    """Order variables by hierarchical clustering of their correlations.

    Falls back to the input order if SciPy's clustering module is unavailable.
    """
    if len(corr) < 3:
        return list(corr.columns)
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        return list(corr.columns)

    distance = 1.0 - corr.abs().to_numpy()
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0  # enforce exact symmetry
    if not np.isfinite(distance).all():
        return list(corr.columns)
    order = dendrogram(linkage(squareform(distance, checks=False), method='average'),
                       no_plot=True)['leaves']
    return [corr.columns[i] for i in order]


@conditionable
def corr_plot(df, pollutants=None, method='pearson', cluster=True,
              annotate=True, colorscale='RdBu', title='Correlation Matrix',
              width=800, height=750, min_periods=10):
    """Plot a correlation matrix between pollutants.

    Parameters:
    - df (pd.DataFrame): Input data.
    - pollutants (list or None): Columns to correlate; defaults to every
      numeric column.
    - method (str): 'pearson', 'spearman' or 'kendall'.
    - cluster (bool): Order variables by hierarchical clustering, so that
      related species sit together.
    - annotate (bool): Print the coefficient in each cell.
    - colorscale (str): Diverging colour scale name.
    - title (str): Plot title.
    - width, height (int): Figure size in pixels.
    - min_periods (int): Minimum overlapping observations per pair.

    Returns:
    - fig (go.Figure): The correlation heat map.
    - corr (pd.DataFrame): The correlation matrix, in plotted order.
    """
    if method not in ('pearson', 'spearman', 'kendall'):
        raise ValueError("method must be 'pearson', 'spearman' or 'kendall'.")

    if pollutants is None:
        data = df.select_dtypes(include=[np.number])
    else:
        missing = [p for p in pollutants if p not in df.columns]
        if missing:
            raise ValueError(f"Column(s) not found in the DataFrame: {missing}")
        data = df[list(pollutants)].select_dtypes(include=[np.number])

    # Constant columns yield undefined correlations; drop them up front.
    data = data.loc[:, data.nunique(dropna=True) > 1]
    if data.shape[1] < 2:
        raise ValueError('At least two varying numeric columns are required.')

    corr = data.corr(method=method, min_periods=min_periods)
    if cluster:
        order = _cluster_order(corr)
        corr = corr.loc[order, order]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=list(corr.columns), y=list(corr.index),
        colorscale=colorscale, zmin=-1, zmax=1, zmid=0,
        colorbar=dict(title=f'{method}<br>r'),
        hovertemplate='%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>',
    ))

    if annotate:
        for i, row_name in enumerate(corr.index):
            for j, col_name in enumerate(corr.columns):
                value = corr.iloc[i, j]
                if not np.isfinite(value):
                    continue
                fig.add_annotation(
                    x=col_name, y=row_name, text=f'{value:.2f}', showarrow=False,
                    font=dict(size=11,
                              color='white' if abs(value) > 0.55 else 'black'),
                )

    fig.update_layout(
        title=f'{title} ({method})',
        template='plotly_white', width=width, height=height,
        xaxis=dict(side='bottom', tickangle=-45, constrain='domain'),
        yaxis=dict(autorange='reversed', scaleanchor='x', constrain='domain'),
    )
    return fig, corr
