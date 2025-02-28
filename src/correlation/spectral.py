import math
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
import logging
from typing import List, Optional

from correlation.plot_clusters import visualize_clusters_tsne
from models.optimizer_utils import get_objective_weights, strategy_performance_metrics

logger = logging.getLogger(__name__)


def compute_affinity_matrix(returns_df: pd.DataFrame) -> np.ndarray:
    """
    Compute the correlation matrix for the asset returns and convert it
    into an affinity matrix in [0, 1]. A simple way is to map correlation values
    from [-1, 1] to [0, 1] via: affinity = (corr + 1) / 2.

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and assets as columns.

    Returns:
        np.ndarray: The affinity matrix.
    """
    corr_matrix = returns_df.corr().values
    # Map correlation from [-1, 1] to [0, 1]
    affinity = (corr_matrix + 1) / 2.0
    return affinity


def filter_correlated_groups_spectral(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    n_clusters: Optional[int] = None,
    top_n_per_cluster: int = 1,
    objective: str = "sharpe",
    plot: bool = False,
) -> List[str]:
    """
    Uses spectral clustering on the asset returns to create fine‑grained, tight clusters.
    The affinity is computed from the correlation matrix (scaled to [0, 1]).

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and assets as columns.
        risk_free_rate (float): Risk-free rate for performance metric calculation.
        n_clusters (int, optional): The number of clusters to form. If None, it is set to 10% of assets.
        top_n_per_cluster (int): How many top assets to select from each cluster.
        plot (bool): If True, display a TSNE visualization of the clusters.

    Returns:
        List[str]: A list of selected asset tickers.
    """
    n_assets = returns_df.shape[1]
    # If n_clusters is not specified, set it to roughly 10% of the assets.
    if n_clusters is None:
        n_clusters = max(2, math.ceil(n_assets * 0.1))
        logger.info(
            f"n_clusters not provided. Setting n_clusters = {n_clusters} for {n_assets} assets."
        )

    affinity = compute_affinity_matrix(returns_df)

    # Run spectral clustering using the precomputed affinity matrix.
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )
    cluster_labels = spectral.fit_predict(affinity)

    # Group tickers by cluster label.
    clusters = {}
    tickers = returns_df.columns.tolist()
    for ticker, label in zip(tickers, cluster_labels):
        clusters.setdefault(label, []).append(ticker)
    logger.info(f"Spectral clustering produced {len(clusters)} clusters.")

    # Compute performance metrics
    objective_weights = get_objective_weights(objective)
    perf_series = strategy_performance_metrics(
        returns_df=returns_df,
        risk_free_rate=risk_free_rate,
        objective_weights=objective_weights,
    )

    # For each cluster, select the top performer(s)
    selected_tickers: List[str] = []
    for label, ticker_group in clusters.items():
        # You can handle noise (if any) separately; here we assume no special noise label.
        group_perf = perf_series[ticker_group].sort_values(ascending=False)
        top_candidates = group_perf.index.tolist()[:top_n_per_cluster]
        selected_tickers.extend(top_candidates)
        logger.info(
            f"Cluster {label}: {len(ticker_group)} assets; selected {top_candidates}"
        )

    if plot:
        visualize_clusters_tsne(returns_df, cluster_labels)

    removed_tickers = set(tickers) - set(selected_tickers)
    logger.info(
        f"Removed {len(removed_tickers)} assets; {len(selected_tickers)} assets remain."
    )

    return selected_tickers
