import math
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from correlation.cluster_utils import get_clusters_top_performers
from correlation.plot_clusters import visualize_clusters_tsne, visualize_clusters_umap
from models.optimizer_utils import get_objective_weights, strategy_performance_metrics
from utils import logger

def cluster_kmeans(corr: np.ndarray, max_clusters: int = 10) -> np.ndarray:
    """
    Cluster assets using KMeans on the correlation matrix.

    Args:
        corr (np.ndarray): Correlation matrix.
        max_clusters (int): Maximum number of clusters to try.

    Returns:
        np.ndarray: Cluster labels for each asset.
    """
    # Transform correlation to a distance metric.
    dist = np.sqrt(0.5 * (1 - corr))
    n_samples = dist.shape[0]

    max_valid_clusters = min(max_clusters, n_samples - 1) if n_samples > 1 else 1
    best_silhouette = -1.0
    best_labels = None

    for k in range(2, max_valid_clusters + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(dist)
        if len(np.unique(labels)) < 2:
            continue
        silhouette = silhouette_samples(dist, labels).mean()
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_labels = labels

    if best_labels is None:
        best_labels = np.zeros(n_samples, dtype=int)

    return best_labels


def get_cluster_labels_kmeans(
    returns_df: pd.DataFrame,
    max_clusters: int = 10,
) -> Dict[str, int]:
    """
    Cluster assets based on their return correlations using KMeans.
    Computes the correlation matrix from returns, transforms it into a distance metric,
    and then applies KMeans clustering over a range of cluster sizes.
    The best clustering is chosen based on the silhouette score.

    Args:
        returns_df (pd.DataFrame): DataFrame with asset returns.
        max_clusters (int): Maximum number of clusters to consider.

    Returns:
        Dict[str, int]: Mapping of asset tickers to cluster labels.
    """
    # Compute correlation matrix and apply KMeans clustering.
    corr = returns_df.corr().values
    labels = cluster_kmeans(corr, max_clusters=max_clusters)
    asset_cluster_map = dict(zip(returns_df.columns, labels))
    
    num_clusters = len(np.unique(labels))
    logger.info(f"KMeans clustering produced {num_clusters} clusters.")
    
    return asset_cluster_map


def filter_correlated_groups_kmeans(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    objective: str = "sharpe",
    plot: bool = False,
    top_n: Optional[int] = None,
    max_clusters: int = 10
) -> List[str]:
    """
    Uses KMeans clustering to group assets based on their correlation, then selects
    top performers from each cluster.

    Args:
        returns_df (pd.DataFrame): DataFrame with asset returns.
        risk_free_rate (float): Risk-free rate for performance metrics calculation.
        objective (str): Objective metric for performance evaluation (e.g., 'sharpe').
        plot (bool): Whether to generate cluster visualizations.
        top_n (Optional[int]): If specified, selects exactly top_n tickers from each cluster.
        max_clusters (int): Maximum number of clusters to consider in KMeans.

    Returns:
        List[str]: List of selected asset symbols after filtering.
    """
    # Obtain the asset cluster map using KMeans clustering.
    asset_cluster_map = get_cluster_labels_kmeans(returns_df, max_clusters=max_clusters)

    # Group tickers by their assigned cluster label.
    clusters: Dict[int, List[str]] = {}
    for ticker, label in asset_cluster_map.items():
        clusters.setdefault(label, []).append(ticker)

    # Compute performance metrics for each asset.
    objective_weights = get_objective_weights(objective)
    perf_series = strategy_performance_metrics(
        returns_df=returns_df,
        risk_free_rate=risk_free_rate,
        objective_weights=objective_weights,
    )

    # Select the best-performing tickers from each cluster.
    selected_tickers = get_clusters_top_performers(clusters, perf_series, top_n=top_n)

    if plot:
        # Visualize clusters in the order of the original DataFrame columns.
        labels_in_order = [asset_cluster_map[ticker] for ticker in returns_df.columns]
        visualize_clusters_tsne(
            returns_df,
            cluster_labels=labels_in_order,
            title="t-SNE Visualization of Asset Clusters via KMeans Clustering",
        )
        visualize_clusters_umap(
            returns_df=returns_df,
            cluster_labels=labels_in_order,
            n_neighbors=50,
            min_dist=0.5,
            title="UMAP Visualization of Asset Clusters via KMeans Clustering",
        )

    removed_tickers = set(returns_df.columns) - set(selected_tickers)
    logger.info(
        f"Removed {len(removed_tickers)} assets; {len(selected_tickers)} assets remain after KMeans filtering."
    )

    return selected_tickers
