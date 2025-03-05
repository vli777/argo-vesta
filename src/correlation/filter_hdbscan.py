from typing import Dict, List, Union
import numpy as np
import pandas as pd

from models.optimizer_utils import strategy_performance_metrics, get_objective_weights
from correlation.plot_clusters import visualize_clusters_tsne, visualize_clusters_umap
from utils.logger import logger


def filter_correlated_groups_hdbscan(
    returns_df: pd.DataFrame,
    asset_cluster_map: Dict[str, int],
    risk_free_rate: float = 0.0,
    plot: bool = False,
    objective: str = "sharpe",
) -> list[str]:
    """
    Uses HDBSCAN to cluster assets based on the distance (1 - correlation) matrix.
    Then, for each cluster, selects the top performing asset(s) based on a composite
    performance metric computed internally.

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and assets as columns.
        asset_cluster_map (dict): Full set of assets mapped to optimized HDBSCAN clusters.
        risk_free_rate (float): Risk-free rate for performance metric calculation.
        plot (bool): If True, display a visualization of clusters.
        objective (str): Optimization objective used as top cluster candidate selection

    Returns:
        list(str): A list of selected ticker symbols after decorrelation.
    """
    # Ensure asset_cluster_map only contains tickers present in returns_df
    asset_cluster_map = {
        ticker: label
        for ticker, label in asset_cluster_map.items()
        if ticker in returns_df.columns
    }
    # Group tickers by their cluster label
    clusters = {}
    for ticker, label in asset_cluster_map.items():
        clusters.setdefault(label, []).append(ticker)

    # Compute performance metrics for each asset
    objective_weights = get_objective_weights(objective)
    perf_series = strategy_performance_metrics(
        returns_df=returns_df,
        risk_free_rate=risk_free_rate,
        objective_weights=objective_weights,
    )

    # Select the best-performing tickers from each cluster
    selected_tickers: list[str] = []
    for label, tickers in clusters.items():
        # For noise (label == -1), include all tickers
        if label == -1:
            selected_tickers.extend(tickers)
        else:
            group_perf = perf_series[tickers].sort_values(ascending=False)
            if len(tickers) < 10:
                top_n = len(tickers)
            elif len(tickers) < 20:
                top_n = max(1, int(0.50 * len(tickers)))
            else:
                top_n = max(1, int(0.33 * len(tickers)))
            top_candidates = group_perf.index.tolist()[:top_n]
            selected_tickers.extend(top_candidates)
            logger.info(
                f"Cluster {label}: {len(tickers)} assets; keeping {top_candidates}"
            )

    removed_tickers = set(returns_df.columns) - set(selected_tickers)
    if removed_tickers:
        logger.info(
            f"Removed {len(removed_tickers)} assets due to high correlation: {sorted(removed_tickers)}"
        )
    else:
        logger.info("No assets were removed.")
    logger.info(f"{len(selected_tickers)} assets remain")

    # Optionally, visualize the clusters using UMAP or TSNE
    if plot:
        labels_in_order = np.array(
            [asset_cluster_map[ticker] for ticker in returns_df.columns]
        )

        visualize_clusters_umap(
            returns_df=returns_df,
            cluster_labels=labels_in_order,
            n_neighbors=50,
            min_dist=0.5,
            title="UMAP Visualization of Asset Clusters via HDBSCAN"
        )

        visualize_clusters_tsne(
            returns_df=returns_df,
            cluster_labels=labels_in_order,
            title="t-SNE Visualization of Asset Clusters via HDBSCAN"
        )

    return selected_tickers
