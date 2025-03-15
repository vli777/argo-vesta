from typing import List
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community as nx_comm


from correlation.correlation_utils import compute_correlation_matrix
from correlation.plot_clusters import visualize_clusters_tsne, visualize_clusters_umap
from models.optimizer_utils import get_objective_weights, strategy_performance_metrics
from correlation.cluster_utils import get_clusters_top_performers
from utils import logger


def get_cluster_labels_mst(
    returns_df: pd.DataFrame,
) -> np.ndarray:
    """
    Cluster assets by constructing a Minimum Spanning Tree (MST) from the distance matrix
    (derived from the correlation matrix of returns) and then detecting communities using
    the greedy modularity algorithm.

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and asset returns as columns.

    Returns:
        np.ndarray: Array of integer cluster labels (one per asset, in the order of returns_df.columns).
    """
    # Step 1: Compute the correlation matrix from returns.
    corr = compute_correlation_matrix(returns_df)

    # Step 2: Convert the correlation matrix into a distance matrix.
    dist = 1 - corr

    n = dist.shape[0]
    tickers = returns_df.columns.tolist()

    # Step 3: Build a complete weighted graph from the distance matrix.
    # networkx.from_numpy_array creates nodes [0, 1, ..., n-1]; we relabel them using tickers.
    # Step 3: Build a complete weighted graph from the distance matrix.
    G_complete = nx.from_numpy_array(dist.values)
    mapping = {i: tickers[i] for i in range(n)}
    G_complete = nx.relabel_nodes(G_complete, mapping)

    # Step 4: Compute the Minimum Spanning Tree (MST) of the complete graph.
    MST = nx.minimum_spanning_tree(G_complete, weight="weight")

    # Step 5: Run community detection on the MST.
    # Using the greedy modularity communities algorithm.
    communities = list(nx_comm.greedy_modularity_communities(MST, weight="weight"))

    # Build the asset cluster map.
    asset_cluster_map = {}
    for label, community in enumerate(communities):
        for ticker in community:
            asset_cluster_map[ticker] = label

    logger.info(f"MST community detection produced {len(communities)} clusters.")
    return asset_cluster_map


def filter_correlated_groups_mst(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    objective: str = "sharpe",
    plot: bool = False,
) -> List[str]:
    """
    Uses MST community detection to group assets, then selects top performers from each community.
    """
    # Obtain the asset cluster map using MST.
    asset_cluster_map = get_cluster_labels_mst(returns_df)

    # Group tickers by their assigned cluster label.
    clusters = {}
    for ticker, label in asset_cluster_map.items():
        clusters.setdefault(label, []).append(ticker)

    # Compute performance metrics.
    objective_weights = get_objective_weights(objective)
    perf_series = strategy_performance_metrics(
        returns_df=returns_df,
        risk_free_rate=risk_free_rate,
        objective_weights=objective_weights,
    )

    # Select the best-performing tickers from each cluster.
    selected_tickers = get_clusters_top_performers(clusters, perf_series)

    if plot:
        # Generate visualizations using the cluster labels in the original order.
        labels_in_order = [asset_cluster_map[ticker] for ticker in returns_df.columns]
        visualize_clusters_tsne(
            returns_df,
            cluster_labels=labels_in_order,
            title="t-SNE Visualization of Asset Clusters via MST Community Detection",
        )
        visualize_clusters_umap(
            returns_df=returns_df,
            cluster_labels=labels_in_order,
            n_neighbors=50,
            min_dist=0.5,
            title="UMAP Visualization of Asset Clusters via MST Community Detection",
        )

    removed_tickers = set(returns_df.columns) - set(selected_tickers)
    logger.info(
        f"Removed {len(removed_tickers)} assets; {len(selected_tickers)} assets remain."
    )

    return selected_tickers
