from typing import Dict, List, Union
import hdbscan
import numpy as np
import pandas as pd
from pathlib import Path

from models.optimizer_utils import strategy_performance_metrics, get_objective_weights
from correlation.correlation_utils import compute_distance_matrix
from correlation.hdbscan_optimize import run_hdbscan_decorrelation_study
from correlation.plot_clusters import visualize_clusters_tsne, visualize_clusters_umap
from correlation.cluster_utils import get_clusters_top_performers
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle
from utils.logger import logger


def get_cluster_labels_hdbscan(
    returns_df: pd.DataFrame,
    cache_dir: str = "optuna_cache",
    reoptimize: bool = False,
    scale_distances: bool = False,
) -> dict[str, int]:
    """
    Cluster assets based on their return correlations using HDBSCAN.
    HDBSCAN parameters are optionally optimized via an Optuna study (with caching).
    Clusters are recomputed every time, ensuring any new or slightly changed data
    is captured.
    """
    # Set up cache for parameters.
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_filename = cache_path / "clustering_hdbscan_params.pkl"

    # Try to load cached parameters.
    cached_params = load_parameters_from_pickle(cache_filename) or {}
    required_params = [
        "epsilon",
        "alpha",
        "cluster_selection_epsilon_max",
        # "min_cluster_size",
        # "min_samples",
    ]
    if not reoptimize and all(param in cached_params for param in required_params):
        epsilon = cached_params["epsilon"]
        alpha = cached_params["alpha"]
        cluster_selection_epsilon_max = cached_params["cluster_selection_epsilon_max"]
        # min_cluster_size = cached_params["min_cluster_size"]
        # min_samples = cached_params["min_samples"]
        logger.info("Using cached HDBSCAN parameters.")
    else:
        logger.info("Optimizing HDBSCAN parameters...")
        best_params = run_hdbscan_decorrelation_study(
            returns_df=returns_df, n_trials=100, scale_distances=scale_distances
        )
        epsilon = best_params["epsilon"]
        alpha = best_params["alpha"]
        cluster_selection_epsilon_max = best_params["cluster_selection_epsilon_max"]
        # min_cluster_size = best_params["min_cluster_size"]
        # min_samples = best_params["min_samples"]
        cached_params = {
            "epsilon": epsilon,
            "alpha": alpha,
            "cluster_selection_epsilon_max": cluster_selection_epsilon_max,
            "min_cluster_size": 2,  # min_cluster_size,
            # "min_samples": min_samples,
        }
        save_parameters_to_pickle(cached_params, cache_filename)
        logger.info("Optimized parameters saved to cache.")

    # Compute the distance matrix (with optional re-scaling).
    distance_matrix = compute_distance_matrix(
        returns_df, scale_distances=scale_distances
    )

    # For reproducibility.
    np.random.seed(42)

    # Initialize HDBSCAN with the tuned parameters.
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        alpha=alpha,
        min_cluster_size=2,  # min_cluster_size,
        # min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method="leaf",
        cluster_selection_epsilon_max=cluster_selection_epsilon_max,
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)

    # Map each asset (ticker) to its assigned cluster.
    asset_cluster_map = dict(zip(returns_df.columns, cluster_labels))

    # Log a brief summary.
    labels_array = np.array(list(asset_cluster_map.values()))
    non_noise = labels_array[labels_array != -1]
    num_clusters = len(np.unique(non_noise))
    num_noise = np.sum(labels_array == -1)
    logger.info(f"Clusters found: {num_clusters}; Noise points: {num_noise}")

    return asset_cluster_map


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
    selected_tickers = get_clusters_top_performers(clusters, perf_series)

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
            title="UMAP Visualization of Asset Clusters via HDBSCAN",
        )

        visualize_clusters_tsne(
            returns_df=returns_df,
            cluster_labels=labels_in_order,
            title="t-SNE Visualization of Asset Clusters via HDBSCAN",
        )

    return selected_tickers
