from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian

from correlation.plot_clusters import visualize_clusters_tsne, visualize_clusters_umap
from models.optimizer_utils import get_objective_weights, strategy_performance_metrics
from correlation.spectral_optimize import (
    compute_affinity_matrix_rbf,
    run_spectral_affinity_study,
)
from correlation.cluster_utils import get_clusters_top_performers
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle
from utils import logger


def get_cluster_labels_spectral(
    returns_df: pd.DataFrame,
    cache_dir: str = "optuna_cache",
    reoptimize: bool = False,
) -> Dict[str, int]:
    """
    Cluster assets based on their return correlations using spectral clustering.
    The affinity matrix is computed using an RBF kernel transformation that is tuned via an Optuna study.
    The asset clustering map (mapping tickers to cluster labels) is created within this function.
    """
    # Set up cache for parameters.
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_filename = cache_path / "clustering_spectral_params.pkl"

    # Try to load cached parameters.
    cached_params = load_parameters_from_pickle(cache_filename) or {}
    if not reoptimize and "gamma" in cached_params and "n_clusters" in cached_params:
        gamma = cached_params["gamma"]
        n_clusters = cached_params["n_clusters"]
        logger.info("Using cached spectral parameters.")
    else:
        logger.info("Optimizing spectral clustering parameters...")
        best_params = run_spectral_affinity_study(returns_df, n_trials=50)
        gamma = best_params["gamma"]
        n_clusters = best_params["n_clusters"]
        cached_params = {"gamma": gamma, "n_clusters": n_clusters}
        save_parameters_to_pickle(cached_params, cache_filename)
        logger.info("Optimized spectral parameters saved to cache.")

    # Compute the tuned affinity matrix.
    affinity = compute_affinity_matrix_rbf(returns_df, gamma)

    # Run spectral clustering using the optimized parameters.
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )
    cluster_labels = spectral.fit_predict(affinity)
    asset_cluster_map = dict(zip(returns_df.columns, cluster_labels))

    num_clusters = len(np.unique(np.array(list(asset_cluster_map.values()))))
    logger.info(f"Spectral clustering produced {num_clusters} clusters.")

    return asset_cluster_map


def estimate_n_clusters(
    affinity: np.ndarray, max_clusters: Optional[int] = None
) -> int:
    """
    Estimate the number of clusters using eigen gap analysis on the normalized Laplacian.

    Args:
        affinity (np.ndarray): The affinity matrix.
        max_clusters (int, optional): Maximum number of clusters to consider.
            Defaults to min(n_assets - 1, 10).

    Returns:
        int: Estimated number of clusters.
    """
    n_assets = affinity.shape[0]
    if max_clusters is None:
        max_clusters = min(n_assets - 1, 10)

    # Compute the normalized Laplacian
    L = laplacian(affinity, normed=True)

    # Compute the eigenvalues and sort them in ascending order.
    eigenvalues = np.linalg.eigvals(L)
    eigenvalues = np.sort(np.real(eigenvalues))

    # Compute the differences (gaps) between consecutive eigenvalues
    gaps = np.diff(eigenvalues[: max_clusters + 1])
    estimated_k = (
        int(np.argmax(gaps)) + 1
    )  # +1 because gap between eigenvalue[i] and eigenvalue[i+1]

    logger.info(f"Eigen gap analysis suggests {estimated_k} clusters.")
    return estimated_k


def filter_correlated_groups_spectral(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    n_clusters: Optional[int] = None,
    objective: str = "sharpe",
    plot: bool = False,
    cache_dir: str = "optuna_cache",
    reoptimize: bool = False,
    top_n: Optional[int] = None,
) -> List[str]:
    """
    Uses spectral clustering on the asset returns to create fine‑grained, tight clusters.
    The affinity is computed from the correlation matrix (scaled to [0, 1]). If n_clusters is not provided,
    eigen gap analysis is used to estimate an optimal number.

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and assets as columns.
        risk_free_rate (float): Risk-free rate for performance metric calculation.
        n_clusters (int, optional): The number of clusters to form. If None, it is estimated via eigen gap analysis.
        objective (str): The performance metric objective for ranking assets.
        plot (bool): If True, display TSNE and UMAP visualizations of the clusters.
        top_n (int): Manual override for top n performers selection from each cluster.

    Returns:
        List[str]: A list of selected asset tickers.
    """
    n_assets = returns_df.shape[1]

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_filename = cache_path / "clustering_spectral_params.pkl"
    cached_params = load_parameters_from_pickle(cache_filename) or {}

    if not reoptimize and "gamma" in cached_params and "n_clusters" in cached_params:
        gamma = cached_params["gamma"]
        n_clusters_cached = cached_params["n_clusters"]
        if n_clusters is None:
            n_clusters = n_clusters_cached
        logger.info(
            f"Using cached spectral parameters: gamma = {gamma}, n_clusters = {n_clusters}"
        )
    else:
        logger.info("Optimizing spectral clustering parameters...")
        best_params = run_spectral_affinity_study(returns_df, n_trials=50)
        gamma = best_params["gamma"]
        n_clusters_optimized = best_params["n_clusters"]
        if n_clusters is None:
            n_clusters = n_clusters_optimized
        cached_params = {"gamma": gamma, "n_clusters": n_clusters_optimized}
        save_parameters_to_pickle(cached_params, cache_filename)
        logger.info("Optimized spectral parameters saved to cache.")

    # Compute the affinity matrix using the RBF kernel transformation.
    affinity = compute_affinity_matrix_rbf(returns_df, gamma)
    logger.info(f"Using n_clusters = {n_clusters} for {n_assets} assets.")

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
    # Select the best-performing tickers from each cluster
    selected_tickers = get_clusters_top_performers(clusters, perf_series, top_n=top_n)

    if plot:
        visualize_clusters_tsne(
            returns_df,
            cluster_labels,
            title="t-SNE Visualization of Asset Clusters via Spectral Clustering",
        )
        visualize_clusters_umap(
            returns_df=returns_df,
            cluster_labels=cluster_labels,
            n_neighbors=50,
            min_dist=0.5,
            title="UMAP Visualization of Asset Clusters via Spectral Clustering",
        )

    removed_tickers = set(tickers) - set(selected_tickers)
    logger.info(
        f"Removed {len(removed_tickers)} assets; {len(selected_tickers)} assets remain."
    )

    return selected_tickers
