import hdbscan
import numpy as np
import pandas as pd
from pathlib import Path

from correlation.correlation_utils import compute_distance_matrix
from correlation.hdbscan_optimize import run_hdbscan_decorrelation_study
from utils.logger import logger
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def get_cluster_labels(
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
    cache_filename = cache_path / "hdbscan_params.pkl"

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
