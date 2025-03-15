import math
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict
from functools import partial
from multiprocessing import Manager

from models.optimize_portfolio import (
    optimize_weights_objective,
)
from models.optimization_plot import plot_global_optimization
from models.scipy_objective_models import sharpe_objective
from correlation.hdbscan_clustering import get_cluster_labels_hdbscan
from correlation.networkx_clustering import get_cluster_labels_mst
from correlation.spectral_clustering import get_cluster_labels_spectral
from correlation.kmeans_clustering import cluster_kmeans
from utils.logger import logger


def get_cluster_labels_from_map(
    returns_df: pd.DataFrame, corr: np.ndarray, cluster_method: str = "kmeans", **kwargs
) -> np.ndarray:
    """
    Cluster assets using the specified method and return an array of cluster labels in the
    order of returns_df.columns.

    Depending on the 'cluster_method' argument, one of the following functions is used:
        - "mst": uses Minimum Spanning Tree community detection (get_cluster_labels_mst)
        - "hdbscan": uses HDBSCAN clustering (get_cluster_labels)
        - "spectral": uses Spectral Clustering (get_cluster_labels_spectral)

    Any additional keyword arguments are passed to the underlying clustering function.

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and asset returns as columns.
        corr: (np.ndarray): Correlation matrix used for default kmeans method
        cluster_method (str): Clustering method to use ("mst", "hdbscan", or "spectral").

    Returns:
        np.ndarray: Array of integer cluster labels corresponding to the order of returns_df.columns.
    """
    if cluster_method.lower() == "mst":
        asset_cluster_map = get_cluster_labels_mst(returns_df)
    elif cluster_method.lower() == "hdbscan":
        asset_cluster_map = get_cluster_labels_hdbscan(returns_df, **kwargs)
    elif cluster_method.lower() == "spectral":
        asset_cluster_map = get_cluster_labels_spectral(returns_df, **kwargs)
    else:
        labels = cluster_kmeans(
            corr, max_clusters=math.ceil(np.sqrt(len(returns_df.columns)))
        )
        return labels

    tickers = returns_df.columns.tolist()
    labels = np.array([asset_cluster_map[ticker] for ticker in tickers])
    return labels


def cov_to_corr(cov):
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # Numerical stability
    return corr


def unified_callback(x, *args, history_list, overall_history):
    """
    A unified callback that appends the candidate x to both the local history list
    and the global overall history.
    """
    # If additional arguments are provided, you can extract f if needed:
    if args:
        f = args[0]
    overall_history.append(np.copy(x))
    history_list.append(np.copy(x))
    return False


def default_intra_callback(x, *args, cl, history_dict, overall_history):
    """
    A default callback that appends the candidate x to the history for cluster cl,
    and also records it in the global overall history.
    """
    if args:
        f = args[0]
    history_dict.setdefault(cl, []).append(np.copy(x))
    overall_history.append(np.copy(x))
    return False


def default_inter_callback(x, *args, history_list, overall_history):
    """
    Callback for inter-cluster optimization, recording the candidate x in both
    the inter-cluster history and the global overall history.
    """
    if args:
        f = args[0]
    overall_history.append(np.copy(x))
    history_list.append(np.copy(x))
    return False


def local_solver_callback(xk, *args, history_list, overall_history):
    """
    Callback for the local solver optimization that records the candidate xk
    in both the provided local history and the global overall history.
    """
    overall_history.append(np.copy(xk))
    history_list.append(np.copy(xk))
    return False


def combined_intra_callback(x, *args, cl, history_dict, local_history, overall_history):
    """
    Combined callback for intra-cluster optimization that records the candidate x
    in the cluster-specific history, the local solver history, and the global overall history.

    This function supports both global optimization (which may pass additional arguments)
    and local optimization (which may only pass x).
    """
    if args:
        f = args[0]
    history_dict.setdefault(cl, []).append(np.copy(x))
    local_history.append(np.copy(x))
    overall_history.append(np.copy(x))
    return False


def nested_clustered_optimization(
    cov: pd.DataFrame,
    mu: Optional[pd.Series] = None,
    returns: Optional[pd.DataFrame] = None,
    objective: str = "sharpe",
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    allow_short: bool = False,
    max_gross_exposure: float = 1.3,
    target: Optional[float] = None,
    order: int = 3,
    target_sum: float = 1.0,
    risk_free_rate: float = 0.0,
    use_annealing: bool = False,
    use_diffusion: bool = False,
    plot: bool = False,
    cluster_method: str = "kmeans",
) -> pd.Series:
    """
    Perform Nested Clustered Optimization with a flexible objective.
    For objectives requiring historica returns a 'returns' DataFrame must be provided.

    Args:
        cov (pd.DataFrame): Covariance matrix of asset returns.
        mu (Optional[pd.Series]): Expected returns.
        returns (Optional[pd.DataFrame]): Historical returns (time series) with assets as columns.
        objective (str): Optimization objective.
        min_weight (float): Minimum weight per asset.
        max_weight (float): Maximum weight per asset.
        allow_short (bool): Allow short positions.
        max_gross_exposure (float): Maximum gross exposure when allowing short positions.
        target (float): Target threshold for Omega ratio tau.
        order (int): Order for downside risk metrics.
        target_sum (float): Sum of weights (default 1.0).
        risk_free_rate (float): Risk free rate (default 0.0).
        use_annealing (bool): Use dual annealing to search for global optima.
        use_diffusion (bool): Use stochastic diffusion to search for global optima.
        plot (bool): Whether to plot the optimization path
        cluster_method (str): Clustering method (default 'kmeans')

    Returns:
        pd.Series: Final portfolio weights.
    """
    manager = Manager()
    # Create a global manager list to store all candidates from parallel optimization runs.
    overall_history = manager.list()

    # Filter assets with sufficient historical data
    valid_assets = cov.index
    if returns is not None:
        valid_assets = returns.columns[
            returns.notna().sum(axis=0) >= (returns.shape[0] * 0.5)
        ]
        returns = returns[valid_assets]
        cov = cov.loc[valid_assets, valid_assets]
        mu = mu.reindex(valid_assets).fillna(0) if mu is not None else None

    if len(valid_assets) < 2:
        logger.warning(
            "Not enough valid assets after filtering. Skipping optimization."
        )
        return pd.Series(dtype=float)

    # Set target if not explicitly provided
    if target is None and returns is not None:
        target = max(risk_free_rate, np.percentile(returns.values.flatten(), 30))
    elif target is None:
        target = risk_free_rate

    # --- Cluster assets ---
    corr = cov_to_corr(cov)
    labels = get_cluster_labels_from_map(
        returns_df=returns,
        cluster_method=cluster_method,
        corr=corr,
        cache_dir="optuna_cache",
        reoptimize=False,
    )
    unique_clusters = np.unique(labels)

    # --- Intra-cluster optimization ---
    intra_weights = pd.DataFrame(
        0.0, index=cov.index, columns=unique_clusters, dtype=float
    )
    intra_search_histories = manager.dict()  # Store candidates per cluster.
    intra_local_histories = manager.dict()  # Store local solver history per cluster.

    for cluster in unique_clusters:
        cluster_assets = cov.index[labels == cluster]
        cluster_cov = cov.loc[cluster_assets, cluster_assets]
        cluster_mu = mu.loc[cluster_assets] if mu is not None else None
        cluster_returns = returns[cluster_assets] if returns is not None else None

        intra_search_histories[cluster] = []
        intra_local_histories[cluster] = manager.list()
        combined_cb = partial(
            combined_intra_callback,
            cl=cluster,
            history_dict=intra_search_histories,
            local_history=intra_local_histories[cluster],
            overall_history=overall_history,
        )

        weights = optimize_weights_objective(
            cluster_cov,
            mu=cluster_mu,
            returns=cluster_returns,
            objective=objective,
            order=order,
            target=target,
            min_weight=min_weight,
            max_weight=max_weight,
            allow_short=allow_short,
            max_gross_exposure=max_gross_exposure,
            target_sum=target_sum,
            use_annealing=use_annealing,
            use_diffusion=use_diffusion,
            callback=combined_cb,
        )

        if np.isscalar(weights):
            weights = np.repeat(weights, len(cluster_assets))
        intra_weights.loc[cluster_assets, cluster] = weights

        logger.info(f"Intra-cluster weights:\n{intra_weights}")

    # --- Inter-cluster optimization ---
    valid_clusters = intra_weights.columns[intra_weights.sum(axis=0) > 1e-6]
    if len(valid_clusters) == 0:
        logger.error("No clusters with valid weights. Optimization aborted.")
        return pd.Series(dtype=float)

    reduced_cov = intra_weights[valid_clusters].T @ cov @ intra_weights[valid_clusters]
    reduced_mu = (intra_weights[valid_clusters].T @ mu) if mu is not None else None

    if returns is not None:
        reduced_returns = pd.DataFrame(
            {
                cluster: returns[intra_weights.index]
                .mul(intra_weights[cluster], axis=1)
                .sum(axis=1)
                for cluster in valid_clusters
            }
        )
    else:
        reduced_returns = None

    inter_search_history = manager.list()
    inter_cb = partial(
        unified_callback,
        history_list=inter_search_history,
        overall_history=overall_history,
    )

    inter_weights = pd.Series(
        optimize_weights_objective(
            reduced_cov,
            mu=reduced_mu,
            returns=reduced_returns,
            objective=objective,
            order=order,
            target=target,
            min_weight=min_weight,
            max_weight=max_weight,
            allow_short=allow_short,
            max_gross_exposure=max_gross_exposure,
            target_sum=target_sum,
            use_annealing=use_annealing,
            use_diffusion=use_diffusion,
            callback=inter_cb,
        ),
        index=valid_clusters,
    )

    logger.info(f"Inter-cluster optimized weights:\n{inter_weights}")

    # --- Combine intra- and inter-cluster weights to get final portfolio weights ---
    final_weights = intra_weights.mul(inter_weights, axis=1).sum(axis=1)

    if final_weights.abs().sum() < 1e-6:
        logger.error("Final weights sum to zero, optimization failed.")
        return pd.Series(dtype=float)

    final_weights /= final_weights.sum()

    if not isinstance(final_weights, pd.Series):
        final_weights = pd.Series(final_weights, index=intra_weights.index)

    logger.info(f"Combined cluster weights:\n{final_weights}")

    # --- Build Overall Search History for plotting ---
    # Instead of converting overall_history (which is inhomogeneous) to an array,
    # we use the inter_search_history candidates (which are homogeneous in dimension)
    # and transform them into the final portfolio space.
    reduced_intra = intra_weights[valid_clusters]
    overall_search_history = []
    for candidate in inter_search_history:
        candidate = np.array(candidate)
        # Pad candidate if needed so that its length equals the number of valid clusters
        if candidate.shape[0] != reduced_intra.shape[1]:
            candidate = np.pad(
                candidate, (0, reduced_intra.shape[1] - candidate.shape[0]), "constant"
            )
        # Combine with intra cluster weights to form a candidate for final weights.
        overall_candidate = (reduced_intra * candidate).sum(axis=1)
        overall_search_history.append(overall_candidate.values)
    overall_search_history = np.array(overall_search_history)
    if overall_search_history.size == 0:
        overall_search_history = final_weights.values.reshape(1, -1)

    # --- Plot Overall Global Optimization Contour ---
    if plot:
        try:
            if overall_search_history.size > 0:
                optimization_method_str = ""
                if use_annealing:
                    optimization_method_str = "via Dual Annealing"
                elif use_diffusion:
                    optimization_method_str = "via Stochastic Diffusion"
                print("search history", overall_search_history)
                print("\n final weights", final_weights.values)
                plot_global_optimization(
                    search_history=overall_search_history,
                    final_solution=final_weights.values,
                    objective_function=lambda w: sharpe_objective(
                        w, mu.to_numpy(), cov.to_numpy()
                    ),
                    grid_resolution=90,
                    title="Global Optimization Contour {}".format(
                        optimization_method_str
                    ),
                    flip_objective=True,
                ).show()
            else:
                logger.warning("No search history available for plotting.")
        except Exception as e:
            logger.warning("Plotting failed: " + str(e))

    # logger.info(f"Final optimized weights:\n{final_weights}")

    return final_weights
