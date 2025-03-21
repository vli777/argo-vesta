import numpy as np
import pandas as pd
from typing import Optional, Union, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from functools import partial
from multiprocessing import Manager

from models.optimize_portfolio import (
    estimated_portfolio_volatility,
    optimize_weights_objective,
)
from models.optimization_plot import plot_global_optimization
from utils.logger import logger


def cov_to_corr(cov):
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # Numerical stability
    return corr


def default_intra_callback(x, f, *args, cl, history_dict):
    """
    A default callback that appends the candidate x to the history for cluster cl.
    """
    history_dict.setdefault(cl, []).append(x.copy())
    return False


def default_inter_callback(x, f, *args, history_list):
    history_list.append(x.copy())
    return False


def nested_clustered_optimization(
    cov: pd.DataFrame,
    mu: Optional[pd.Series] = None,
    returns: Optional[pd.DataFrame] = None,
    objective: str = "sharpe",
    max_clusters: int = 10,
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
) -> pd.Series:
    """
    Perform Nested Clustered Optimization with a flexible objective.
    For objectives requiring historica returns a 'returns' DataFrame must be provided.

    Args:
        cov (pd.DataFrame): Covariance matrix of asset returns.
        mu (Optional[pd.Series]): Expected returns.
        returns (Optional[pd.DataFrame]): Historical returns (time series) with assets as columns.
        objective (str): Optimization objective.
        max_clusters (int): Maximum number of clusters.
        max_weight (float): Maximum weight per asset.
        allow_short (bool): Allow short positions.
        max_gross_exposure (float): Maximum gross exposure when allowing short positions.
        target (float): Target threshold for Omega ratio tau.
        order (int): Order for downside risk metrics.
        target_sum (float): Sum of weights (default 1.0).
        risk_free_rate (float): Risk free rate (default 0.0).
        use_annealing (bool): Use dual annealing to search for global optima.
        use_diffusion (bool): Use stochastic diffusion to search for global optima.

    Returns:
        pd.Series: Final portfolio weights.
    """
    # Filter assets with enough historical data
    min_data_threshold = cov.shape[0] * 0.5
    valid_assets = cov.index[cov.notna().sum(axis=1) >= min_data_threshold]
    if len(valid_assets) < 2:
        logger.warning(
            "Not enough valid assets after filtering. Skipping optimization."
        )
        return pd.Series(dtype=float)
    cov = cov.loc[valid_assets, valid_assets]
    if mu is not None:
        mu = mu.reindex(valid_assets).fillna(0)
    if returns is not None:
        returns = returns[valid_assets]
    if target is None and returns is not None:
        target = max(risk_free_rate, np.percentile(returns.to_numpy().flatten(), 30))
    else:
        target = risk_free_rate

    manager = Manager()
    # --- Cluster assets ---
    corr = cov_to_corr(cov)
    labels = cluster_kmeans(corr, max_clusters)
    unique_clusters = np.unique(labels)

    # --- Intra-cluster optimization (per cluster) ---
    intra_search_histories = manager.dict()
    intra_weights = pd.DataFrame(
        0, index=cov.index, columns=unique_clusters, dtype=float
    )

    for cluster in unique_clusters:
        cluster_assets = cov.index[labels == cluster]
        cluster_cov = cov.loc[cluster_assets, cluster_assets]
        cluster_mu = mu.loc[cluster_assets] if mu is not None else None
        cluster_returns = returns[cluster_assets] if returns is not None else None

        intra_search_histories[cluster] = []
        intra_cb = partial(
            default_intra_callback, cl=cluster, history_dict=intra_search_histories
        )

        weights = optimize_weights_objective(
            cluster_cov,
            mu=cluster_mu,
            returns=cluster_returns,
            objective=objective,
            order=order,
            target=target,
            max_weight=max_weight,
            allow_short=allow_short,
            max_gross_exposure=max_gross_exposure,
            target_sum=target_sum,
            use_annealing=use_annealing,
            use_diffusion=use_diffusion,
            callback=intra_cb,
        )
        intra_weights.loc[cluster_assets, cluster] = weights

    # --- Inter-cluster (global) optimization ---
    reduced_cov = intra_weights.T @ cov @ intra_weights
    reduced_mu = (intra_weights.T @ mu) if mu is not None else None
    if returns is not None:
        reduced_returns = pd.DataFrame(
            {
                cluster: (
                    returns.loc[:, intra_weights.index]
                    .mul(intra_weights[cluster], axis=1)
                    .sum(axis=1)
                )
                for cluster in unique_clusters
            }
        )
    else:
        reduced_returns = None

    # Record inter-cluster search history.
    inter_search_history = manager.list()
    inter_cb = partial(default_inter_callback, history_list=inter_search_history)
    inter_weights = pd.Series(
        optimize_weights_objective(
            reduced_cov,
            mu=reduced_mu,
            returns=reduced_returns,
            objective=objective,
            order=order,
            target=target,
            max_weight=max_weight,
            allow_short=allow_short,
            max_gross_exposure=max_gross_exposure,
            target_sum=target_sum,
            use_annealing=use_annealing,
            use_diffusion=use_diffusion,
            callback=inter_cb,
        ),
        index=unique_clusters,
    )

    # --- Combine intra- and inter-cluster weights to get final portfolio weights ---
    final_weights = intra_weights.mul(inter_weights, axis=1).sum(axis=1)
    if not isinstance(final_weights, pd.Series):
        final_weights = pd.Series(final_weights, index=intra_weights.index)

    # --- Build Overall Search History ---
    overall_search_history = []
    # For each candidate inter-cluster weight vector, compute overall candidate.
    for candidate in inter_search_history:
        candidate = np.array(candidate)
        # Ensure candidate length equals number of clusters (which should match intra_weights' columns)
        if candidate.shape[0] != intra_weights.shape[1]:
            candidate = np.pad(
                candidate, (0, intra_weights.shape[1] - candidate.shape[0]), "constant"
            )
        overall_candidate = (intra_weights * candidate).sum(axis=1)
        overall_search_history.append(overall_candidate.values)
    overall_search_history = np.array(overall_search_history)
    # Ensure overall_search_history is 2D.
    if overall_search_history.size == 0:
        overall_search_history = final_weights.values.reshape(1, -1)

    # --- Define Overall Objective Function ---
    def overall_objective(w):
        # Ensure mu is aligned with w's dimension using final_weights index
        aligned_mu = mu.reindex(final_weights.index).fillna(0)
        port_return = w @ aligned_mu.to_numpy()
        port_vol = estimated_portfolio_volatility(w, cov.to_numpy())
        return -port_return / port_vol if port_vol > 0 else 1e6

    # --- Plot Overall Global Optimization Contour ---
    if plot:
        try:
            fig = plot_global_optimization(
                search_history=overall_search_history,
                final_solution=final_weights.values,
                objective_function=overall_objective,
                grid_resolution=120,
                title="Global Optimization Contour",
                flip_objective=True,
            )
            fig.show()
        except Exception as e:
            logger.warning("Plotting failed: " + str(e))

    return final_weights


def cluster_kmeans(corr: np.ndarray, max_clusters: int = 10) -> np.ndarray:
    """
    Cluster assets using KMeans on the correlation matrix.
    """
    # Transform correlation to a distance metric
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
