import numpy as np
import pandas as pd
from typing import Optional, Union, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from functools import partial
from multiprocessing import Manager

from models.optimize_portfolio import (
    optimize_weights_objective,
)
from models.optimization_plot import plot_global_optimization
from models.scipy_objective_models import sharpe_objective
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
    max_clusters: int = 50,
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

    Returns:
        pd.Series: Final portfolio weights.
    """
    manager = Manager()

    # Filter assets with sufficient historical data
    if returns is not None:
        valid_assets = returns.columns[
            returns.notna().sum(axis=0) >= (returns.shape[0] * 0.5)
        ]
        returns = returns[valid_assets]
        cov = cov.loc[valid_assets, valid_assets]
        if mu is not None:
            mu = mu.reindex(valid_assets).fillna(0)
    else:
        valid_assets = cov.index

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
    labels = cluster_kmeans(corr, max_clusters)
    unique_clusters = np.unique(labels)

    # --- Intra-cluster optimization ---
    intra_weights = pd.DataFrame(
        0.0, index=cov.index, columns=unique_clusters, dtype=float
    )
    intra_search_histories = manager.dict()

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
            min_weight=min_weight,
            max_weight=max_weight,
            allow_short=allow_short,
            max_gross_exposure=max_gross_exposure,
            target_sum=target_sum,
            use_annealing=use_annealing,
            use_diffusion=use_diffusion,
            callback=intra_cb,
        )

        if np.isscalar(weights):
            weights = np.repeat(weights, len(cluster_assets))
        elif hasattr(weights, "__len__"):
            if len(weights) == 1 and len(cluster_assets) > 1:
                weights = np.repeat(weights, len(cluster_assets))
            elif len(weights) != len(cluster_assets):
                raise ValueError(
                    f"Mismatch in cluster {cluster}: expected {len(cluster_assets)} weights, got {len(weights)}."
                )
        else:
            raise ValueError(f"Unexpected weights type in cluster {cluster}.")

        intra_weights.loc[cluster_assets, cluster] = weights

        # logger.info(
        #     f"Intra-cluster weights before global optimization:\n{intra_weights}"
        # )

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

    # logger.info(f"Reduced covariance matrix:\n{reduced_cov}")
    # logger.info(f"Reduced expected returns (mu):\n{reduced_mu}")

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

    # logger.info(f"Inter-cluster optimized weights:\n{inter_weights}")

    # --- Combine intra- and inter-cluster weights to get final portfolio weights ---
    final_weights = intra_weights.mul(inter_weights, axis=1).sum(axis=1)

    if final_weights.abs().sum() < 1e-6:
        logger.error("Final weights sum to zero, optimization failed.")
        return pd.Series(dtype=float)

    final_weights /= final_weights.sum()

    if not isinstance(final_weights, pd.Series):
        final_weights = pd.Series(final_weights, index=intra_weights.index)

    # logger.info(f"Combined cluster weights:\n{final_weights}")

    # --- Build Overall Search History ---
    # Use only the valid clusters from intra_weights
    reduced_intra = intra_weights[valid_clusters]
    overall_search_history = []
    # For each candidate inter-cluster weight vector, compute overall candidate.
    for candidate in inter_search_history:
        candidate = np.array(candidate)
        # Pad candidate if needed so that its length equals number of valid clusters
        if candidate.shape[0] != reduced_intra.shape[1]:
            candidate = np.pad(
                candidate, (0, reduced_intra.shape[1] - candidate.shape[0]), "constant"
            )
        overall_candidate = (reduced_intra * candidate).sum(axis=1)
        overall_search_history.append(overall_candidate.values)
    overall_search_history = np.array(overall_search_history)
    # Ensure overall_search_history is 2D.
    if overall_search_history.size == 0:
        overall_search_history = final_weights.values.reshape(1, -1)

    # --- Plot Overall Global Optimization Contour ---
    if plot:
        overall_search_history = np.array(
            [
                (
                    intra_weights
                    * np.pad(
                        np.array(candidate),
                        (0, intra_weights.shape[1] - np.array(candidate).shape[0]),
                        mode="constant",
                    )
                )
                .sum(axis=1)
                .values
                for candidate in inter_search_history
            ]
        )
        try:
            if overall_search_history.size > 0:
                plot_global_optimization(
                    search_history=overall_search_history,
                    final_solution=final_weights.values,
                    objective_function=lambda w: sharpe_objective(
                        w, mu.to_numpy(), cov.to_numpy()
                    ),
                    grid_resolution=120,
                    title="Global Optimization Contour via {}".format(
                        "Dual Annealing" if use_annealing else "Stochastic Diffusion"
                    ),
                    flip_objective=True,
                ).show()
            else:
                logger.warning("No search history available for plotting.")
        except Exception as e:
            logger.warning("Plotting failed: " + str(e))

    # logger.info(f"Final optimized weights:\n{final_weights}")

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
