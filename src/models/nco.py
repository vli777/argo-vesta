import numpy as np
import pandas as pd
from typing import Optional, Union, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from models.optimize_portfolio import optimize_weights_objective
from utils.logger import logger


def cov_to_corr(cov):
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # Numerical stability
    return corr


def nested_clustered_optimization(
    cov: pd.DataFrame,
    mu: Optional[pd.Series] = None,
    returns: Optional[pd.DataFrame] = None,
    objective: str = "sharpe",
    max_clusters: int = 10,
    max_weight: float = 1.0,
    allow_short: bool = False,
    target: Optional[float] = None,
    order: int = 3,
    target_sum: float = 1.0,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    Perform Nested Clustered Optimization with a flexible objective.
    For objectives requiring historical returns a 'returns' DataFrame must be provided.

    Args:
        cov (pd.DataFrame): Covariance matrix of asset returns.
        mu (Optional[pd.Series]): Expected returns.
        returns (Optional[pd.DataFrame]): Historical returns (time series) with assets as columns.
        objective (str): Optimization objective.
        max_clusters (int): Maximum number of clusters.
        max_weight (float): Maximum weight per asset.
        allow_short (bool): Allow short positions.
        target (float): Target threshold for Omega ratio tau.
        order (int): Order for downside risk metrics.
        target_sum (float): Sum of weights (default 1.0).
        risk_free_rate (float): Risk free rate (default 0.0).

    Returns:
        pd.Series: Final portfolio weights.
    """
    # Filter assets with enough historical data
    min_data_threshold = cov.shape[0] * 0.5  # At least 50% valid history
    valid_assets = cov.index[cov.notna().sum(axis=1) >= min_data_threshold]

    if len(valid_assets) < 2:
        logger.warning(
            "Not enough valid assets after filtering. Skipping optimization."
        )
        return pd.Series(dtype=float)  # Return an empty portfolio

    cov = cov.loc[valid_assets, valid_assets]
    if mu is not None:
        mu = mu.loc[valid_assets]
    if returns is not None:
        returns = returns[valid_assets]

    if target is None and returns is not None:
        target = max(risk_free_rate, np.percentile(returns.to_numpy().flatten(), 30))
    else:
        target = risk_free_rate

    # Create the correlation matrix and cluster the assets
    corr = cov_to_corr(cov)
    labels = cluster_kmeans(corr, max_clusters)
    unique_clusters = np.unique(labels)

    # Intra-cluster optimization: optimize weights for assets within each cluster
    intra_weights = pd.DataFrame(
        0, index=cov.index, columns=unique_clusters, dtype=float
    )
    for cluster in unique_clusters:
        cluster_assets = cov.index[labels == cluster]
        cluster_cov = cov.loc[cluster_assets, cluster_assets]
        cluster_mu = None if mu is None else mu.loc[cluster_assets]
        cluster_returns = None if returns is None else returns[cluster_assets]

        weights = optimize_weights_objective(
            cluster_cov,
            mu=cluster_mu,
            returns=cluster_returns,
            objective=objective,
            order=order,
            target=target,
            max_weight=max_weight,
            allow_short=allow_short,
            target_sum=target_sum,
        )
        intra_weights.loc[cluster_assets, cluster] = weights

    # Inter-cluster optimization: optimize how to weight each cluster
    reduced_cov = intra_weights.T @ cov @ intra_weights
    reduced_mu = None if mu is None else intra_weights.T @ mu

    # For historical returns aggregation, build a DataFrame where each column is the cluster's time series.
    reduced_returns = None
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
            target_sum=target_sum,
        ),
        index=unique_clusters,
    )

    # Combine intra- and inter-cluster weights to get the final portfolio allocation
    final_weights = intra_weights.mul(inter_weights, axis=1).sum(axis=1)
    final_weights = final_weights[
        final_weights.abs() >= 0.01
    ]  # Apply a minimum weight threshold

    if not isinstance(final_weights, pd.Series):
        final_weights = pd.Series(final_weights, index=intra_weights.index)

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
