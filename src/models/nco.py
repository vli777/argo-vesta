import numpy as np
import pandas as pd
from typing import Optional, Union, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from models.optimize_portfolio import estimated_portfolio_volatility, optimize_weights_objective
from models.optimization_plot import plot_global_optimization
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
    max_gross_exposure: float = 1.3,
    target: Optional[float] = None,
    order: int = 3,
    target_sum: float = 1.0,
    risk_free_rate: float = 0.0,
    use_annealing: bool = False
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
        max_gross_exposure (float): Maximum gross exposure when allowing short positions.
        target (float): Target threshold for Omega ratio tau.
        order (int): Order for downside risk metrics.
        target_sum (float): Sum of weights (default 1.0).
        risk_free_rate (float): Risk free rate (default 0.0).
        use_annealing (bool): Use dual annealing to search for global optima.

    Returns:
        pd.Series: Final portfolio weights.
    """
    # Filter assets with enough historical data
    min_data_threshold = cov.shape[0] * 0.5  
    valid_assets = cov.index[cov.notna().sum(axis=1) >= min_data_threshold]
    if len(valid_assets) < 2:
        logger.warning("Not enough valid assets after filtering. Skipping optimization.")
        return pd.Series(dtype=float)
    cov = cov.loc[valid_assets, valid_assets]
    if mu is not None:
        mu = mu.loc[valid_assets]
    if returns is not None:
        returns = returns[valid_assets]
    if target is None and returns is not None:
        target = max(risk_free_rate, np.percentile(returns.to_numpy().flatten(), 30))
    else:
        target = risk_free_rate

    # --- Cluster assets ---
    corr = cov_to_corr(cov)
    labels = cluster_kmeans(corr, max_clusters)
    unique_clusters = np.unique(labels)

    # --- Intra-cluster optimization (per cluster) ---
    intra_search_histories = {} 
    intra_weights = pd.DataFrame(0, index=cov.index, columns=unique_clusters, dtype=float)
    
    for cluster in unique_clusters:
        cluster_assets = cov.index[labels == cluster]
        cluster_cov = cov.loc[cluster_assets, cluster_assets]
        cluster_mu = mu.loc[cluster_assets] if mu is not None else None
        cluster_returns = returns[cluster_assets] if returns is not None else None
        
        intra_search_histories[cluster] = []
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
            callback=lambda x, f, ctx, cl=cluster: intra_search_histories[cl].append(x.copy())
        )
        intra_weights.loc[cluster_assets, cluster] = weights

    # --- Inter-cluster (global) optimization ---
    reduced_cov = intra_weights.T @ cov @ intra_weights
    reduced_mu = (intra_weights.T @ mu) if mu is not None else None
    if returns is not None:
        reduced_returns = pd.DataFrame({
            cluster: (returns.loc[:, intra_weights.index]
                      .mul(intra_weights[cluster], axis=1)
                      .sum(axis=1))
            for cluster in unique_clusters
        })
    else:
        reduced_returns = None

    # Here we record search history only for the global (inter-cluster) stage.
    inter_search_history = []  # to record candidate vectors during annealing
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
            callback=lambda x, f, ctx: inter_search_history.append(x.copy())
        ),
        index=unique_clusters,
    )

    # --- Combine intra- and inter-cluster weights to get final portfolio weights ---
    final_weights = intra_weights.mul(inter_weights, axis=1).sum(axis=1)
    final_weights = final_weights[final_weights.abs() >= 0.01]
    if not isinstance(final_weights, pd.Series):
        final_weights = pd.Series(final_weights, index=intra_weights.index)

    overall_search_history = []
    for candidate in inter_search_history:
        # candidate is an array of length equal to number of clusters.
        # Compute overall candidate: for each asset, its weight is:
        #   overall_weight = sum_{cluster in unique_clusters} (intra_weights[asset, cluster] * candidate[cluster])
        overall_candidate = (intra_weights * candidate).sum(axis=1)
        overall_search_history.append(overall_candidate.values)
    overall_search_history = np.array(overall_search_history)

    # --- Define Overall Objective Function ---
    def overall_objective(w):
        # Compute portfolio Sharpe (here negative Sharpe, which we'll flip)
        port_return = w @ mu
        port_vol = estimated_portfolio_volatility(w, cov.to_numpy())
        return -port_return / port_vol if port_vol > 0 else 1e6

    # --- Plot Overall Global Optimization Contour ---
    # Here we use our Plotly function to plot the 3D contour of the overall objective.
    fig = plot_global_optimization(
        search_history=overall_search_history,
        final_solution=final_weights.values,
        objective_function=overall_objective,
        grid_resolution=50,
        title="Overall Max Sharpe Contour for Portfolio",
        flip_objective=True  # flip so that maximum Sharpe appears as a peak
    )
    fig.show()

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
