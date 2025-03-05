import numpy as np
import optuna
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from utils import logger


def compute_affinity_matrix_rbf(returns_df: pd.DataFrame, gamma: float) -> np.ndarray:
    """
    Compute the correlation matrix from asset returns and convert it into an affinity matrix using an RBF kernel.
    This transformation maps the correlation space into [0, 1] and penalizes negative correlations more strongly.

    The transformation is:
        distance = 1 - correlation
        affinity = exp(-gamma * distance)

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and assets as columns.

    Returns:
        np.ndarray: The affinity matrix.
    """
    corr_matrix = returns_df.corr().values
    # Map correlation from [-1, 1] to [0, 1]
    distance = 1 - corr_matrix  # distance in [0, 2]
    affinity = np.exp(-gamma * distance)
    return affinity


def objective_spectral_clustering(
    trial: optuna.Trial, returns_df: pd.DataFrame
) -> float:
    """
    Objective function for tuning spectral clustering.
    It tunes the gamma parameter for the RBF affinity transformation and the number of clusters (n_clusters)
    to maximize the silhouette score.
    """
    # Tune the gamma parameter for the RBF kernel.
    gamma = trial.suggest_float("gamma", 0.1, 2.0, step=0.1)
    n_assets = returns_df.shape[1]
    # Tune n_clusters between 2 and a maximum (e.g., min(n_assets, 10)).
    n_clusters = trial.suggest_int("n_clusters", 2, min(n_assets, 10))

    # Compute the affinity matrix with the tuned gamma.
    affinity = compute_affinity_matrix_rbf(returns_df, gamma)

    # Run spectral clustering.
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )
    cluster_labels = spectral.fit_predict(affinity)

    # For silhouette_score we need a distance matrix. Here we define it as:
    # distance = 1 - affinity.
    distance_for_score = 1 - affinity

    # If spectral clustering ends up with only one cluster, reject this trial.
    if len(np.unique(cluster_labels)) < 2:
        logger.debug("Trial rejected: less than 2 clusters.")
        return -1.0

    try:
        sil_score = silhouette_score(
            distance_for_score, cluster_labels, metric="precomputed"
        )
    except Exception as e:
        logger.debug(f"Silhouette score computation failed: {e}")
        return -1.0

    logger.debug(
        f"Trial params: gamma={gamma}, n_clusters={n_clusters} | Silhouette: {sil_score:.4f}"
    )
    return sil_score


def run_spectral_affinity_study(returns_df: pd.DataFrame, n_trials: int = 50) -> dict:
    """
    Run an Optuna study to optimize the gamma and n_clusters parameters for spectral clustering.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_spectral_clustering(trial, returns_df),
        n_trials=n_trials,
    )
    best_params = study.best_trial.params
    logger.info(
        f"Best spectral parameters: {best_params} with silhouette score {study.best_value:.4f}"
    )
    return best_params
