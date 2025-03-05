import numpy as np
import optuna
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from scipy.spatial.distance import pdist, squareform


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


def compute_silhouette(data, labels, metric="euclidean"):
    """Compute the silhouette score."""
    return silhouette_score(data, labels, metric=metric)


def compute_calinski_harabasz(data, labels):
    """Compute the Calinski-Harabasz score."""
    return calinski_harabasz_score(data, labels)


def compute_davies_bouldin(data, labels):
    """Compute the Davies-Bouldin score."""
    return davies_bouldin_score(data, labels)


def compute_dunn_index(data, labels):
    """
    Compute the Dunn Index.
    Dunn Index = (minimum inter-cluster distance) / (maximum intra-cluster distance).
    """
    distances = squareform(pdist(data))
    clusters = np.unique(labels)

    # Compute minimum inter-cluster distance.
    inter_cluster_distances = []
    for i in clusters:
        for j in clusters:
            if i < j:
                idx_i = np.where(labels == i)[0]
                idx_j = np.where(labels == j)[0]
                if len(idx_i) > 0 and len(idx_j) > 0:
                    inter_cluster_distances.append(
                        np.min(distances[np.ix_(idx_i, idx_j)])
                    )
    min_inter = np.min(inter_cluster_distances) if inter_cluster_distances else 0

    # Compute maximum intra-cluster distance.
    intra_cluster_distances = []
    for cluster in clusters:
        idx = np.where(labels == cluster)[0]
        if len(idx) > 1:
            intra_cluster_distances.append(np.max(distances[np.ix_(idx, idx)]))
    max_intra = np.max(intra_cluster_distances) if intra_cluster_distances else 0

    if max_intra == 0:
        return np.inf
    return min_inter / max_intra


def compute_composite_score(data, labels):
    """
    Compute a composite clustering score by combining four indices:
      - Silhouette Score (normalized from [-1, 1] to [0, 1])
      - Calinski-Harabasz Score (normalized as ch/(ch+1))
      - Davies-Bouldin Score (lower is better; normalized as 1/(1+db))
      - Dunn Index (normalized as dunn/(1+dunn))

    The composite score is the average of these normalized values.
    """
    try:
        sil = compute_silhouette(data, labels)
    except Exception as e:
        logger.debug(f"Silhouette computation failed: {e}")
        sil = -1  # Worst-case

    try:
        ch = compute_calinski_harabasz(data, labels)
    except Exception as e:
        logger.debug(f"Calinski-Harabasz computation failed: {e}")
        ch = 0

    try:
        db = compute_davies_bouldin(data, labels)
    except Exception as e:
        logger.debug(f"Davies-Bouldin computation failed: {e}")
        db = np.inf

    try:
        dunn = compute_dunn_index(data, labels)
    except Exception as e:
        logger.debug(f"Dunn index computation failed: {e}")
        dunn = 0

    # Normalize each metric:
    norm_sil = (sil + 1) / 2  # Maps silhouette from [-1, 1] to [0, 1].
    norm_ch = ch / (ch + 1)  # A simple normalization for Calinski-Harabasz.
    norm_db = 1 / (1 + db)  # Lower Davies-Bouldin is better.
    norm_dunn = dunn / (1 + dunn)  # Normalize Dunn index.

    composite = (norm_sil + norm_ch + norm_db + norm_dunn) / 4
    return composite


def objective_spectral_clustering(trial, returns_df: pd.DataFrame) -> float:
    """
    Objective function for tuning spectral clustering using a composite score.
    It tunes the gamma parameter for the RBF affinity and the number of clusters,
    where n_clusters ranges from 2 to ceil(sqrt(n_assets)).
    """
    # Tune the gamma parameter.
    gamma = trial.suggest_float("gamma", 0.1, 2.0, step=0.1)
    n_assets = returns_df.shape[1]
    max_clusters = int(np.ceil(np.sqrt(n_assets)))
    n_clusters = trial.suggest_int("n_clusters", 2, max_clusters)

    # Compute the affinity matrix using your custom RBF function.
    affinity = compute_affinity_matrix_rbf(returns_df, gamma)

    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )
    cluster_labels = spectral.fit_predict(affinity)

    if len(np.unique(cluster_labels)) < 2:
        logger.debug("Trial rejected: less than 2 clusters.")
        return -1.0

    # For evaluation, treat each asset (column in returns_df) as a sample.
    data = returns_df.T.values
    composite = compute_composite_score(data, cluster_labels)

    logger.debug(
        f"Trial params: gamma={gamma}, n_clusters={n_clusters} | Composite Score: {composite:.4f}"
    )
    return composite


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
