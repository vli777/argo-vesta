import pandas as pd
import numpy as np
from typing import List, Union
from sklearn.covariance import LedoitWolf

from utils.logger import logger


def compute_lw_covariance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Ledoit-Wolf shrinkage covariance matrix for asset returns.

    Args:
        df (pd.DataFrame): Asset returns DataFrame (T x n), where T is time.

    Returns:
        pd.DataFrame: Ledoit-Wolf covariance matrix (n x n).
    """
    lw = LedoitWolf()
    covariance = lw.fit(df).covariance_
    return pd.DataFrame(covariance, index=df.columns, columns=df.columns)


def compute_lw_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a Ledoit-Wolf covariance matrix and converts it to a correlation matrix.

    Args:
        df (pd.DataFrame): Asset returns DataFrame (T x n), where T is time.

    Returns:
        pd.DataFrame: Ledoit-Wolf correlation matrix (n x n).
    """
    covariance_df = compute_lw_covariance(df)
    cov_array = covariance_df.values  # Convert to NumPy array
    stddev = np.sqrt(np.diag(cov_array))
    # Compute correlation array
    corr_array = cov_array / np.outer(stddev, stddev)
    # Fill diagonal with zeros
    np.fill_diagonal(corr_array, 0)
    return pd.DataFrame(
        corr_array, index=covariance_df.index, columns=covariance_df.columns
    )


def compute_correlation_matrix(
    df: pd.DataFrame, lw_threshold: int = 50, use_abs: bool = True
) -> pd.DataFrame:
    """
    Compute a correlation matrix for the given returns DataFrame.
    Uses the Ledoit-Wolf estimator if the number of columns exceeds lw_threshold.
    Otherwise, uses the standard Pearson correlation.

    Args:
        df (pd.DataFrame): DataFrame of returns with tickers as columns.
        lw_threshold (int): If number of columns > lw_threshold, use Ledoit-Wolf.
        use_abs (bool): If True, take the absolute value of correlations.

    Returns:
        pd.DataFrame: Correlation matrix with the diagonal filled with zeros.
    """
    if len(df.columns) > lw_threshold:
        # Use the Ledoit-Wolf based correlation matrix.
        corr_matrix = compute_lw_correlation(df)
    else:
        # Use standard correlation.
        corr_matrix = df.corr()
        if use_abs:
            corr_matrix = corr_matrix.abs()
        # Fill diagonal with zeros.
        corr_matrix = corr_matrix.copy()  # Ensure mutability
        np.fill_diagonal(corr_matrix.values, 0)
    return corr_matrix


def compute_distance_matrix(
    returns_df: pd.DataFrame, scale_distances: bool = False
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Compute a normalized distance matrix from the correlation matrix of returns.
    The conversion uses the transformation: distance = (1 - correlation) / 2.
    Optionally, if scale_distances is True, the distance matrix is re-scaled via
    min-max normalization to fully span the [0, 1] interval.
    Ensures the returned distance matrix matches the type of `compute_correlation_matrix()`.
    """
    # Compute the correlation matrix
    corr_matrix = compute_correlation_matrix(returns_df)

    # Convert correlations to distances in [0, 1]
    distance_matrix = (1 - corr_matrix) / 2

    # Scale distances if required
    if scale_distances:
        dmin, dmax = np.min(distance_matrix), np.max(distance_matrix)
        if not np.isclose(dmax, dmin):
            distance_matrix = (distance_matrix - dmin) / (dmax - dmin)
            logger.debug("Distance matrix re-scaled to full [0, 1] range.")

    # Preserve original return type (DataFrame or NumPy array)
    if isinstance(corr_matrix, pd.DataFrame):
        return pd.DataFrame(
            distance_matrix, index=corr_matrix.index, columns=corr_matrix.columns
        )
    return distance_matrix  # NumPy array if `compute_correlation_matrix()` returned one
