import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime
from types import SimpleNamespace
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from correlation.correlation_utils import compute_correlation_matrix
from utils.portfolio_utils import normalize_weights


def format_asset_cluster_map(
    asset_cluster_map: Dict[str, int], global_cache: Dict[str, Dict]
) -> Dict[str, List[str]]:
    """
    Formats the asset_cluster_map so that each noise ticker (-1) is treated as its own cluster.
    Also incorporates cluster information from global_cache to ensure consistency.

    Args:
        asset_cluster_map (dict): A dictionary mapping tickers to cluster IDs.
        global_cache (dict): Cached parameters, including cluster assignments.

    Returns:
        dict: A formatted dictionary where each noise ticker is its own cluster.
    """
    formatted_clusters = {}

    for ticker, cluster_id in asset_cluster_map.items():
        # If cluster info exists in global_cache, use it
        if ticker in global_cache and "cluster" in global_cache[ticker]:
            cluster_id = global_cache[ticker]["cluster"]

        # Ensure noise tickers (-1) are stored as their own clusters
        if cluster_id == -1 or cluster_id == np.int64(-1):
            formatted_clusters[ticker] = [ticker]
        else:
            formatted_clusters.setdefault(cluster_id, []).append(ticker)

    return formatted_clusters


def is_cache_stale(reversion_cache_file: str, max_age_days: int = 30) -> bool:
    """Check if the cache is stale based on the last update timestamp."""
    # Check if the cache file exists and determine if it is stale based on its last modified time.
    if os.path.exists(reversion_cache_file):
        cache_modified_time = datetime.fromtimestamp(
            os.path.getmtime(reversion_cache_file)
        )
        cache_age_days = (datetime.now() - cache_modified_time).days
        return cache_age_days >= max_age_days
    else:
        return True


def calculate_continuous_composite_signal(signals: dict, ticker_params: dict) -> dict:
    composite = {}
    for ticker, sig_data in signals.items():
        params = ticker_params.get(ticker, {})
        mode = params.get("mode", "fallback")
        if mode == "cointegration":
            # For cointegration, assume only daily signal is used.
            latest_val = 0
            daily_signal = sig_data.get("daily", None)
            if daily_signal is not None:
                if isinstance(daily_signal, pd.Series) and not daily_signal.empty:
                    latest_val = daily_signal.iloc[-1]
                elif isinstance(daily_signal, dict) and daily_signal:
                    latest_val = daily_signal[max(daily_signal.keys())]
            composite[ticker] = latest_val
        else:
            # For fallback, blend daily and weekly signals.
            wd = params.get("weight_daily", 0.7)
            wd = max(0.0, min(wd, 1.0))
            ww = 1.0 - wd
            daily_val = 0
            daily_signal = sig_data.get("daily", None)
            if daily_signal is not None:
                if isinstance(daily_signal, pd.Series) and not daily_signal.empty:
                    daily_val = daily_signal.iloc[-1]
                elif isinstance(daily_signal, dict) and daily_signal:
                    daily_val = daily_signal[max(daily_signal.keys())]

            weekly_val = 0
            weekly_signal = sig_data.get("weekly", None)
            if weekly_signal is not None:
                if isinstance(weekly_signal, pd.Series) and not weekly_signal.empty:
                    weekly_val = weekly_signal.iloc[-1]
                elif isinstance(weekly_signal, dict) and weekly_signal:
                    weekly_val = weekly_signal[max(weekly_signal.keys())]

            composite[ticker] = wd * daily_val + ww * weekly_val

    return composite


def group_ticker_params_by_cluster(ticker_params: dict) -> dict:
    """
    Convert a global cache keyed by ticker into a dictionary keyed by cluster id.
    Each value is a dictionary with keys:
      - "tickers": a list of tickers in that cluster
      - "params": the parameters for that cluster (assumed to be the same for all tickers in the cluster)
    """
    group_parameters = {}
    for ticker, params in ticker_params.items():
        cluster = params.get("cluster", "Unknown")
        if cluster not in group_parameters:
            group_parameters[cluster] = {"tickers": [], "params": params}
        group_parameters[cluster]["tickers"].append(ticker)
    return group_parameters


def propagate_signals_by_similarity(
    composite_signals: dict,
    group_mapping: dict,
    returns_df: pd.DataFrame,
    signal_strength: float = 0.5,
    lw_threshold: int = 50,
) -> dict:
    """
    Propagate composite signals within clusters by blending each ticker's own signal
    with a weighted average of the signals from the other tickers in the cluster. The
    weights come from the positive correlations (similarity) between tickers.

    This prevents a multiplicative effect when multiple source signals exist in the
    same cluster by normalizing the contribution.

    Args:
        composite_signals (dict): Original composite signals (ticker -> signal).
        group_mapping (dict): Mapping of cluster IDs to group info, which includes 'tickers'.
        returns_df (pd.DataFrame): Returns DataFrame (dates as index, tickers as columns).
        lw_threshold (int): Size threshold for using Ledoit Wolf vs Pearson correlation.
        signal_strength (float): Factor when propagating original source signals.

    Returns:
        dict: Updated composite signals with propagated, normalized values.
    """
    updated_signals = composite_signals.copy()

    for cluster_id, group_data in group_mapping.items():
        tickers_in_group = group_data.get("tickers", [])
        group_params = group_data.get("params", {})
        # Only keep tickers that exist in the returns data.
        available_tickers = [
            ticker for ticker in tickers_in_group if ticker in returns_df.columns
        ]
        if not available_tickers:
            continue

        # Extract group parameters.
        # Expecting keys: "window_daily", "window_weekly", "weight_daily"
        wd = group_params.get("weight_daily", 0.7)
        ww = 1.0 - wd
        window_daily = group_params.get("window_daily", 20)  # default if missing
        window_weekly = group_params.get("window_weekly", 5)  # in weeks

        # Convert weekly window to days (assuming 5 trading days per week)
        window_weekly_days = window_weekly * 5

        # Compute effective window as a weighted combination.
        effective_window = wd * window_daily + ww * window_weekly_days
        effective_window = int(round(effective_window))

        # Convert log returns to simple returns before computing rolling mean
        simple_returns = np.exp(returns_df[available_tickers]) - 1
        cluster_returns = simple_returns.rolling(window=effective_window).mean()

        # Drop NaNs before computing covariance
        cluster_returns = cluster_returns.dropna()

        rolling_corr = compute_correlation_matrix(
            cluster_returns, lw_threshold=lw_threshold, use_abs=True
        )

        # For each ticker, compute the weighted average of the signals from other tickers.
        for ticker in available_tickers:
            original_signal = composite_signals.get(ticker, 0)
            weighted_sum = 0.0
            sum_similarity = 0.0

            for source_ticker in available_tickers:
                if source_ticker == ticker:
                    continue

                source_signal = composite_signals.get(source_ticker, 0)
                similarity = (
                    rolling_corr.at[source_ticker, ticker]
                    if source_ticker in rolling_corr.index
                    else 0
                )

                # Use positively correlated pairs only
                if similarity > 0:
                    weighted_sum += source_signal * similarity
                    sum_similarity += similarity

            # Normalize the propagated signal by the total similarity weight.
            propagated_signal = (
                (weighted_sum / sum_similarity) if sum_similarity > 0 else 0
            )

            # Combine the original and the normalized propagated signal.
            updated_signals[ticker] = (
                1 - signal_strength
            ) * original_signal + signal_strength * propagated_signal

    return updated_signals


def adjust_allocation_with_mean_reversion(
    baseline_allocation: pd.Series,
    composite_signals: dict,
    alpha: float = 0.2,
    allow_short: bool = False,
    returns_df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Adjust the baseline allocation using a continuous mean reversion signal.
    The adjustment is multiplicative:
         new_weight = baseline_weight * (1 + adaptive_alpha * (composite_signal - 1))
    so that if composite_signal == 1, the allocation remains unchanged.
    Adaptive alpha scales based on volatility if returns_df is provided.

    Args:
        baseline_allocation (pd.Series): Series with index = ticker and values = baseline weights.
        composite_signals (dict): Mapping from ticker to continuous signal (adjustment factor) with a baseline of 1.
        alpha (float): Base sensitivity factor.
        allow_short (bool): If False, negative adjusted weights are set to zero.
        returns_df (pd.DataFrame, optional): Historical returns used to adaptively adjust alpha by realized volatility.

    Returns:
        pd.Series: Adjusted and normalized allocation.
    """
    composite_signals = pd.Series(composite_signals)
    if isinstance(baseline_allocation, dict):
        baseline_allocation = pd.Series(baseline_allocation).astype(float)

    composite_signals = composite_signals.reindex(
        baseline_allocation.index, fill_value=1
    )

    if returns_df is not None:
        realized_vol = (
            returns_df.rolling(window=30, min_periods=5).std().mean(axis=1).iloc[-1]
        )
        realized_vol = max(realized_vol, 1e-6)
        adaptive_alpha = alpha / (1 + realized_vol)
    else:
        adaptive_alpha = alpha

    adjusted = baseline_allocation * (1 + adaptive_alpha * (composite_signals - 1))

    if not allow_short:
        adjusted = adjusted.clip(lower=0)

    normalized_adjusted = normalize_weights(adjusted)
    return normalized_adjusted


def johansen_test(prices_df, det_order=0, k_ar_diff=1):
    """
    Runs the Johansen cointegration test and returns a simple namespace with:
      - cointegration_found (bool): True if any test statistic exceeds its critical value at 5%
      - eigenvector: The cointegrating vector corresponding to the first (largest) test statistic.
    """
    result = coint_johansen(prices_df, det_order, k_ar_diff)
    # Use the trace statistic and its critical values at the 5% level.
    cointegration_found = any(result.lr1 > result.cvt[:, 1])
    eigenvector = result.evec[:, 0] if cointegration_found else None
    return SimpleNamespace(
        cointegration_found=cointegration_found, eigenvector=eigenvector
    )


def compute_spread(prices_df: pd.DataFrame, eigenvector: np.ndarray) -> pd.Series:
    """
    Compute the cointegrated spread from asset prices using the cointegrating vector.

    The spread is calculated as a weighted sum of the asset prices, where the weights
    are derived from the cointegrating eigenvector. In cases where the sum of the eigenvector
    is close to zero, the eigenvector is normalized by its Euclidean norm instead.

    Args:
        prices_df (pd.DataFrame): DataFrame with dates as index and asset prices as columns.
        eigenvector (np.ndarray): Cointegrating vector from the Johansen test.

    Returns:
        pd.Series: Time series representing the computed spread.
    """
    # Normalize the eigenvector. If the sum is near zero, use the Euclidean norm.
    if np.isclose(eigenvector.sum(), 0):
        weights = eigenvector / np.linalg.norm(eigenvector)
    else:
        weights = eigenvector / eigenvector.sum()

    # Compute the spread as the weighted sum of prices.
    spread = prices_df.dot(weights)
    return spread
