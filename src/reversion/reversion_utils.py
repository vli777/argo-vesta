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


def calculate_continuous_composite_signal(
    signals: Dict[str, Union[pd.Series, dict]],
    ticker_params: dict,
    expected_index: Optional[pd.Index] = None,
) -> Dict[str, pd.Series]:
    """
    Build a composite signal for each ticker by combining daily and, if applicable, weekly signals.

    For tickers in 'cointegration' mode, the full daily signal series is returned.
    In 'fallback' mode, the daily and weekly signals are blended based on provided weights.
    If a signal is missing or empty, a neutral signal (constant 1.0) is used instead.

    Args:
        signals (dict): Mapping from ticker to signal data (daily and/or weekly); the values may be pd.Series or dict.
        ticker_params (dict): Mapping from ticker to parameters, including 'mode', 'weight_daily', etc.
        expected_index (Optional[pd.Index]): An optional pd.Index that represents the expected date range.
            If provided and a ticker's signal is missing or empty, a neutral series over this index is returned.

    Returns:
        dict: Mapping from ticker to composite signal as a full pd.Series.
    """
    composite = {}
    for ticker, sig_data in signals.items():
        params = ticker_params.get(ticker, {})
        mode = params.get("mode", "fallback")

        if mode == "cointegration":
            daily_signal = sig_data.get("daily", None)
            # If daily_signal comes in as a dict, convert it.
            if isinstance(daily_signal, dict):
                daily_signal = pd.Series(daily_signal)
            # If it's a non-empty pd.Series, use it; otherwise, use a neutral series.
            if isinstance(daily_signal, pd.Series) and not daily_signal.empty:
                composite[ticker] = daily_signal.copy()
            else:
                # Create a neutral signal. If expected_index is provided, use it;
                # otherwise, create a single-value series.
                if expected_index is not None:
                    composite[ticker] = pd.Series(1.0, index=expected_index)
                else:
                    composite[ticker] = pd.Series([1.0])
        else:
            # Fallback mode: blend daily and weekly signals.
            weight_daily = float(params.get("weight_daily", 0.7))
            weight_daily = np.clip(weight_daily, 0.0, 1.0)
            weight_weekly = 1.0 - weight_daily

            daily_signal = sig_data.get("daily", pd.Series(dtype=float))
            weekly_signal = sig_data.get("weekly", pd.Series(dtype=float))

            # Convert dict to Series if needed.
            if isinstance(daily_signal, dict):
                daily_signal = pd.Series(daily_signal)
            if isinstance(weekly_signal, dict):
                weekly_signal = pd.Series(weekly_signal)

            # Ensure they are pd.Series.
            if not isinstance(daily_signal, pd.Series):
                daily_signal = pd.Series(dtype=float)
            if not isinstance(weekly_signal, pd.Series):
                weekly_signal = pd.Series(dtype=float)

            # Align the two series; fill missing values with a neutral value of 1.0.
            aligned_daily, aligned_weekly = daily_signal.align(
                weekly_signal, fill_value=1.0
            )

            # If the aligned series are empty and expected_index is provided, fill with neutral values.
            if aligned_daily.empty and expected_index is not None:
                aligned_daily = pd.Series(1.0, index=expected_index)
            if aligned_weekly.empty and expected_index is not None:
                aligned_weekly = pd.Series(1.0, index=expected_index)

            composite[ticker] = (
                weight_daily * aligned_daily + weight_weekly * aligned_weekly
            )

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


def adjust_allocation_series_with_mean_reversion(
    baseline_allocation: Union[pd.Series, dict],
    composite_signals: Dict[str, pd.Series],
    alpha: float = 0.2,
    allow_short: bool = False,
    returns_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Adjust the baseline allocation over time using the full composite signal series.

    For each date, the allocation for each ticker is adjusted as:
        new_weight = baseline_weight * (1 + adaptive_alpha * (signal - 1))
    where adaptive_alpha may be scaled using a global realized volatility estimate from returns_df.

    If the adjustment for a given date results in a zero total weight, then the function falls back to
    the baseline allocation. If the baseline allocation itself is empty, an empty DataFrame is returned.

    Args:
        baseline_allocation (pd.Series or dict): Baseline weights for each ticker.
        composite_signals (dict): Mapping from ticker to full composite signal pd.Series.
        alpha (float): Base sensitivity factor.
        allow_short (bool): If False, negative allocations are set to zero.
        returns_df (pd.DataFrame, optional): Historical returns for adaptive alpha scaling.

    Returns:
        pd.DataFrame: Time series DataFrame where each row contains allocation weights for a given date.
    """
    # Ensure baseline_allocation is a pd.Series.
    if isinstance(baseline_allocation, dict):
        baseline_allocation = pd.Series(baseline_allocation, dtype=float)

    # Check that baseline_allocation is not empty.
    if baseline_allocation.empty:
        # Return an empty DataFrame since there's nothing to allocate.
        return pd.DataFrame()

    tickers = baseline_allocation.index.tolist()
    composite_df = pd.DataFrame(composite_signals).reindex(columns=tickers).fillna(1.0)

    # Safeguard: if composite_df is empty, create a default index.
    if composite_df.empty:
        if returns_df is not None and not returns_df.empty:
            composite_df = pd.DataFrame(
                {ticker: 1.0 for ticker in tickers}, index=returns_df.index
            )
        else:
            composite_df = pd.DataFrame(
                {ticker: [1.0] for ticker in tickers}, index=[pd.Timestamp.today()]
            )

    # Determine adaptive alpha based on global realized volatility, if available.
    if returns_df is not None and not returns_df.empty:
        realized_vol = (
            returns_df.rolling(window=30, min_periods=5).std().mean(axis=1).iloc[-1]
        )
        realized_vol = max(realized_vol, 1e-6)
        adaptive_alpha = alpha / (1 + realized_vol)
    else:
        adaptive_alpha = alpha

    allocation_over_time = []
    for date, signal_row in composite_df.iterrows():
        raw_adjustment = baseline_allocation * (1 + adaptive_alpha * (signal_row - 1))
        if not allow_short:
            raw_adjustment = raw_adjustment.clip(lower=0)

        # If the computed adjustment sums to zero, fall back to the baseline allocation.
        if np.isclose(raw_adjustment.sum(), 0.0):
            # If baseline_allocation is also zero-sum, return equal weights.
            if (
                np.isclose(baseline_allocation.sum(), 0.0)
                or len(baseline_allocation) == 0
            ):
                # This branch should not occur because of the earlier check, but we safeguard.
                fallback = pd.Series(dtype=float)
            else:
                fallback = normalize_weights(baseline_allocation)
            normalized = fallback
        else:
            normalized = normalize_weights(raw_adjustment)
        allocation_over_time.append(normalized)

    adjusted_allocations = pd.DataFrame(allocation_over_time, index=composite_df.index)
    return adjusted_allocations


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
