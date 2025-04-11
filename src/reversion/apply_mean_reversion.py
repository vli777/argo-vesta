from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Tuple
import pandas as pd
from config import Config

from reversion.cluster_mean_reversion import cluster_mean_reversion
from reversion.reversion_utils import (
    adjust_allocation_series_with_mean_reversion,
    calculate_continuous_composite_signal,
    group_ticker_params_by_cluster,
    is_cache_stale,
    propagate_signals_by_similarity,
)
from reversion.optimize_reversion_strength import tune_reversion_alpha
from models.optimizer_utils import get_objective_weights
from reversion.reversion_signals import (
    compute_cluster_stateful_signal,
    compute_individual_stateful_signals_for_group,
)
from reversion.reversion_plots import plot_reversion_signals
from utils.portfolio_utils import normalize_weights
from utils import logger
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def apply_mean_reversion(
    price_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    asset_cluster_map: Dict[str, int],
    baseline_allocation: pd.Series,
    config: Config,
    cache_dir: str = "optuna_cache",
) -> pd.Series:
    """
    Generate continuous mean reversion signals on clusters of stocks and overlay
    the adjustment onto the baseline allocation using a continuous adjustment factor.

    Args:
        price_df (pd.DataFrame): Adj close price for each ticker.
        returns_df (pd.DataFrame): Log returns for each ticker.
        asset_cluster_map (Dict[str, int]): Maps each ticker to a cluster ID.
        baseline_allocation (pd.Series): Baseline allocation for each ticker.
        config (Config): Configuration object with optimization objectives, etc.
        cache_dir (str): Directory where optimization results are cached.

    Returns:
        pd.Series: The final mean-reversion-adjusted allocation.
    """
    # 1. Load or initialize the cache
    reversion_cache_file = (
        f"{cache_dir}/z_reversion_cache_{config.optimization_objective}.pkl"
    )
    reversion_cache = load_parameters_from_pickle(reversion_cache_file)
    if not isinstance(reversion_cache, dict):
        reversion_cache = {}

    # 2. Ensure the cache has a "params" section for storing hyperparameters
    reversion_cache.setdefault("params", {})
    cache_is_stale = is_cache_stale(reversion_cache_file)

    # 3. Identify which tickers are missing parameters in the cache
    missing_tickers = [
        t for t in returns_df.columns if t not in reversion_cache["params"]
    ]

    # 4. Objective weights (e.g., sharpe, cumulative return, etc.)
    objective_weights = get_objective_weights(objective=config.optimization_objective)

    # 5. Optimize (if stale or missing)
    if cache_is_stale or missing_tickers:
        # Only pass missing tickers if partial fill, otherwise pass all
        returns_subset = returns_df[missing_tickers] if missing_tickers else returns_df

        # This function updates 'global_cache' in-place with new parameters for missing clusters
        cluster_mean_reversion(
            price_df=price_df,
            returns_df=returns_subset,
            asset_cluster_map=asset_cluster_map,
            objective_weights=objective_weights,
            n_trials=50,
            n_jobs=-1,
            global_cache=reversion_cache["params"],
            checkpoint_file=reversion_cache_file,
        )

        # Update cache
        save_parameters_to_pickle(reversion_cache, reversion_cache_file)
    else:
        print("Cache is fresh; skipping reversion optimization.")

    # 6. Load the parameters from the cache
    ticker_params = reversion_cache["params"]
    print(f"Reversion parameters loaded for {len(ticker_params)} tickers.")

    # 7. Group tickers by their cluster ID (only include tickers in the returns data)
    cluster_groups = defaultdict(list)
    for ticker, params in ticker_params.items():
        if ticker in returns_df.columns:
            cluster_groups[params.get("cluster")].append(ticker)

    # 8. Compute cluster-level signals for each cluster group.
    cluster_signals = {}
    for cluster_id, tickers in cluster_groups.items():
        # Get the returns for tickers in this cluster.
        group_returns = returns_df[tickers].dropna(how="all", axis=1)
        # Use the tuned parameters from one ticker in the cluster (they should be similar across the cluster).
        tuned_params = ticker_params[tickers[0]]

        # Compute signals in a consistent dictionary format with "daily" and "weekly" keys.
        if tuned_params.get("mode") == "cointegration":
            # Cointegration produces only a daily signal.
            daily_signal = compute_cluster_stateful_signal(
                group_returns=group_returns,
                tuned_params=tuned_params,
            )
            # For consistency, assign the same daily signal to "weekly" as well.
            signals = {"daily": daily_signal, "weekly": daily_signal}
        elif tuned_params.get("mode") == "fallback":
            # Compute separate signals for daily and weekly frequencies.
            daily_signal = compute_individual_stateful_signals_for_group(
                group_returns=group_returns,
                tuned_params=tuned_params,
                frequency="daily",
            )
            weekly_signal = compute_individual_stateful_signals_for_group(
                group_returns=group_returns,
                tuned_params=tuned_params,
                frequency="weekly",
            )
            signals = {"daily": daily_signal, "weekly": weekly_signal}
        else:
            # Default to empty signals if mode is unrecognized.
            signals = {
                "daily": pd.Series(dtype=float),
                "weekly": pd.Series(dtype=float),
            }

        # Assign the computed signals to each ticker in the current cluster.
        for ticker in tickers:
            cluster_signals[ticker] = signals

    # Now cluster_signals is a dict mapping each ticker to its computed cluster-level signal.
    logger.info(f"Computed cluster-level signals for {len(cluster_signals)} tickers.")
    logger.info(f"Cluster group signals: {cluster_signals}")

    # 9. Collapse daily/weekly signals into one scalar per ticker
    composite_signals = calculate_continuous_composite_signal(
        signals=cluster_signals,
        ticker_params=ticker_params,
    )
    logger.info(f"Composite_signals: {composite_signals}")

    # Optionally plot signals
    if config.plot_reversion:
        plot_reversion_signals(composite_signals)

    # 10. Conditionally propagate signals through cluster.
    group_mapping = group_ticker_params_by_cluster(ticker_params)

    # Separate fallback signals from cointegration signals.
    propagation_input_signals = {}
    for ticker, signal in composite_signals.items():
        mode = ticker_params.get(ticker, {}).get("mode", "fallback")
        if mode == "fallback":
            propagation_input_signals[ticker] = signal
        # For cointegration signals, we can leave them out of propagation.

    # Propagate only fallback signals.
    if propagation_input_signals:
        propagated_signals = propagate_signals_by_similarity(
            composite_signals=propagation_input_signals,
            group_mapping=group_mapping,
            returns_df=returns_df,
            signal_strength=0.88,
        )
    else:
        propagated_signals = {}

    # Combine propagated fallback signals with cointegration signals.
    updated_composite_signals = composite_signals.copy()
    updated_composite_signals.update(propagated_signals)
    logger.info(
        f"Updated_composite_signals after propagation: {updated_composite_signals}"
    )

    # 11. Set signal strength alpha based on mode.
    # Separate tickers by mode from the baseline (ensuring they exist in ticker_params)
    cointegration_tickers = [
        ticker
        for ticker, base_alloc in baseline_allocation.items()
        if ticker_params.get(ticker, {}).get("mode", "fallback") == "cointegration"
    ]
    fallback_tickers = [
        ticker
        for ticker, base_alloc in baseline_allocation.items()
        if ticker_params.get(ticker, {}).get("mode", "fallback") == "fallback"
    ]

    if fallback_tickers:
        # Tune alpha only for fallback signals.
        fallback_signals = {
            ticker: updated_composite_signals[ticker] for ticker in fallback_tickers
        }
        base_alpha_fallback = tune_reversion_alpha(
            returns_df=returns_df,
            baseline_allocation=baseline_allocation,
            composite_signals=fallback_signals,
            group_mapping=group_mapping,
            objective_weights=objective_weights,
            ticker_params=ticker_params,
            hv_window=50,
        )
        realized_volatility = returns_df.rolling(window=20).std().mean(axis=1)
        adaptive_alpha_fallback = base_alpha_fallback / (
            1 + realized_volatility.iloc[-1]
        )
    else:
        adaptive_alpha_fallback = None  # not used if no fallback assets

    # Build vectorized baseline allocation and composite signals for each group.
    baseline_cointegration = pd.Series(
        {ticker: baseline_allocation[ticker] for ticker in cointegration_tickers}
    )
    baseline_fallback = pd.Series(
        {ticker: baseline_allocation[ticker] for ticker in fallback_tickers}
    )

    composite_cointegration = pd.Series(
        {ticker: updated_composite_signals[ticker] for ticker in cointegration_tickers}
    )
    composite_fallback = pd.Series(
        {ticker: updated_composite_signals[ticker] for ticker in fallback_tickers}
    )

    # For cointegrated tickers, use alpha=1.0 (full signal application).
    adjusted_cointegration = adjust_allocation_series_with_mean_reversion(
        baseline_allocation=baseline_cointegration,
        composite_signals=composite_cointegration,
        alpha=1.0,
        allow_short=config.allow_short,
    )

    # For fallback tickers, use the tuned adaptive alpha.
    fallback_alpha = (
        adaptive_alpha_fallback if adaptive_alpha_fallback is not None else 1.0
    )
    adjusted_fallback = adjust_allocation_series_with_mean_reversion(
        baseline_allocation=baseline_fallback,
        composite_signals=composite_fallback,
        alpha=fallback_alpha,
        allow_short=config.allow_short,
    )

    # Combine the adjusted allocations
    final_allocation_series = pd.concat([adjusted_cointegration, adjusted_fallback])

    # Optionally, re-normalize final allocation (if normalize_weights doesn't already guarantee it)
    final_allocation_series = normalize_weights(final_allocation_series)

    # Convert to dict if desired.
    final_allocation = final_allocation_series.to_dict()

    return final_allocation
