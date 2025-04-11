from typing import Dict
import numpy as np
import pandas as pd

from reversion.optimize_mean_reversion import optimize_robust_mean_reversion
from reversion.reversion_utils import (
    format_asset_cluster_map,
    johansen_test,
)
from reversion.optimize_period_weights import find_optimal_weights
from reversion.optimize_cointegration_mean_reversion import (
    optimize_cointegration_mean_reversion,
)
from reversion.reversion_signals import compute_individual_stateful_signals_for_group
from utils import logger
from utils.caching_utils import save_parameters_to_pickle


def cluster_mean_reversion(
    price_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    asset_cluster_map: Dict[str, int],
    objective_weights: dict,
    n_trials: int,
    n_jobs: int,
    global_cache: dict,
    checkpoint_file: str,
):
    """
    For each cluster, test cointegration and tune reversion parameters.
    Uses the formatting method to treat noise (cluster_id = -1) as single-ticker clusters.
    Updates global_cache in-place with the tuned parameters and a mode flag ("cointegration" or "fallback").
    """
    # Reformat the cluster map to group tickers by cluster, handling noise separately.
    formatted_clusters = format_asset_cluster_map(asset_cluster_map, global_cache)

    for cluster_label, tickers in formatted_clusters.items():
        # Only keep tickers that exist in both price_df and returns_df.
        tickers = [
            ticker
            for ticker in tickers
            if ticker in price_df.columns and ticker in returns_df.columns
        ]

        # Skip clusters with a single asset (noise) or handle them with a default strategy.
        if len(tickers) < 2:
            logger.info(
                f"Skipping tuning for noise cluster {cluster_label} with ticker(s): {tickers}"
            )
            # Optionally, assign a default parameter set for noise tickers.
            for ticker in tickers:
                global_cache[ticker] = {
                    "mode": "noise",
                    "cluster": cluster_label,
                    "default_params": True,
                }
            continue

        # Get the cluster's price and return series.
        cluster_prices = price_df[tickers].dropna()
        cluster_returns = returns_df[tickers].dropna()

        # Run the Johansen cointegration test on the prices.
        try:
            coint_result = johansen_test(cluster_prices)
        except Exception as e:
            logger.warning(f"Johansen test failed for cluster {cluster_label}: {e}")
            coint_result = None

        # Decide which mode to use based on cointegration test.
        if coint_result and coint_result.cointegration_found:
            tuned_params, _ = optimize_cointegration_mean_reversion(
                prices_df=cluster_prices,
                objective_weights=objective_weights,
                test_window_range=range(5, 61, 5),
                n_trials=n_trials,
                n_jobs=n_jobs,
            )
            tuned_params["mode"] = "cointegration"  # Store mode!
            tuned_params["weight_daily"] = 1.0
            tuned_params["weight_weekly"] = 0.0
        else:
            best_params_daily, _ = optimize_robust_mean_reversion(
                returns_df=cluster_returns,
                objective_weights=objective_weights,
                test_window_range=range(5, 61, 5),
                n_trials=n_trials,
                n_jobs=n_jobs,
            )
            weekly_returns = cluster_returns.resample("W").last()
            best_params_weekly, _ = optimize_robust_mean_reversion(
                returns_df=weekly_returns,
                objective_weights=objective_weights,
                test_window_range=range(1, 26),
                n_trials=n_trials,
                n_jobs=n_jobs,
            )
            tuned_params = {
                "window_daily": round(best_params_daily.get("window", 20), 1),
                "z_threshold_daily_positive": round(
                    best_params_daily.get("z_threshold_positive", 1.5), 1
                ),
                "z_threshold_daily_negative": round(
                    best_params_daily.get("z_threshold_negative", 1.5), 1
                ),
                "window_weekly": round(best_params_weekly.get("window", 5), 1),
                "z_threshold_weekly_positive": round(
                    best_params_weekly.get("z_threshold_positive", 1.5), 1
                ),
                "z_threshold_weekly_negative": round(
                    best_params_weekly.get("z_threshold_negative", 1.5), 1
                ),
                "weight_daily": 0.7,
                "weight_weekly": 0.3,
            }
            best_period_weights = find_optimal_weights(
                daily_signals_df=compute_individual_stateful_signals_for_group(
                    cluster_returns,
                    tuned_params,
                    frequency="daily",
                ),
                weekly_signals_df=compute_individual_stateful_signals_for_group(
                    cluster_returns,
                    tuned_params,
                    frequency="weekly",
                ),
                returns_df=returns_df,
                objective_weights=objective_weights,
                n_trials=n_trials,
                n_jobs=n_jobs,
                group_id=cluster_label,
            )
            tuned_params["weight_daily"] = best_period_weights.get("weight_daily", 0.7)
            tuned_params["weight_weekly"] = 1.0 - tuned_params["weight_daily"]
            tuned_params["mode"] = "fallback"  # Store mode here as well.

        for ticker in tickers:
            global_cache[ticker] = tuned_params.copy()
            global_cache[ticker]["cluster"] = cluster_label

    # Save the updated cache to the checkpoint file.
    save_parameters_to_pickle(global_cache, checkpoint_file)
