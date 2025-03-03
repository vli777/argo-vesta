from datetime import datetime
from typing import Any, Dict, Tuple
import pandas as pd
from config import Config

from reversion.cluster_mean_reversion import cluster_mean_reversion
from reversion.reversion_utils import (
    adjust_allocation_with_mean_reversion,
    calculate_continuous_composite_signal,
    group_ticker_params_by_cluster,
    is_cache_stale,
    propagate_signals_by_similarity,
)
from reversion.optimize_reversion_strength import tune_reversion_alpha
from models.optimizer_utils import get_objective_weights
from reversion.reversion_signals import compute_group_stateful_signals
from reversion.reversion_plots import plot_reversion_signals
from utils import logger
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def apply_mean_reversion(
    asset_cluster_map: Dict[str, int],
    baseline_allocation: pd.Series,
    returns_df: pd.DataFrame,
    config: Config,
    cache_dir: str = "optuna_cache",
) -> pd.Series:
    """
    Generate continuous mean reversion signals on clusters of stocks and overlay
    the adjustment onto the baseline allocation using a continuous adjustment factor.

    Args:
        asset_cluster_map (Dict[str, int]): Maps each ticker to a cluster ID.
        baseline_allocation (pd.Series): Baseline allocation for each ticker.
        returns_df (pd.DataFrame): Log returns for each ticker.
        config (Config): Configuration object with optimization objectives, etc.
        cache_dir (str): Directory where optimization results are cached.

    Returns:
        pd.Series: The final mean-reversion-adjusted allocation.
    """
    # 1. Load or initialize the cache
    reversion_cache_file = (
        f"{cache_dir}/reversion_cache_{config.options["optimization_objective"]}.pkl"
    )
    reversion_cache = load_parameters_from_pickle(reversion_cache_file)
    if not isinstance(reversion_cache, dict):
        reversion_cache = {}

    # 2. Ensure the cache has a "params" section for storing hyperparameters
    reversion_cache.setdefault("params", {})
    last_updated = reversion_cache.get("last_updated")
    cache_is_stale = is_cache_stale(last_updated)

    # 3. Identify which tickers are missing parameters in the cache
    missing_tickers = [
        t for t in returns_df.columns if t not in reversion_cache["params"]
    ]

    # 4. Objective weights (e.g., sharpe, cumulative return, etc.)
    objective_weights = get_objective_weights(objective=config.options["optimization_objective"])

    # 5. Optimize (if stale or missing)
    if cache_is_stale or missing_tickers:
        # Only pass missing tickers if partial fill, otherwise pass all
        returns_subset = returns_df[missing_tickers] if missing_tickers else returns_df

        # This function updates 'global_cache' in-place with new parameters for missing clusters
        # and returns only the parameters, not signals
        cluster_mean_reversion(
            asset_cluster_map=asset_cluster_map,
            returns_df=returns_subset,
            objective_weights=objective_weights,
            n_trials=50,
            n_jobs=-1,
            global_cache=reversion_cache["params"],
            checkpoint_file=reversion_cache_file,
        )

        # Update cache timestamps and persist
        reversion_cache["last_updated"] = datetime.now().isoformat()
        save_parameters_to_pickle(reversion_cache, reversion_cache_file)
    else:
        print("Cache is fresh; skipping reversion optimization.")

    # 6. Load the parameters from the cache
    ticker_params = reversion_cache["params"]
    print(f"Reversion parameters loaded for {len(ticker_params)} tickers.")

    # 7. Dynamically compute signals from the cached parameters
    #    Each ticker's daily/weekly parameters, decay, etc. are used here
    param_tickers = list(reversion_cache["params"].keys())
    valid_tickers = [t for t in param_tickers if t in returns_df.columns]

    signals_dict = compute_group_stateful_signals(
        group_returns=returns_df[valid_tickers],
        tickers=valid_tickers,
        params=reversion_cache["params"],
        target_decay=0.5,
        reset_factor=0.5,
    )
    # logger.info(f"Signals: {signals_dict}")
    # Collapse daily/weekly signals into one scalar per ticker
    composite_signals = calculate_continuous_composite_signal(
        signals=signals_dict,
        ticker_params=ticker_params,
    )

    if config.options["plot_reversion"]:
        plot_reversion_signals(composite_signals)
    # logger.info(f"composite_signals: {composite_signals}")
    # 8. Propagate signals by similarity (if desired)
    group_mapping = group_ticker_params_by_cluster(ticker_params)
    updated_composite_signals = propagate_signals_by_similarity(
        composite_signals=composite_signals,
        group_mapping=group_mapping,
        returns_df=returns_df,
        signal_strength=0.88,
        lw_threshold=50,
    )
    # logger.info(
    #     f"updated_composite_signals after propagation: {updated_composite_signals}"
    # )
    # 9. Tune alpha for how aggressively you act on the signals
    base_alpha = tune_reversion_alpha(
        returns_df=returns_df,
        baseline_allocation=baseline_allocation,
        composite_signals=updated_composite_signals,
        group_mapping=group_mapping,
        objective_weights=objective_weights,
        hv_window=50,
    )

    # # 10. Adjust alpha by realized volatility
    realized_volatility = returns_df.rolling(window=20).std().mean(axis=1)
    adaptive_alpha = base_alpha / (1 + realized_volatility.iloc[-1])

    # 11. Adjust the baseline allocation with the final signals and alpha
    final_allocation = adjust_allocation_with_mean_reversion(
        baseline_allocation=baseline_allocation,
        composite_signals=updated_composite_signals,
        alpha=adaptive_alpha,
        allow_short=config.options["allow_short"],
    )

    return final_allocation
