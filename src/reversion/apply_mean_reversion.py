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
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def apply_mean_reversion(
    asset_cluster_map: Dict[str, int],
    baseline_allocation: pd.Series,
    returns_df: pd.DataFrame,
    config: Config,
    cache_dir: str = "optuna_cache",
) -> pd.Series:
    reversion_cache_file = (
        f"{cache_dir}/reversion_cache_{config.optimization_objective}.pkl"
    )
    reversion_cache = load_parameters_from_pickle(reversion_cache_file)
    if not isinstance(reversion_cache, dict):
        reversion_cache = {}

    reversion_cache.setdefault("params", {})
    reversion_cache.setdefault("signals", {})
    last_updated = reversion_cache.get("last_updated")
    cache_is_stale = is_cache_stale(last_updated)

    existing_signal_tickers = set()
    for cluster_key, cluster_signals in reversion_cache["signals"].items():
        if isinstance(cluster_signals, dict):
            existing_signal_tickers.update(cluster_signals.keys())

    missing_tickers = [
        ticker for ticker in returns_df.columns if ticker not in existing_signal_tickers
    ]

    objective_weights = get_objective_weights(objective=config.optimization_objective)

    # Only re-optimize if the cache is stale or some tickers are missing.
    if cache_is_stale or missing_tickers:
        returns_subset = returns_df[missing_tickers] if missing_tickers else returns_df
        updated_signals = cluster_mean_reversion(
            asset_cluster_map=asset_cluster_map,
            returns_df=returns_subset,
            objective_weights=objective_weights,
            n_trials=50,
            n_jobs=-1,
            global_cache=reversion_cache["params"],
            checkpoint_file=reversion_cache_file,  # Incremental checkpointing.
        )

        reversion_cache["signals"].update(updated_signals)
        reversion_cache["last_updated"] = datetime.now().isoformat()
        save_parameters_to_pickle(reversion_cache, reversion_cache_file)
    else:
        print("Cache is fresh; skipping reversion optimization.")

    print("Reversion Signals Generated.")
    ticker_params = reversion_cache["params"]
    print(f"Loaded Ticker Parameters for {len(ticker_params)} tickers.")

    composite_signals = calculate_continuous_composite_signal(
        signals=reversion_cache["signals"], ticker_params=ticker_params
    )
    group_mapping = group_ticker_params_by_cluster(ticker_params)
    updated_composite_signals = propagate_signals_by_similarity(
        composite_signals=composite_signals,
        group_mapping=group_mapping,
        returns_df=returns_df,
        signal_dampening=0.5,
        lw_threshold=50,
    )

    base_alpha = tune_reversion_alpha(
        returns_df=returns_df,
        baseline_allocation=baseline_allocation,
        composite_signals=updated_composite_signals,
        group_mapping=group_mapping,
        objective_weights=objective_weights,
        hv_window=50,
    )
    print(f"Baseline alpha: {base_alpha}")
    realized_volatility = returns_df.rolling(window=20).std().mean(axis=1)
    adaptive_alpha = base_alpha / (1 + realized_volatility.iloc[-1])
    print(f"Adaptive alpha: {adaptive_alpha}")

    final_allocation = adjust_allocation_with_mean_reversion(
        baseline_allocation=baseline_allocation,
        composite_signals=updated_composite_signals,
        alpha=adaptive_alpha,
        allow_short=config.allow_short,
    )

    return final_allocation
