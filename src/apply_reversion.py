from typing import Dict, Tuple
import numpy as np
import pandas as pd

from reversion.apply_mean_reversion import apply_mean_reversion
from result_output import build_final_result_dict, compute_performance_results
from config import Config
from stat_arb.plot_ou_signals import plot_all_ticker_signals
from stat_arb.apply_adaptive_weighting import apply_adaptive_weighting
from stat_arb.multi_asset_plots import (
    plot_multi_ou_signals,
)
from stat_arb.multi_asset_reversion import MultiAssetReversion
from stat_arb.portfolio_allocator import PortfolioAllocator
from stat_arb.single_asset_reversion import SingleAssetReversion
from utils.portfolio_utils import normalize_weights
from utils.logger import logger


def apply_z_reversion(
    dfs: dict,
    normalized_avg_weights: dict,
    combined_input_files: str,
    combined_models: str,
    sorted_time_periods: list,
    config: Config,
    asset_cluster_map: dict,
    returns_df: pd.DataFrame,
) -> dict:
    """
    Applies Z-score-based mean reversion to update the portfolio weights
    and returns a final result dictionary.
    """
    logger.info("\nApplying Z-score-based mean reversion on normalized weights...")

    # Use the preprocessed adjusted close price DataFrame
    price_df = dfs["data"]

    # Compute mean-reverted weights
    mean_reverted_weights = apply_mean_reversion(
        price_df=price_df,
        returns_df=returns_df,
        asset_cluster_map=asset_cluster_map,
        baseline_allocation=normalized_avg_weights,
        config=config,
        cache_dir="optuna_cache",
    )
    sorted_symbols_post = sorted(mean_reverted_weights.keys())
    # Filter the data to only include the post-reversion symbols
    dfs["data"] = dfs["data"].filter(items=sorted_symbols_post)

    # Compute performance metrics
    (
        post_daily_returns,
        post_cumulative_returns,
        post_boxplot_stats,
        return_contributions_pct,
        risk_contributions_pct,
        valid_symbols,
    ) = compute_performance_results(
        data=dfs["data"],
        start_date=str(dfs["start"]),
        end_date=str(dfs["end"]),
        allocation_weights=mean_reverted_weights,
        sorted_symbols=sorted_symbols_post,
        combined_input_files=combined_input_files,
        combined_models=combined_models,
        sorted_time_periods=sorted_time_periods,
        config=config,
    )

    # Build and return the final result dictionary
    return build_final_result_dict(
        start_date=str(dfs["start"]),
        end_date=str(dfs["end"]),
        models=combined_models,
        symbols=valid_symbols,
        normalized_avg=mean_reverted_weights,
        daily_returns=post_daily_returns,
        cumulative_returns=post_cumulative_returns,
        boxplot_stats=post_boxplot_stats,
        return_contributions=return_contributions_pct,
        risk_contributions=risk_contributions_pct,
    )
