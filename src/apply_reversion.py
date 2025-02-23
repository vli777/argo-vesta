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
    # Compute mean-reverted weights
    mean_reverted_weights = apply_mean_reversion(
        asset_cluster_map=asset_cluster_map,
        baseline_allocation=normalized_avg_weights,
        returns_df=returns_df,
        config=config,
        cache_dir="optuna_cache",
    )
    sorted_symbols_post = sorted(mean_reverted_weights.keys())
    # Filter the data to only include the post-reversion symbols
    dfs["data"] = dfs["data"].filter(items=sorted_symbols_post)

    # Compute performance metrics using the common helper
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


def apply_ou_reversion(
    dfs: dict,
    normalized_avg_weights: dict,
    combined_input_files: str,
    combined_models: str,
    sorted_time_periods: list,
    config: Config,
    returns_df: pd.DataFrame,
    asset_cluster_map: dict[str, int] = None,
    allow_short: bool = False,
    use_weekly: bool = False,
) -> dict:
    """
    Applies OU-based (heat potential) mean reversion and returns the final
    result dictionary.
    """
    logger.info("\nApplying OU-based mean reversion...")

    if use_weekly:  # Resample prices to weekly to remove noise
        data_df = dfs["data"].resample("W").last().dropna()
        returns_df = data_df.pct_change().dropna()
    else:
        data_df = dfs["data"]

    # --- Prepare individual OU strategies for each ticker
    ou_strategies = {
        ticker: SingleAssetReversion(data_df[ticker], returns_df[ticker])
        for ticker in data_df.columns
    }

    # Generate trading signals for each ticker
    ou_signals = {
        ticker: ou.generate_trading_signals() for ticker, ou in ou_strategies.items()
    }

    # Simulate strategy performance using the generated signals
    ou_results = {
        ticker: ou.simulate_strategy(ou_signals[ticker])
        for ticker, ou in ou_strategies.items()
    }

    # Extract individual returns from strategy results
    individual_returns = {
        ticker: pd.Series(result[0], index=data_df.index).fillna(0)
        for ticker, result in ou_results.items()
    }

    # Ensure consistent length by reindexing to match the full date range of data_df
    if use_weekly:
        individual_returns = {
            ticker: s.reindex(data_df.index, fill_value=0)
            for ticker, s in individual_returns.items()
        }
    else:
        max_len = max(s.size for s in individual_returns.values())
        individual_returns = {
            ticker: s.reindex(range(max_len), fill_value=0)
            for ticker, s in individual_returns.items()
        }

    # --- Cross-asset (multi-asset) mean reversion
    multi_asset_strategy = MultiAssetReversion(
        prices_df=data_df,
        asset_cluster_map=asset_cluster_map,
        hidden_channels=64,
        num_epochs=500,
    )
    multi_asset_results = multi_asset_strategy.optimize_and_trade()

    # Get hedge ratios for each asset (ensure order matches data_df.columns)
    weights_series = pd.Series(multi_asset_results["Hedge Ratios"]).reindex(
        data_df.columns, fill_value=0
    )
    if weights_series.sum() != 0:
        weights_series /= weights_series.abs().sum()

    # Compute basket returns based on the selected data frequency
    basket_returns = (
        data_df.pct_change().fillna(0).mul(weights_series, axis=1).sum(axis=1)
    )

    # Compute multi-asset returns using the signals from the multi-asset strategy
    multi_asset_returns = (
        multi_asset_results["Signals"]["Position"].shift(1) * basket_returns
    )
    multi_asset_returns = multi_asset_returns.fillna(0)

    if config.plot_reversion:
        filtered_price_data = {
            ticker: dfs["data"][ticker]
            for ticker, signals in ou_signals.items()
            if signals is not None and not signals.empty
        }
        plot_all_ticker_signals(
            price_data=filtered_price_data,
            signal_data=ou_signals,
        )
        signals = multi_asset_strategy.generate_trading_signals()
        stop_loss, take_profit = multi_asset_strategy.calculate_optimal_bounds
        fig = plot_multi_ou_signals(
            mar=multi_asset_strategy,
            signals=signals,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        fig.show()

    latest_ou_signals = {}
    for ticker, ou in ou_strategies.items():
        signals = ou.generate_trading_signals()  # your DataFrame of signals
        # Get the last non-"NO_SIGNAL" row if available, otherwise default to neutral:
        non_neutral = signals[signals["Position"] != "NO_SIGNAL"]
        if not non_neutral.empty:
            last_signal = non_neutral.iloc[-1]["Position"]
        else:
            last_signal = "NO_SIGNAL"
        # Compute current deviation from the most recent price:
        current_price = ou.prices.iloc[-1]
        current_deviation = np.log(current_price) - ou.mu
        # Also get the optimal thresholds for this asset:
        stop_loss, take_profit = ou.calculate_optimal_bounds
        # Save a richer dictionary per ticker:
        latest_ou_signals[ticker] = {
            "signal": last_signal,
            "current_deviation": current_deviation,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "sigma": ou.sigma,
        }

    portfolio_allocator = PortfolioAllocator()
    reversion_allocations = portfolio_allocator.compute_allocations(
        individual_returns,
        multi_asset_returns,
        hedge_ratios=multi_asset_results["Hedge Ratios"],
        individual_signals=latest_ou_signals,
    )
    normalized_reversion_allocations = normalize_weights(reversion_allocations)
    stat_arb_adjusted_allocation = apply_adaptive_weighting(
        baseline_allocation=normalized_avg_weights,
        mean_reversion_weights=normalized_reversion_allocations,
        returns_df=returns_df,
        base_alpha=0.2,
        allow_short=allow_short,
    )
    sorted_stat_arb_allocation = stat_arb_adjusted_allocation.sort_values(
        ascending=False
    )

    # Compute performance with the adjusted (OU-based) allocations.
    # (Assuming that the symbols remain the same; otherwise, update as needed.)
    (
        adjusted_daily_returns,
        adjusted_cumulative_returns,
        adjusted_boxplot_stats,
        return_contributions_pct,
        risk_contributions_pct,
        valid_symbols,
    ) = compute_performance_results(
        data=dfs["data"],
        start_date=str(dfs["start"]),
        end_date=str(dfs["end"]),
        allocation_weights=stat_arb_adjusted_allocation,
        sorted_symbols=sorted(normalized_avg_weights.keys()),
        combined_input_files=combined_input_files,
        combined_models=combined_models,
        sorted_time_periods=sorted_time_periods,
        config=config,
    )

    return build_final_result_dict(
        start_date=str(dfs["start"]),
        end_date=str(dfs["end"]),
        models=combined_models,
        symbols=valid_symbols,  # sorted(normalized_avg_weights.keys()),
        normalized_avg=sorted_stat_arb_allocation,
        daily_returns=adjusted_daily_returns,
        cumulative_returns=adjusted_cumulative_returns,
        boxplot_stats=adjusted_boxplot_stats,
        return_contributions=return_contributions_pct,
        risk_contributions=risk_contributions_pct,
    )
