# file: src/core.py

from pathlib import Path
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from config import Config
from plotly_graphs import plot_graphs
from pipeline.portfolio_optimization import run_optimization_and_save
from apply_reversion import (
    apply_z_reversion,
    build_final_result_dict,
    compute_performance_results,
)
from correlation.correlation_utils import compute_lw_covariance
from risk_constraints import apply_risk_constraints
from models.optimize_portfolio import estimated_portfolio_volatility
from pipeline.data_processing import (
    calculate_returns,
    load_data,
    load_symbols,
    perform_post_processing,
    preprocess_data,
)
from utils.logger import logger
from utils.performance_metrics import conditional_var
from utils.caching_utils import cleanup_cache
from utils.date_utils import calculate_start_end_dates
from utils.portfolio_utils import (
    normalize_weights,
)


def run_pipeline(
    config: Config,
    symbols_override: Optional[List[str]] = None,
    run_local: bool = False,
) -> Dict[str, Any]:
    """
    Orchestrates the data loading, optimization, and analysis pipeline.

    Args:
        config (Config): Configuration object parsed from YAML.
        symbols_override (Optional[List[str]]): Override for ticker symbols.
        run_local (bool): If True, display local plots and log to console.

    Returns:
        Dict[str, Any]: JSON-serializable dictionary containing final results or empty if no data.
    """
    try:
        all_symbols = load_symbols(config, symbols_override)
        if not all_symbols:
            logger.warning("No symbols found. Aborting pipeline.")
            return {}
    except ValueError as e:
        logger.error(f"Symbol override validation failed: {e}")
        return {}

    # Initialize structures
    stack: Dict[str, Any] = {}
    dfs: Dict[str, Any] = {}

    # Process the models dictionary (keys are period strings) into floats.
    active_models = [float(k) for k, v in config.models.items() if v]
    sorted_time_periods = sorted(active_models, reverse=True)
    # Use the API-provided period (from config.period) as the longest period.
    # api_longest = config.period  # API-specified longest period (min 1.0)
    # if (api_longest not in sorted_time_periods) or (
    #     api_longest > sorted_time_periods[0]
    # ):
    #     if api_longest in sorted_time_periods:
    #         sorted_time_periods.remove(api_longest)
    #     sorted_time_periods.insert(0, api_longest)
    longest_period = sorted_time_periods[0]

    # Determine the full date range from the longest period.
    start_long, end_long = calculate_start_end_dates(longest_period)
    dfs["start"] = start_long
    dfs["end"] = end_long

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df_all = load_data(all_symbols, start_long, end_long, config=config)

    # Ensure we use the **largest valid date range** for returns_df
    all_dates = df_all.index  # Keep full range before filtering

    # Calculate log returns on multi index df
    returns_df = calculate_returns(df_all)

    # Apply optional preprocessing (anomaly and decorrelation filters)
    filtered_returns_df, asset_cluster_map = preprocess_data(returns_df, config)

    # Update valid symbols post-filtering
    valid_symbols = list(filtered_returns_df.columns)
    if not valid_symbols:
        logger.warning("No valid symbols remain after filtering. Aborting pipeline.")
        return None  # Or raise an exception if this is an unrecoverable state

    logger.debug(f"Symbols selected for optimization {valid_symbols}")

    try:
        if valid_symbols:
            dfs["data"] = df_all.xs("Adj Close", level=1, axis=1)[valid_symbols]
            logger.debug(f"dfs['data'] shape: {dfs['data'].shape}")
        else:
            logger.warning("No valid symbols available for slicing df_all.")
            dfs["data"] = pd.DataFrame()  # Or handle gracefully
    except KeyError as e:
        logger.error(f"Error slicing df_all: {e}")
        raise

    # Iterate through each time period and perform optimization
    for period in sorted_time_periods:
        # For non-longest periods, temporarily disable anomaly and clustering plots.
        config.plot_anomalies if period == longest_period else False
        config.plot_clustering if period == longest_period else False

        start, end = calculate_start_end_dates(period)
        logger.debug(f"Processing period: {period} from {start} to {end}")

        # Preserve full available date range
        df_period = df_all.loc[start:end].copy()

        # Flatten df_period for optimization
        if isinstance(
            df_period.columns, pd.MultiIndex
        ) and "Adj Close" in df_period.columns.get_level_values(1):
            try:
                df_period = df_period.xs("Adj Close", level=1, axis=1)

                # Ensure stocks with shorter histories remain in the dataset
                all_tickers = df_period.columns.get_level_values(0).unique()
                df_period = df_period.reindex(columns=all_tickers, fill_value=np.nan)

                df_period.columns.name = None  # Flatten MultiIndex properly
            except KeyError:
                logger.warning(
                    "Adj Close column not found. Returning original DataFrame."
                )

        # Align stocks with different start dates properly
        df_period = df_period.reindex(
            index=all_dates, columns=valid_symbols, fill_value=np.nan
        )
        df_period.columns.name = None  # Remove column name
        df_period.index.name = "Date"  # Set index name for clarity

        # Prevent removal of stocks due to shorter histories
        if config.test_mode:
            df_period.to_csv("full_df.csv")
            visible_length = int(len(df_period) * config.test_data_visible_pct)
            df_period = df_period.head(visible_length)
            logger.info(
                f"Test mode active: saved full_df.csv and limited data to {visible_length} records."
            )

        if period == longest_period:
            logger.info(f"Running optimization with {len(valid_symbols)} assets.")

        # For the longest period, enable plotting if configured.
        plot_flag = config.plot_optimization and (period == longest_period)

        run_optimization_and_save(
            df=df_period,
            config=config,
            start_date=start,
            end_date=end,
            symbols=valid_symbols,
            stack=stack,
            years=period,
            plot=plot_flag,
        )

    logger.info("Post-processing optimization results...")

    if not stack:
        logger.warning("No optimization results found.")
        return {}

    # Post-processing of optimization results
    normalized_avg_weights = perform_post_processing(
        stack_weights=stack, config=config, period_weights=None
    )
    if not normalized_avg_weights:
        return {}

    if config.portfolio_max_vol is not None or config.portfolio_max_cvar is not None:
        # Preprocess df_all to compute risk estimates for the merged portfolio.
        # This follows the original logic: flatten the MultiIndex and reindex by all_dates and valid_symbols.
        df_risk = df_all.loc[start_long:end_long].copy()
        if isinstance(
            df_risk.columns, pd.MultiIndex
        ) and "Adj Close" in df_risk.columns.get_level_values(1):
            try:
                df_risk = df_risk.xs("Adj Close", level=1, axis=1)
                all_tickers = df_risk.columns.get_level_values(0).unique()
                df_risk = df_risk.reindex(columns=all_tickers, fill_value=np.nan)
                df_risk.columns.name = None  # Flatten MultiIndex properly
            except KeyError:
                logger.warning(
                    "Adj Close column not found when processing risk estimates. Using original DataFrame."
                )

        # Align stocks with different start dates using the full available date range.
        df_risk = df_risk.reindex(
            index=all_dates, columns=valid_symbols, fill_value=np.nan
        )
        df_risk.index.name = "Date"
        df_risk.columns.name = None

        # Compute returns based on the processed data.
        asset_returns = np.log(df_risk).diff().dropna(how="all")
        valid_assets = asset_returns.dropna(
            thresh=int(len(asset_returns) * 0.5), axis=1
        ).columns
        asset_returns = asset_returns[valid_assets]

        # Compute risk estimates.
        try:
            cov_daily = compute_lw_covariance(asset_returns)
        except ValueError as e:
            logger.error(f"Covariance computation failed: {e}")
            return {}

        trading_days_per_year = 252
        mu_daily = asset_returns.mean()
        mu_annual = mu_daily * trading_days_per_year
        cov_annual = cov_daily * trading_days_per_year
        mu_annual = mu_annual.loc[valid_assets].reindex(valid_assets)

        risk_estimates = {
            "cov": cov_annual,
            "mu": mu_annual,
            "returns": asset_returns,
        }

        # Convert weights dict to numpy array
        if not isinstance(normalized_avg_weights, pd.Series):
            normalized_avg_weights = pd.Series(normalized_avg_weights)

        # Align weights with covariance matrix assets
        cov_assets = risk_estimates["cov"].index
        normalized_avg_weights = normalized_avg_weights.reindex(
            cov_assets, fill_value=0
        )

        # Compute the current portfolio risk measures from the unconstrained weights.
        current_vol = estimated_portfolio_volatility(
            normalized_avg_weights.values, risk_estimates["cov"]
        )
        current_cvar = conditional_var(
            pd.Series(risk_estimates["returns"] @ normalized_avg_weights.values),
            alpha=0.05,
        )

        # If portfolio_max_vol is not specified, set it to the current portfolio's volatility.
        if config.portfolio_max_cvar is not None and config.portfolio_max_vol is None:
            logger.info(
                f"portfolio_max_vol not specified with portfolio_max_cvar enabled; using current portfolio vol: {current_vol:.2f} and shorting enabled"
            )
            config.portfolio_max_vol = current_vol
            config.allow_short = True

        # Final pass: apply risk constraints to the merged portfolio
        risk_adjusted_weights = (
            apply_risk_constraints(normalized_avg_weights, risk_estimates, config)
            if (config.portfolio_max_cvar or config.portfolio_max_vol)
            else normalized_avg_weights
        )

        final_weights = normalize_weights(
            weights=risk_adjusted_weights,
            min_weight=config.min_weight,
        )
    else:
        final_weights = normalized_avg_weights

    # Prepare input metadata
    valid_models = [
        model for models in config.models.values() if models for model in models
    ]
    combined_models = ", ".join(sorted(set(valid_models)))
    combined_input_files = ", ".join(config.input_files)

    # Convert dict to Pandas Series
    final_weights_series = pd.Series(final_weights)
    if not (final_weights_series.equals(normalized_avg_weights)):
        if config.portfolio_max_vol is not None:
            combined_models += f" + σ <= {float(config.portfolio_max_vol):.2f}"
        if config.portfolio_max_cvar is not None:
            combined_models += f" + CVaR <= {float(config.portfolio_max_cvar):.2f}"

    if config.use_global_optimization:
        if config.global_optimization_type == "annealing":
            combined_models += " + annealing"
        elif config.global_optimization_type == "diffusion":
            combined_models += " + diffusion"

    # Sort symbols and filter DataFrame accordingly
    sorted_symbols = sorted(final_weights.keys())
    dfs["data"] = dfs["data"].filter(items=sorted_symbols)

    # Prevent empty DataFrame after filtering
    if dfs["data"].empty:
        logger.error("No valid symbols remain in the DataFrame after alignment.")
        return {}

    # --- Baseline (Pre-Mean Reversion) Results ---
    (
        daily_returns,
        cumulative_returns,
        pre_boxplot_stats,
        return_contributions_pct,
        risk_contributions_pct,
        valid_symbols,
    ) = compute_performance_results(
        data=dfs["data"],
        start_date=str(dfs["start"]),
        end_date=str(dfs["end"]),
        allocation_weights=final_weights,
        sorted_symbols=sorted_symbols,
        combined_input_files=combined_input_files,
        combined_models=combined_models,
        sorted_time_periods=sorted_time_periods,
        config=config,
    )

    final_result_dict = build_final_result_dict(
        start_date=str(dfs["start"]),
        end_date=str(dfs["end"]),
        models=combined_models,
        symbols=valid_symbols,
        normalized_avg=final_weights,
        daily_returns=daily_returns,
        cumulative_returns=cumulative_returns,
        boxplot_stats=pre_boxplot_stats,
        return_contributions=return_contributions_pct,
        risk_contributions=risk_contributions_pct,
    )

    # --- Mean Reversion Branches ---
    if config.use_reversion:
        final_result_dict = apply_z_reversion(
            dfs=dfs,
            normalized_avg_weights=final_weights,
            combined_input_files=combined_input_files,
            combined_models=f"{combined_models} + z-reversion",
            sorted_time_periods=sorted_time_periods,
            config=config,
            asset_cluster_map=asset_cluster_map,
            returns_df=returns_df,
        )

    # Optional plotting (only on local runs)
    if run_local:
        plot_graphs(
            daily_returns=final_result_dict["daily_returns"],
            cumulative_returns=final_result_dict["cumulative_returns"],
            return_contributions=final_result_dict["return_contributions"],
            risk_contributions=final_result_dict["risk_contributions"],
            plot_contribution=config.plot_contribution,
            plot_cumulative_returns=config.plot_cumulative_returns,
            plot_daily_returns=config.plot_daily_returns,
            symbols=final_result_dict["symbols"],
            theme="light",
        )

    # Cleanup
    cleanup_cache("cache")
    logger.info("Pipeline execution completed successfully.")

    return final_result_dict
