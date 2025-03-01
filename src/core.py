# file: src/core.py

from pathlib import Path
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from config import Config
from plotly_graphs import plot_graphs
from portfolio_optimization import run_optimization_and_save
from process_symbols import process_symbols
from anomaly.anomaly_detection import remove_anomalous_stocks
from correlation.filter_hdbscan import (
    filter_correlated_groups_hdbscan,
)
from apply_reversion import (
    apply_ou_reversion,
    apply_z_reversion,
    build_final_result_dict,
    compute_performance_results,
)
from correlation.cluster_assets import get_cluster_labels
from correlation.correlation_utils import compute_lw_covariance
from risk_constraints import apply_risk_constraints
from models.optimize_portfolio import estimated_portfolio_volatility
from utils.logger import logger
from utils.performance_metrics import conditional_var
from utils.caching_utils import cleanup_cache
from utils.data_utils import download_multi_ticker_data, process_input_files
from utils.date_utils import calculate_start_end_dates
from utils.portfolio_utils import (
    normalize_weights,
    stacked_output,
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

    def validate_symbols_override(overrides: List[str]) -> None:
        if not all(isinstance(symbol, str) for symbol in overrides):
            raise ValueError("All elements in symbols_override must be strings.")
        logger.info(f"Symbols overridden: {overrides}")

    def load_symbols() -> List[str]:
        if symbols_override:
            validate_symbols_override(symbols_override)
            return symbols_override
        watchlist_paths = [
            Path(config.input_files_dir) / file for file in config.input_files
        ]
        symbols = process_input_files(watchlist_paths)
        # logger.info(f"Loaded symbols from watchlists: {symbols}")
        return symbols

    def load_data(
        all_symbols: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        logger.debug(
            f"Loading data for symbols: {all_symbols} from {start_date} to {end_date}"
        )
        try:
            data = process_symbols(
                # data = download_multi_ticker_data(
                symbols=all_symbols,
                start_date=start_date,
                end_date=end_date,
                data_path=Path(config.data_dir),
                download=config.download,
            )
            if data.empty:
                logger.warning("Loaded data is empty.")
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily log returns from a multiindex DataFrame with adjusted close prices.

        Args:
            df (pd.DataFrame): Multiindex DataFrame with adjusted close prices under the level "Adj Close".

        Returns:
            pd.DataFrame: DataFrame containing daily log returns for each stock.
        """
        try:
            # Extract only 'Adj Close' level
            adj_close = df.xs("Adj Close", level=1, axis=1)

            # Compute log returns (log difference of prices)
            log_returns = np.log(adj_close).diff()

            # Fill leading NaNs for assets with different start dates
            log_returns = log_returns.ffill()

            # Drop any remaining NaNs or fill with zeros
            log_returns = log_returns.fillna(0)  # Alternative: log_returns.dropna()

            logger.debug(
                f"Calculated daily log returns, {log_returns.isna().sum().sum()} NaN remaining"
            )

            return log_returns

        except KeyError as e:
            logger.error(
                f"Adjusted close prices not found in the DataFrame. Error: {e}"
            )
            raise

    def preprocess_data(
        df: pd.DataFrame, config: Config
    ) -> Tuple[pd.DataFrame, dict[str, Any]]:
        """
        Preprocess the input DataFrame to calculate returns and optionally remove anomalous stocks.

        Args:
            df (pd.DataFrame): Multiindex DataFrame with adjusted close prices.
            config: Configuration object with settings like anomaly detection threshold and plotting options.

        Returns:
            pd.DataFrame: Processed DataFrame with daily returns, with optional anomaly filtering and decorrelation applied.
        """
        returns_df = calculate_returns(df)
        # Create asset cluster map
        asset_cluster_map = get_cluster_labels(
            returns_df=returns_df, cache_dir="optuna_cache", reoptimize=False
        )
        filtered_returns_df = returns_df  # Ensure it's always assigned

        if config.use_anomaly_filter:
            logger.debug("Applying anomaly filter.")
            valid_symbols = remove_anomalous_stocks(
                returns_df=returns_df,
                reoptimize=False,
                plot=config.plot_anomalies,
            )
            filtered_returns_df = returns_df[valid_symbols]
        else:
            valid_symbols = returns_df.columns.tolist()  # Ensure it exists

        # Apply decorrelation filter if enabled
        if config.use_decorrelation:
            logger.info("Filtering correlated assets...")
            valid_symbols = filter_correlated_assets(
                filtered_returns_df, config, asset_cluster_map
            )

            valid_symbols = [
                symbol
                for symbol in valid_symbols
                if symbol in filtered_returns_df.columns
            ]

        filtered_returns_df = filtered_returns_df[valid_symbols]  # Always valid

        # Remove rows where all columns are NaN after all filtering steps
        return filtered_returns_df.dropna(how="all"), asset_cluster_map

    def filter_correlated_assets(
        returns_df: pd.DataFrame,
        config: Config,
        asset_cluster_map: Dict[str, int],
    ) -> List[str]:
        """
        Apply mean reversion and decorrelation filters to return valid asset symbols.
        Falls back to the original returns_df columns if filtering results in an empty list.
        """
        original_symbols = list(returns_df.columns)  # Preserve original symbols
        trading_days_per_year = 252
        risk_free_rate_log_daily = (
            np.log(1 + config.risk_free_rate) / trading_days_per_year
        )

        try:
            decorrelated_tickers = filter_correlated_groups_hdbscan(
                returns_df=returns_df,
                asset_cluster_map=asset_cluster_map,
                risk_free_rate=risk_free_rate_log_daily,
                plot=config.plot_clustering,
                objective=config.optimization_objective,
            )

            valid_symbols = [
                symbol for symbol in original_symbols if symbol in decorrelated_tickers
            ]

        except Exception as e:
            logger.error(f"Correlation threshold optimization failed: {e}")
            # Fall back to original symbols if optimization fails
            valid_symbols = original_symbols

        # Ensure a non-empty list of symbols is returned
        if not valid_symbols:
            logger.warning(
                "No valid symbols after filtering. Returning original symbols."
            )
            valid_symbols = original_symbols

        return valid_symbols

    def perform_post_processing(
        stack_weights: Dict[str, Any],
        period_weights: Optional[Union[Dict[str, float], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Perform post-processing on the stack data to calculate normalized weights.
        Supports period weighting for combining asset weights.

        Args:
            stack_weights (Dict[str, Any]): The stack weights data containing optimization results.
                Expected as a dict mapping time period identifiers to asset weight data (dict or pd.Series).
            period_weights (Optional[Union[Dict[str, float], np.ndarray]]): Optional weights for each time period.
                If provided, these weights will be used when averaging.

        Returns:
            Dict[str, Any]: Normalized weights as a dictionary.
        """
        # Convert any pd.Series in stack_weights to dictionaries.
        processed_stack = {
            key: (value.to_dict() if isinstance(value, pd.Series) else value)
            for key, value in stack_weights.items()
        }

        # Compute average weights (weighted or arithmetic).
        average_weights = stacked_output(processed_stack, period_weights)
        if not average_weights:
            logger.warning(
                "No valid averaged weights found. Skipping further processing."
            )
            return {}

        # Sort weights in descending order.
        sorted_weights = dict(
            sorted(average_weights.items(), key=lambda item: item[1], reverse=True)
        )

        # Normalize weights. (normalize_weights is used elsewhere so we leave it unchanged)
        normalized_weights = normalize_weights(sorted_weights, config.min_weight)
        # logger.debug(f"\nNormalized avg weights: {normalized_weights}")

        # Ensure output is a dictionary.
        if isinstance(normalized_weights, pd.Series):
            normalized_weights = normalized_weights.to_dict()

        return normalized_weights

    ###############################
    ########    MAIN    ###########
    ###############################

    # Load and validate symbols
    try:
        all_symbols = load_symbols()
        if not all_symbols:
            logger.warning("No symbols found. Aborting pipeline.")
            return {}
    except ValueError as e:
        logger.error(f"Symbol override validation failed: {e}")
        return {}

    # Initialize structures
    stack: Dict[str, Any] = {}
    dfs: Dict[str, Any] = {}
    active_models = [k for k, v in config.models.items() if v]
    sorted_time_periods = sorted(active_models, reverse=True)

    # Determine date range based on the longest period
    longest_period = sorted_time_periods[0]
    start_long, end_long = calculate_start_end_dates(longest_period)
    dfs["start"] = start_long
    dfs["end"] = end_long

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df_all = load_data(all_symbols, start_long, end_long)

    # Ensure we use the **largest valid date range** for returns_df
    all_dates = df_all.index  # Keep full range before filtering

    # apply all pre-optimization filters
    returns_df, asset_cluster_map = preprocess_data(df=df_all, config=config)

    valid_symbols = list(returns_df.columns)
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

    #####################################
    ##########  OPTIMIZATION  ###########
    #####################################
    # Iterate through each time period and perform optimization
    for period in sorted_time_periods:
        if period != longest_period:
            config.plot_anomalies = False
            config.plot_clustering = False

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
        df_period.columns.name = None  # Remove column name (Ticker)
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

        run_optimization_and_save(
            df=df_period,
            config=config,
            start_date=start,
            end_date=end,
            symbols=valid_symbols,
            stack=stack,
            years=period,
            plot=config.plot_optimization and period == longest_period,
        )

    ################################
    ######  POST-PROCESSING  #######
    ################################
    logger.info("Post-processing optimization results...")

    if not stack:
        logger.warning("No optimization results found.")
        return {}

    # Post-processing of optimization results
    normalized_avg_weights = perform_post_processing(stack_weights=stack)
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
            combined_models += f" + σ <= {config.portfolio_max_vol:.2f}"
        if config.portfolio_max_cvar is not None:
            combined_models += f" + CVaR <= {config.portfolio_max_cvar:.2f}"

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
        if config.reversion_type == "z":
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
        else:
            final_result_dict = apply_ou_reversion(
                dfs=dfs,
                normalized_avg_weights=final_weights,
                combined_input_files=combined_input_files,
                combined_models=f"{combined_models} + ou-reversion",
                sorted_time_periods=sorted_time_periods,
                config=config,
                # asset_cluster_map=asset_cluster_map,
                returns_df=returns_df,
                allow_short=config.allow_short,
            )

    # Optional plotting (only on local runs)
    if run_local:
        plot_graphs(
            daily_returns=final_result_dict["daily_returns"],
            cumulative_returns=final_result_dict["cumulative_returns"],
            return_contributions=final_result_dict["return_contributions"],
            risk_contributions=final_result_dict["risk_contributions"],
            config=config,
            symbols=final_result_dict["symbols"],
        )

    # Cleanup
    cleanup_cache("cache")
    logger.info("Pipeline execution completed successfully.")

    return final_result_dict
