from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path

from config import Config
from anomaly.anomaly_detection import remove_anomalous_stocks
from correlation.cluster_assets import get_cluster_labels
from correlation.filter_hdbscan import filter_correlated_groups_hdbscan
from pipeline.process_symbols import process_symbols
from utils.portfolio_utils import normalize_weights, stacked_output
from utils.data_utils import download_multi_ticker_data, process_input_files
from utils.logger import logger


def validate_symbols_override(overrides: List[str]) -> None:
    if not all(isinstance(symbol, str) for symbol in overrides):
        raise ValueError("All elements in symbols_override must be strings.")
    logger.info(f"Symbols overridden: {overrides}")


def load_symbols(
    config: Config, symbols_override: Optional[List[str]] = None
) -> List[str]:
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
    all_symbols: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    config: Config,
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
        logger.error(f"Adjusted close prices not found in the DataFrame. Error: {e}")
        raise


def preprocess_data(
    returns_df: pd.DataFrame, 
    config: Config
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocess the returns DataFrame by applying optional anomaly and decorrelation filters.

    Args:
        returns_df (pd.DataFrame): DataFrame with daily returns for each stock.
        config (Config): Configuration object with settings for filters and clustering.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Filtered DataFrame and asset cluster map.
    """
    # Create asset cluster map (independent of filtering)
    asset_cluster_map = get_cluster_labels(
        returns_df=returns_df, cache_dir="optuna_cache", reoptimize=False
    )

    filtered_returns_df = returns_df

    # Apply anomaly filter if configured
    if config.use_anomaly_filter:
        logger.debug("Applying anomaly filter.")
        valid_symbols = remove_anomalous_stocks(
            returns_df=returns_df,
            reoptimize=False,
            plot=config.plot_anomalies,
        )
        filtered_returns_df = returns_df[valid_symbols]
    else:
        valid_symbols = returns_df.columns.tolist()

    # Apply decorrelation filter if enabled
    if config.use_decorrelation:
        logger.info("Filtering correlated assets...")
        valid_symbols = filter_correlated_assets(
            filtered_returns_df, config, asset_cluster_map
        )

        valid_symbols = [
            symbol for symbol in valid_symbols if symbol in filtered_returns_df.columns
        ]

    filtered_returns_df = filtered_returns_df[valid_symbols]

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
    risk_free_rate_log_daily = np.log(1 + config.risk_free_rate) / trading_days_per_year

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
        logger.warning("No valid symbols after filtering. Returning original symbols.")
        valid_symbols = original_symbols

    return valid_symbols


def perform_post_processing(
    stack_weights: Dict[str, Any],
    config: Config,
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
        logger.warning("No valid averaged weights found. Skipping further processing.")
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
