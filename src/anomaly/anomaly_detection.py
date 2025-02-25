from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from joblib import Parallel, delayed


from anomaly.plot_anomalies import plot_anomaly_overview
from anomaly.isolation_forest import apply_isolation_forest
from anomaly.plot_optimization_summary import plot_optimization_summary
from anomaly.kalman_filter import apply_kalman_filter
from anomaly.anomaly_utils import apply_fixed_zscore, get_cache_filename
from anomaly.optimize_anomaly_threshold import optimize_threshold_for_ticker
from utils.logger import logger
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def remove_anomalous_stocks(
    returns_df: pd.DataFrame,
    weight_dict: Optional[Dict[str, float]] = None,
    plot: bool = False,
    n_jobs: int = -1,
    reoptimize: bool = False,
    max_anomaly_fraction: float = 0.01,
    contamination: Union[float, str, None] = None,
) -> List[str]:
    """
    Filters out stocks with anomalous returns using Isolation Forest (IF),
    Kalman Filter (KF), or fixed Z-score. Uses per-ticker optimization and caches
    method-specific parameters as well as a list of stocks previously flagged as anomalous.
    The anomalous stock list is saved to 'anomalous_assets.pkl' so that in future runs,
    stocks already determined to be anomalous can be skipped regardless of the filter used.

    Args:
        returns_df (pd.DataFrame): DataFrame where each column corresponds to a stock's returns.
        weight_dict (Optional[Dict[str, float]]): Weights for optimization criteria.
        plot (bool): If True, generate anomaly and optimization plots.
        n_jobs (int): Number of parallel jobs to run.
        reoptimize (bool): If True, bypass cache and reoptimize thresholds.
        max_anomaly_fraction (float): Maximum fraction of anomalous data allowed per stock.
        contamination (Union[float, str, None]): Contamination parameter for Isolation Forest.

    Returns:
        List[str]: List of tickers that are not flagged as anomalous.
    """
    weight_dict = weight_dict or {"kappa": 0.8, "stability": 0.2}
    n_stocks = len(returns_df.columns)
    # Use "auto" if contamination is None to avoid sklearn errors.
    contamination_val = contamination if contamination is not None else "auto"

    if n_stocks > 20:
        method = "IF"
        use_isolation_forest, use_kalman_filter, use_fixed_zscore = True, False, False
    elif 5 < n_stocks <= 20:
        method = "KF"
        use_kalman_filter, use_isolation_forest, use_fixed_zscore = True, False, False
    else:
        method = "Z-score"
        use_fixed_zscore, use_isolation_forest, use_kalman_filter = True, False, False

    # Load the tuning parameters cache for the selected method.
    cache_filename = get_cache_filename(method)
    tuning_cache: Dict[str, Any] = (
        {} if reoptimize else load_parameters_from_pickle(cache_filename) or {}
    )

    # Load the anomalous assets cache (list of stocks already flagged as anomalous) regardless of detection method.
    anomalous_cache_filename = "optuna_cache/anomalous_assets.pkl"
    anomalous_cache: List[str] = (
        []
        if reoptimize
        else load_parameters_from_pickle(anomalous_cache_filename) or []
    )

    def process_ticker(stock: str) -> Optional[Dict[str, Any]]:
        # If the stock is already flagged as anomalous, skip processing.
        if (stock in anomalous_cache) and not reoptimize:
            return {
                "stock": stock,
                "threshold": None,
                "anomaly_flags": None,
                "estimates": None,
                "anomaly_fraction": 1.0,
            }

        series = returns_df[stock].dropna()
        if series.empty:
            logger.warning(f"No data for stock {stock}. Skipping.")
            return None

        if use_isolation_forest:
            if stock in tuning_cache and not reoptimize:
                ticker_info = tuning_cache[stock]
            else:
                ticker_info = optimize_threshold_for_ticker(
                    series, weight_dict, stock, method, contamination=contamination_val
                )
            thresh = ticker_info["threshold"]
            anomaly_flags, estimates = apply_isolation_forest(
                series, threshold=thresh, contamination=contamination_val
            )
        elif use_kalman_filter:
            thresh = 7.0
            anomaly_flags, estimates = apply_kalman_filter(series, threshold=thresh)
        elif use_fixed_zscore:
            thresh = 3.0
            anomaly_flags, estimates = apply_fixed_zscore(series, threshold=thresh)
        else:
            logger.error("No anomaly detection method selected.")
            return None

        ticker_info = {
            "stock": stock,
            "threshold": thresh,
            "anomaly_flags": anomaly_flags,
            "estimates": estimates,
            "anomaly_fraction": float(anomaly_flags.mean()),
        }
        return ticker_info

    processed_info = Parallel(n_jobs=n_jobs)(
        delayed(process_ticker)(stock) for stock in returns_df.columns
    )

    new_anomalous_stocks: List[str] = []
    for info in processed_info:
        if info is None:
            continue
        tuning_cache[info["stock"]] = info
        if info["anomaly_fraction"] > max_anomaly_fraction:
            new_anomalous_stocks.append(info["stock"])

    # Update the tuning parameters cache.
    save_parameters_to_pickle(tuning_cache, cache_filename)

    # Merge previously cached anomalous stocks with newly detected ones (if not reoptimizing).
    if not reoptimize:
        all_anomalous = list(set(anomalous_cache).union(set(new_anomalous_stocks)))
    else:
        all_anomalous = new_anomalous_stocks

    # Save the updated list of anomalous stocks.
    save_parameters_to_pickle(all_anomalous, anomalous_cache_filename)

    valid_tickers = [
        stock for stock in returns_df.columns if stock not in all_anomalous
    ]
    logger.info(
        f"Removed {len(all_anomalous)} stocks due to high anomaly fraction: {sorted(all_anomalous)}"
        if all_anomalous
        else "No stocks were removed."
    )

    if plot and tuning_cache and all_anomalous:
        optimization_summary = list(tuning_cache.values())
        plot_optimization_summary(
            optimization_summary=optimization_summary,
            max_anomaly_fraction=max_anomaly_fraction,
        )
        plot_anomaly_overview(all_anomalous, tuning_cache, returns_df)

    return valid_tickers
