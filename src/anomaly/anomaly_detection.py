from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from joblib import Parallel, delayed


from anomaly.plot_anomalies import plot_anomaly_overview
from anomaly.isolation_forest import apply_isolation_forest
from anomaly.plot_optimization_summary import plot_optimization_summary
from anomaly.kalman_filter import apply_kalman_filter
from anomaly.anomaly_utils import (
    apply_fixed_zscore,
    get_cache_filename,
    update_tuning_cache,
)
from anomaly.optimize_anomaly_threshold import optimize_threshold_for_ticker
from utils.logger import logger
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


CACHE_EXPIRATION_DAYS = 90  # Cache expires every quarter (90 days)


def remove_anomalous_stocks(
    returns_df: pd.DataFrame,
    weight_dict: Optional[Dict[str, float]] = None,
    plot: bool = False,
    n_jobs: int = -1,
    reoptimize: bool = False,
    contamination: Union[float, str, None] = None,
    max_anomaly_fraction: float = 0.01,
) -> List[str]:
    """
    Filters out stocks with anomalous returns using Isolation Forest (IF),
    Kalman Filter (KF), or fixed Z-score. Uses per-ticker optimization and caches
    method-specific parameters as well as a list of stocks previously flagged as anomalous.
    The anomalous stock list is saved to 'optuna_cache/anomalous_assets.pkl' so that in future runs,
    stocks already determined to be anomalous can be skipped regardless of the filter used.

    The cache for anomalous stocks expires every quarter (90 days).

    Args:
        returns_df (pd.DataFrame): DataFrame where each column corresponds to a stock's returns.
        weight_dict (Optional[Dict[str, float]]): Weights for optimization criteria.
        plot (bool): If True, generate anomaly and optimization plots.
        n_jobs (int): Number of parallel jobs to run.
        reoptimize (bool): If True, bypass cache and reoptimize thresholds.
        contamination (Union[float, str, None]): Contamination parameter for Isolation Forest.
        max_anomaly_fraction (float): Maximum fraction of anomalous data allowed per stock.

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
    if returns_df is not None:
        tuning_cache = update_tuning_cache(tuning_cache, returns_df.index)

    # Load the anomalous assets cache with expiration check.
    anomalous_cache_filename = "optuna_cache/anomalous_assets.pkl"
    anomalous_cache_data: Dict[str, Any] = (
        {}
        if reoptimize
        else load_parameters_from_pickle(anomalous_cache_filename) or {}
    )

    # Check if the cache is expired
    cache_timestamp = anomalous_cache_data.get("timestamp", datetime.min)
    if datetime.now() - cache_timestamp > timedelta(days=CACHE_EXPIRATION_DAYS):
        anomalous_cache: List[str] = []
        print("Anomalous cache expired. Starting fresh.")
    else:
        anomalous_cache: List[str] = anomalous_cache_data.get("anomalous_stocks", [])

    def process_ticker(stock: str) -> Optional[Dict[str, Any]]:
        if (stock in anomalous_cache) and not reoptimize:
            ticker_info = tuning_cache.get(stock)
            if ticker_info is not None:
                return ticker_info

        series = returns_df[stock].dropna()
        if series.empty:
            logger.warning(f"No data for stock {stock}. Skipping.")
            return None

        if use_isolation_forest:
            if stock in tuning_cache and not reoptimize:
                ticker_info = tuning_cache.get(stock, {})
            else:
                ticker_info = optimize_threshold_for_ticker(
                    series, weight_dict, stock, method, contamination=contamination_val
                )

            thresh = ticker_info.get("threshold", 6.0)
            if thresh is None:
                logger.error(f"Threshold for stock {stock} is None. Defaulting to 6.0.")
                thresh = 6.0
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

    # Collect anomaly fractions for all processed tickers.
    anomaly_fractions = []
    new_anomalous_stocks: List[str] = []
    for info in processed_info:
        if info is None:
            continue
        tuning_cache[info["stock"]] = info
        anomaly_fractions.append(info["anomaly_fraction"])

    # Determine a dynamic threshold.
    if anomaly_fractions:
        median_fraction = np.median(anomaly_fractions)
        mad_fraction = np.median(np.abs(anomaly_fractions - median_fraction))
        # Use median + 1.5 * MAD as a threshold (you can adjust the multiplier)
        dynamic_threshold = median_fraction + 1.5 * mad_fraction
        logger.info(f"Dynamic max anomaly fraction set to: {dynamic_threshold:.4f}")
    else:
        dynamic_threshold = max_anomaly_fraction  # fallback

    for info in processed_info:
        if info is None:
            continue
        if info["anomaly_fraction"] > dynamic_threshold:
            new_anomalous_stocks.append(info["stock"])

    # Update the tuning parameters cache.
    save_parameters_to_pickle(tuning_cache, cache_filename)

    # Merge previously cached anomalous stocks with newly detected ones (if not reoptimizing).
    if not reoptimize:
        all_anomalous = list(set(anomalous_cache).union(set(new_anomalous_stocks)))
    else:
        all_anomalous = new_anomalous_stocks

    # Save the updated list of anomalous stocks with a timestamp.
    save_parameters_to_pickle(
        {"timestamp": datetime.now(), "anomalous_stocks": all_anomalous},
        anomalous_cache_filename,
    )

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
