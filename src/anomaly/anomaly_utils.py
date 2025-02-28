import numpy as np
import pandas as pd


def detect_meme_stocks(
    df: pd.DataFrame, lookback_days: int = 90, crash_days: int = 30
) -> set:
    """
    Identify meme/manipulated stocks based on extreme price increases and sudden crashes.

    Args:
        df (pd.DataFrame): DataFrame with stock prices (columns = stock tickers, rows = dates).
        lookback_days (int): Number of days to check for extreme price jumps.
        crash_days (int): Number of days to check for post-spike crashes.

    Returns:
        set: A set of stock tickers classified as meme/manipulated stocks.
    """
    # Calculate price multiples over the lookback period (e.g., 3 months)
    price_multiple = df.iloc[-1] / df.shift(lookback_days).iloc[-1]

    # Threshold: Any stock in the **top 1% percentile** for price increase
    threshold = price_multiple.quantile(0.99)
    high_flyers = set(price_multiple[price_multiple > threshold].index)

    # Detect stocks that crashed **50%+ from their recent peak in the last crash_days**
    rolling_max = df.rolling(crash_days).max()
    drawdown = df.iloc[-1] / rolling_max.iloc[-1]
    crashing_stocks = set(drawdown[drawdown < 0.5].index)  # Stocks that lost 50%+

    # Combine both anomaly sets
    meme_candidates = high_flyers | crashing_stocks  # Union of sets
    return meme_candidates


def get_cache_filename(method: str) -> str:
    """Return the correct cache filename based on the anomaly detection method."""
    cache_map = {
        "IF": "optuna_cache/anomaly_thresholds_IF.pkl",
        "KF": "optuna_cache/anomaly_thresholds_KF.pkl",
        "Z-score": "optuna_cache/anomaly_thresholds_Z.pkl",
    }
    return cache_map.get(
        method, "optuna_cache/anomaly_thresholds_IF.pkl"
    )  # Default to IF


def apply_fixed_zscore(series: pd.Series, threshold: float = 3.0):
    """Detect anomalies using a fixed Z-score threshold.

    Returns:
        anomaly_flags (pd.Series): Boolean series indicating anomalies.
        estimates (pd.Series): In this case, simply the original series.
    """
    residuals = (series - series.mean()) / series.std()
    anomaly_flags = np.abs(residuals) > threshold
    return anomaly_flags, series.copy()


def update_tuning_cache(cache: dict, new_index: pd.Index) -> dict:
    """
    Update the tuning cache so that each asset's 'estimates' Series is reindexed
    to match the new index from the returns DataFrame.

    Args:
        cache (dict): The cached tuning parameters (e.g., loaded from anomaly_thresholds_IF.pkl).
                      Each asset's entry is expected to be a dict with an 'estimates' key containing a pd.Series.
        new_index (pd.Index): The new DateTimeIndex from the latest returns DataFrame.

    Returns:
        dict: The updated cache dictionary.
    """
    updated_cache = {}
    for stock, info in cache.items():
        if isinstance(info, dict):
            for key in ["anomaly_flags", "estimates"]:
                if key in info and isinstance(info[key], pd.Series):
                    info[key] = info[key].reindex(new_index, fill_value=np.nan)
        updated_cache[stock] = info
    return updated_cache
