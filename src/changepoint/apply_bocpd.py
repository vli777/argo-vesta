from typing import List
import pandas as pd
import numpy as np

from changepoint.bocpd import bocpd
from changepoint.plot_bocpd import plot_bocpd_result
from changepoint.bocpd_optimize import get_bocpd_params


def detect_regime_change(
    feature_series: pd.Series,
    plot: bool = False,
    bullish_threshold: float = None,
    bearish_threshold: float = None,
    threshold_multiplier: float = 0.5,
    bocpd_params: dict = None,
) -> str:
    """
    Runs BOCPD on the given feature series using provided parameters and returns the current regime.
    Optionally plots the BOCPD run-length heatmap with regime overlays.
    """
    # Use default parameters if none provided.
    if bocpd_params is None:
        bocpd_params = {
            "hazard_rate0": 1 / 60,
            "mu0": 0.001,
            "kappa0": 0.05,
            "alpha0": 3.0,
            "beta0": 0.0005,
            "epsilon": 1e-8,
            "truncation_threshold": 1e-4,
            "rolling_window": 10,
        }
    else:
        if "agg_window" in bocpd_params:
            del bocpd_params["agg_window"]
        if "hazard_inv" in bocpd_params:
            bocpd_params["hazard_rate0"] = 1 / bocpd_params["hazard_inv"]
            del bocpd_params["hazard_inv"]
        if "threshold_multiplier" in bocpd_params:
            del bocpd_params["threshold_multiplier"]

    R_matrix = bocpd(data=feature_series, **bocpd_params)
    T = len(feature_series)
    # Set thresholds if not provided.
    if bullish_threshold is None or bearish_threshold is None:
        mean_val = feature_series.mean()
        stdev = feature_series.std()
        threshold = threshold_multiplier * stdev
        bullish_threshold = mean_val + threshold
        bearish_threshold = mean_val - threshold

    most_likely_run = np.argmax(R_matrix[1:], axis=1)
    change_points = list(np.where(np.diff(most_likely_run) < 0)[0] + 1)
    regime_boundaries = [0] + change_points + [T]

    per_segment_labels = []
    for i in range(len(regime_boundaries) - 1):
        start = regime_boundaries[i]
        end = regime_boundaries[i + 1]
        seg_mean = feature_series.iloc[start:end].mean()
        if seg_mean >= bullish_threshold:
            seg_label = "Bullish"
        elif seg_mean <= bearish_threshold:
            seg_label = "Bearish"
        else:
            seg_label = "Neutral"
        per_segment_labels.append(seg_label)

    current_regime = per_segment_labels[-1] if per_segment_labels else "Neutral"

    if plot:
        dates = feature_series.index
        fig = plot_bocpd_result(
            R_matrix,
            feature_series=feature_series,
            series_title="Aggregate Mean Return",
            series_label="Return",
            title="Bayesian Online Changepoint Detection Heatmap",
            dates=dates,
            regime_boundaries=regime_boundaries,
            regime_labels=per_segment_labels,
        )
        fig.show()

    return current_regime


def apply_bocpd(
    returns_df: pd.DataFrame,
    plot: bool = False,
    bullish_threshold: float = None,
    bearish_threshold: float = None,
    cache_dir: str = "optuna_cache",
    reoptimize: bool = False,
) -> str:
    """
    Aggregates a DataFrame of log returns internally and applies BOCPD regime detection
    using optimized parameters from cache (or via an Optuna study if not available).

    Returns:
      The current regime ("Bullish", "Neutral", or "Bearish").
    """
    # Load or optimize BOCPD parameters.
    params = get_bocpd_params(returns_df, cache_dir=cache_dir, reoptimize=reoptimize)
    # Use the optimized aggregation window (agg_window) for the cross-asset aggregation.
    aggregated_series = (
        returns_df.rolling(window=params["agg_window"]).mean().dropna().mean(axis=1)
    )

    current_regime = detect_regime_change(
        feature_series=aggregated_series,
        plot=plot,
        bullish_threshold=bullish_threshold,
        bearish_threshold=bearish_threshold,
        threshold_multiplier=params["threshold_multiplier"],
        bocpd_params=params,
    )
    return current_regime
