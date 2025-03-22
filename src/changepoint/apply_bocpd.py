from typing import List
import pandas as pd
import numpy as np

from changepoint.bocpd import bocpd
from changepoint.plot_bocpd import plot_bocpd_result


def detect_regime_change(
    feature_series: pd.Series,
    plot: bool = False,
    bullish_threshold: float = None,
    bearish_threshold: float = None,
) -> str:
    """
    Runs BOCPD on the given feature series (e.g. momentum, volatility, or rolling returns)
    and classifies the detected segments as bullish, neutral, or bearish.
    If `plot` is True, the BOCPD run-length matrix is plotted with the regime segments
    overlaid. Only the current regime (of the last segment) is returned.

    Parameters:
      feature_series: pandas Series of the feature (with a datetime index).
      plot: whether to plot the BOCPD results with regime boundaries and labels.
      bullish_threshold: (optional) Mean value above which a segment is bullish.
      bearish_threshold: (optional) Mean value below which a segment is bearish.

    Returns:
      A string: "Bullish", "Neutral", or "Bearish" for the current regime.
    """
    # Compute the run-length probability matrix using BOCPD.
    R = bocpd(
        feature_series,
        hazard_rate=1 / 15,
        mu0=0.001,
        alpha0=3.0,
        beta0=0.0005,
        kappa0=0.05,
    )

    T = len(feature_series)

    # If thresholds are not provided, calculate them from the feature's statistics.
    if bullish_threshold is None or bearish_threshold is None:
        mean_val = feature_series.mean()
        std_val = feature_series.std()
        bullish_threshold = mean_val + std_val
        bearish_threshold = mean_val - std_val

    # Extract most likely run-length for each observation (ignoring the initial prior row).
    most_likely_run = np.argmax(R[1:], axis=1)
    # Identify change points when the run-length resets (i.e. a drop in most likely run-length).
    change_points = list(np.where(np.diff(most_likely_run) < 0)[0] + 1)
    # Define regime boundaries (segment start and end indices).
    regime_boundaries = [0] + change_points + [T]

    regime_labels = []
    for i in range(len(regime_boundaries) - 1):
        start = regime_boundaries[i]
        end = regime_boundaries[i + 1]
        segment_mean = feature_series.iloc[start:end].mean()
        if segment_mean >= bullish_threshold:
            regime = "Bullish"
        elif segment_mean <= bearish_threshold:
            regime = "Bearish"
        else:
            regime = "Neutral"
        regime_labels.append(regime)

    # The current regime is the regime for the latest segment.
    current_regime = regime_labels[-1] if regime_labels else "Neutral"

    # If plotting is enabled, display the BOCPD run-length matrix with overlaid regime segments.
    if plot:
        dates = feature_series.index
        fig = plot_bocpd_result(
            R,
            feature_series=feature_series,
            series_title="Rolling 7-Day Mean Return",
            title="Bayesian Online Changepoint Detection Heatmap",
            dates=dates,
            regime_boundaries=regime_boundaries,
            regime_labels=regime_labels,
        )
        fig.show()

    return current_regime


def test_regime_detection_on_features(
    features: dict,
    plot: bool = False,
    bullish_threshold: float = None,
    bearish_threshold: float = None,
) -> dict:
    """
    Tests regime detection on multiple feature series. Each key in the dictionary
    is a feature name (e.g., "Momentum", "Volatility", "Rolling Returns") and the value
    is a pandas Series containing that feature.

    Parameters:
      features: dict mapping feature names to pandas Series.
      plot: whether to plot each feature's BOCPD output.
      bullish_threshold: optional threshold for bullish classification.
      bearish_threshold: optional threshold for bearish classification.

    Returns:
      regimes: dict mapping feature names to the detected current regime.
    """
    regimes = {}
    for feature_name, series in features.items():
        print(f"Processing feature: {feature_name}")
        regime = detect_regime_change(
            series,
            plot=plot,
            bullish_threshold=bullish_threshold,
            bearish_threshold=bearish_threshold,
        )
        regimes[feature_name] = regime
        print(f"Detected regime for {feature_name}: {regime}")
    return regimes
