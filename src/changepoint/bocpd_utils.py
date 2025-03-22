import numpy as np
import pandas as pd


def compute_regime_labels(feature_series: pd.Series, R) -> list:
    """
    Computes regime labels per observation given a feature series and BOCPD matrix.
    Returns a list of labels ("Bullish", "Bearish", "Neutral") for each observation.
    """
    T = len(feature_series)
    most_likely_run = np.argmax(R[1:], axis=1)
    change_points = list(np.where(np.diff(most_likely_run) < 0)[0] + 1)
    regime_boundaries = [0] + change_points + [T]

    # Default thresholds: using half the standard deviation around the mean.
    mean_val = feature_series.mean()
    stdev = feature_series.std()
    bullish_threshold = mean_val + 0.5 * stdev
    bearish_threshold = mean_val - 0.5 * stdev

    labels = []
    for i in range(len(regime_boundaries) - 1):
        start = regime_boundaries[i]
        end = regime_boundaries[i + 1]
        seg_mean = feature_series.iloc[start:end].mean()
        if seg_mean >= bullish_threshold:
            labels.extend(["Bullish"] * (end - start))
        elif seg_mean <= bearish_threshold:
            labels.extend(["Bearish"] * (end - start))
        else:
            labels.extend(["Neutral"] * (end - start))
    return labels
