from typing import List
import pandas as pd
import numpy as np

from changepoint.bocpd import bocpd
from changepoint.plot_bocpd import plot_bocpd_result


def detect_regime_change(
    returns_df: pd.DataFrame,
    plot: bool,
    bullish_threshold: float = None,
    bearish_threshold: float = None,
) -> str:
    """
    Detect regime changes and classify segments as bullish, neutral, or bearish.
    If `plot` is True, the BOCPD run-length probability matrix is plotted with
    overlaid regime segments (vertical bands and labels). The function returns
    the regime label for the latest segment (i.e., the current regime).

    Parameters:
      returns_df: DataFrame where each column is a return series and the index is time.
      plot: Whether to plot the BOCPD run-length probability matrix with regime segments.
      bullish_threshold: (optional) Mean return above which a regime is considered bullish.
      bearish_threshold: (optional) Mean return below which a regime is considered bearish.

    Returns:
      current_regime: A string representing the regime of the most recent segment.
    """
    # Aggregate returns (e.g., averaging across assets)
    aggregated_returns = returns_df.mean(axis=1)

    # Compute the run-length probability matrix using BOCPD.
    R = bocpd(aggregated_returns, hazard_rate=1 / 50)

    T = len(aggregated_returns)

    # If thresholds are not provided, compute them from the overall data statistics.
    if bullish_threshold is None or bearish_threshold is None:
        mean_return = aggregated_returns.mean()
        std_return = aggregated_returns.std()
        bullish_threshold = mean_return + std_return
        bearish_threshold = mean_return - std_return

    # Extract the most likely run-length for each observation (ignoring row 0, the prior).
    most_likely_run = np.argmax(R[1:], axis=1)

    # Identify change points when the most likely run-length resets (i.e., a drop is detected).
    change_points = list(np.where(np.diff(most_likely_run) < 0)[0] + 1)
    # Define segment boundaries (start and end indices)
    regime_boundaries = [0] + change_points + [T]

    regime_labels = []
    for i in range(len(regime_boundaries) - 1):
        start = regime_boundaries[i]
        end = regime_boundaries[i + 1]
        segment_mean = aggregated_returns.iloc[start:end].mean()
        if segment_mean >= bullish_threshold:
            regime = "Bullish"
        elif segment_mean <= bearish_threshold:
            regime = "Bearish"
        else:
            regime = "Neutral"
        regime_labels.append(regime)

    # The current regime is the regime of the latest segment.
    current_regime = regime_labels[-1] if regime_labels else "Neutral"

    # If plotting is enabled, overlay the regime segments into the BOCPD plot.
    if plot:
        dates = returns_df.index
        fig = plot_bocpd_result(
            R,
            title="Bayesian Online Change Point Detection",
            dates=dates,
            regime_boundaries=regime_boundaries,
            regime_labels=regime_labels,
        )
        fig.show()

    return current_regime
