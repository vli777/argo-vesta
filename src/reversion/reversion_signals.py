import numpy as np
import pandas as pd


def compute_stateful_signal_with_decay(
    series: pd.Series,
    params: dict,
    target_decay: float = 0.5,
    reset_factor: float = 0.5,
    sensitivity: float = 1.0,
    baseline: float = 1.0,
) -> pd.Series:
    """
    Compute a continuous adjustment factor for a ticker's allocation based on a stateful signal.

    The signal is generated using rolling z-scores and is triggered as follows:
      - If the z-score exceeds z_threshold_positive, the state is set to -1 (overbought/short),
        which will reduce the allocation.
      - If the z-score falls below -z_threshold_negative, the state is set to +1 (oversold/long),
        which will increase the allocation.
      - The state decays over time, and resets when the z-score falls back below a fraction of the trigger threshold.

    The final adjustment factor is computed as:
         adjustment_factor = baseline * (1 + sensitivity * (state * signal_magnitude * decay_multiplier))

    A baseline of 1 means no change from the current allocation. Values above 1 boost the allocation;
    values below 1 reduce it (with strongly overbought conditions potentially driving it toward 0).

    Args:
        series (pd.Series): Price or return series.
        params (dict): Must contain:
            - "window": rolling window size.
            - "z_threshold_positive": threshold for triggering an overbought state.
            - "z_threshold_negative": threshold for triggering an oversold state.
        target_decay (float): Fraction of the original signal remaining after the optimal window.
        reset_factor (float): Factor to derive the reset threshold from the trigger threshold.
        sensitivity (float): How strongly the raw signal affects the adjustment.
        baseline (float): The baseline allocation (default 1.0).

    Returns:
        pd.Series: A time series of allocation adjustment factors.
    """
    window = int(params.get("window", 20))
    trigger_threshold_pos = params.get("z_threshold_positive", 1.5)
    trigger_threshold_neg = params.get("z_threshold_negative", 1.5)

    # Define reset thresholds.
    reset_threshold_pos = trigger_threshold_pos * reset_factor
    reset_threshold_neg = trigger_threshold_neg * reset_factor

    # Compute rolling z-scores.
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = (
        series.rolling(window=window, min_periods=window).std().replace(0, np.nan)
    )
    z_scores = (series - rolling_mean) / rolling_std
    # Initialize arrays.
    state = np.zeros(len(series))  # 0: neutral, -1: overbought, +1: oversold.
    state_age = np.zeros(len(series))
    raw_signal = np.zeros(len(series))

    # Iterate over the series.
    for i in range(1, len(series)):
        if np.isnan(z_scores.iloc[i]):
            continue

        if state[i - 1] == 0:
            if z_scores.iloc[i] > trigger_threshold_pos:
                state[i] = -1  # Overbought (signal to reduce weight).
                state_age[i] = 0
            elif z_scores.iloc[i] < -trigger_threshold_neg:
                state[i] = 1  # Oversold (signal to increase weight).
                state_age[i] = 0
        else:
            # Continue previous state.
            state[i] = state[i - 1]
            state_age[i] = state_age[i - 1] + 1

            # Reset the state if the z-score falls back below the reset threshold.
            if state[i] == -1 and z_scores.iloc[i] < reset_threshold_pos:
                state[i] = 0
                state_age[i] = 0
            elif state[i] == 1 and z_scores.iloc[i] > -reset_threshold_neg:
                state[i] = 0
                state_age[i] = 0

        # Compute adaptive decay rate based on realized volatility
        # rolling_vol = rolling_std.copy()
        # adaptive_decay_rate = target_decay ** (rolling_vol / rolling_vol.mean())
        # decay_multiplier = (
        #     adaptive_decay_rate.iloc[i] ** state_age[i] if state[i] != 0 else 0
        # )

        # Compute decay multiplier proportional to the series optimal reversion window.
        if state[i] != 0:
            decay_multiplier = target_decay ** (state_age[i] / window)
        else:
            decay_multiplier = 0

        # Determine magnitude
        thresh = (
            trigger_threshold_pos
            if state[i] == -1
            else trigger_threshold_neg if state[i] == 1 else 0
        )
        signal_magnitude = (
            abs(z_scores.iloc[i]) if abs(z_scores.iloc[i]) >= thresh else 0
        )
        raw_signal[i] = state[i] * signal_magnitude * decay_multiplier

        # Debug output for nonzero states.
        # if state[i] != 0:
        #     print(
        #         f"{series.name} @ {series.index[i]}: z_score={z_scores.iloc[i]:.2f}, "
        #         f"state={state[i]}, age={state_age[i]}, raw_signal={raw_signal[i]:.2f}"
        #     )

    # Compute the final adjustment factor.
    # A value of baseline means no change; values above baseline increase allocation,
    # values below baseline reduce allocation.
    adjustment_factor = baseline * (1 + sensitivity * raw_signal)
    adjustment_factor = np.clip(adjustment_factor, 0, None)  # Ensure non-negative.
    return pd.Series(adjustment_factor, index=series.index)


def compute_ticker_stateful_signals(
    ticker_series: pd.Series,
    params: dict,
    target_decay: float = 0.5,
    reset_factor: float = 0.5,
) -> dict:
    """
    Compute stateful signals for a ticker on both daily and weekly data.

    Returns:
        dict: {
            "daily": pd.Series,
            "weekly": pd.Series
        }
    """
    # Build daily parameters dictionary.
    daily_params = {
        "window": int(params.get("window_daily", 20)),
        "z_threshold_positive": params.get("z_threshold_daily_positive", 1.5),
        "z_threshold_negative": params.get("z_threshold_daily_negative", 1.5),
    }
    daily_signal = compute_stateful_signal_with_decay(
        ticker_series,
        daily_params,
        target_decay=target_decay,
        reset_factor=reset_factor,
    )

    # Build weekly parameters dictionary.
    weekly_series = ticker_series.resample("W").last()
    weekly_params = {
        "window": int(params.get("window_weekly", 5)),
        "z_threshold_positive": params.get("z_threshold_weekly_positive", 1.5),
        "z_threshold_negative": params.get("z_threshold_weekly_negative", 1.5),
    }
    weekly_signal = compute_stateful_signal_with_decay(
        weekly_series,
        weekly_params,
        target_decay=target_decay,
        reset_factor=reset_factor,
    )

    return {"daily": daily_signal, "weekly": weekly_signal}


def compute_group_stateful_signals(
    group_returns: pd.DataFrame,
    tickers: list,
    params: dict,
    target_decay: float = 0.5,
    reset_factor: float = 0.5,
) -> dict:
    """
    Given a group of tickers and their returns, compute stateful signals for each ticker.

    Returns:
        dict: Mapping from ticker to its signals, e.g.
            {
                "AAPL": {"daily": {date: signal, ...}, "weekly": {date: signal, ...}},
                "MSFT": {...},
                ...
            }
    """
    signals = {}
    for ticker in tickers:
        series = group_returns[ticker].dropna()
        if series.empty:
            signals[ticker] = {"daily": {}, "weekly": {}}
        else:
            signals[ticker] = compute_ticker_stateful_signals(
                series, params, target_decay=target_decay, reset_factor=reset_factor
            )
    return signals
