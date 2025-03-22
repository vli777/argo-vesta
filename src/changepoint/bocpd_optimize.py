import json
import optuna
import numpy as np
import pandas as pd
from pathlib import Path

from changepoint.bocpd import bocpd
from changepoint.bocpd_utils import (
    compute_regime_labels,
)
from utils import logger
from utils.caching_utils import load_parameters_from_pickle, save_parameters_to_pickle


def get_bocpd_params(
    returns_df: pd.DataFrame, cache_dir: str = "optuna_cache", reoptimize: bool = False
) -> dict:
    """
    Attempts to load optimized BOCPD parameters from cache.
    If not available or reoptimize is True, runs an Optuna study to optimize parameters.
    The optimized dictionary includes both the aggregation window and BOCPD parameters.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_filename = cache_path / "bocpd_params.pkl"

    cached_params = load_parameters_from_pickle(cache_filename) or {}
    required_keys = [
        "agg_window",
        "rolling_window",
        "hazard_inv",
        "mu0",
        "kappa0",
        "alpha0",
        "beta0",
        "threshold_multiplier",
    ]
    if not reoptimize and all(key in cached_params for key in required_keys):
        logger.info("Using cached BOCPD parameters.")
        return cached_params
    else:
        logger.info("Optimizing BOCPD parameters...")
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        objective = create_bocpd_objective(returns_df)
        study.optimize(objective, n_trials=100)
        best_params = study.best_params
        logger.info("Optimized parameters: %s", best_params)
        save_parameters_to_pickle(best_params, cache_filename)
        return best_params


def compute_penalty(
    label_series,
    aggregated_series,
    known_bearish_periods,
    penalty_strength=1.0,
    coverage_threshold=0.8,
):
    """
    Computes a penalty based on how well known bearish periods are detected.
    Only periods with sufficient data coverage (>= coverage_threshold) are penalized.

    Parameters:
      label_series: pandas Series of regime labels aligned with aggregated_series.index.
      aggregated_series: pandas Series with a DatetimeIndex.
      known_bearish_periods: list of tuples, each (start_date, end_date) where dates are strings or Timestamps.
      penalty_strength: scaling factor for the penalty.
      coverage_threshold: minimum fraction of the known period that must be covered by data to apply a penalty.

    Returns:
      Total penalty (float) to subtract from your objective metric.
    """
    penalty = 0.0
    data_start = aggregated_series.index.min()
    data_end = aggregated_series.index.max()

    for period in known_bearish_periods:
        # Parse dates (if not already Timestamps)
        start_date = pd.Timestamp(period["start_date"])
        end_date = pd.Timestamp(period["end_date"])

        # Check if there's any overlap with our data
        if end_date < data_start or start_date > data_end:
            continue

        # Determine the overlapping window
        overlap_start = max(start_date, data_start)
        overlap_end = min(end_date, data_end)

        # Calculate total days in the known period and in the overlap
        known_days = (end_date - start_date).days + 1
        overlap_days = (overlap_end - overlap_start).days + 1

        # Only consider this period if sufficient coverage exists
        if overlap_days / known_days < coverage_threshold:
            continue

        # Build mask for the overlapping window
        mask = (label_series.index >= overlap_start) & (
            label_series.index <= overlap_end
        )
        n_days_in_window = mask.sum()
        if n_days_in_window == 0:
            continue

        window_labels = label_series[mask]
        n_bearish = (window_labels == "Bearish").sum()
        fraction_bearish = n_bearish / n_days_in_window

        # Penalize if the fraction of bearish days is below 100%
        penalty += penalty_strength * (1.0 - fraction_bearish)

    return penalty


def create_bocpd_objective(returns_df: pd.DataFrame) -> callable:
    """
    Returns an objective function that optimizes parameters for BOCPD.
    It aggregates the returns using an aggregation window (agg_window) and then
    runs a measure (e.g. Sharpe ratio) on scaled returns based on regime labels.
    """

    def objective(trial):
        # Optimize the aggregation window size (for converting the returns DF)
        agg_window = trial.suggest_int("agg_window", 3, 120)
        # Optimize the BOCPD internal window size (renamed as rolling_window for consistency)
        rolling_window = trial.suggest_int("rolling_window", 5, 50)
        hazard_inv = trial.suggest_float("hazard_inv", 5, 100, log=True)
        mu0 = trial.suggest_float("mu0", 0.0005, 0.002)
        kappa0 = trial.suggest_float("kappa0", 0.01, 0.1)
        alpha0 = trial.suggest_float("alpha0", 2.5, 4.0)
        beta0 = trial.suggest_float("beta0", 0.0003, 0.0008)
        # Optimize the regime classification threshold multiplier
        threshold_multiplier = trial.suggest_float("threshold_multiplier", 0.3, 1.0)

        # Aggregate returns: compute rolling mean per asset, then cross-asset mean.
        aggregated_series = (
            returns_df.rolling(window=agg_window).mean().dropna().mean(axis=1)
        )

        # Build BOCPD parameter dictionary using consistent key names.
        bocpd_params = {
            "hazard_rate0": 1 / hazard_inv,  # computed directly here
            "rolling_window": rolling_window,
            "mu0": mu0,
            "kappa0": kappa0,
            "alpha0": alpha0,
            "beta0": beta0,
            "epsilon": 1e-8,
            "truncation_threshold": 1e-4,
        }

        R_matrix = bocpd(data=aggregated_series, **bocpd_params)
        regime_labels = compute_regime_labels(
            aggregated_series, R_matrix, threshold_multiplier
        )
        # Scale returns based on regime labels.
        scaled_returns = []
        for label, val in zip(regime_labels, aggregated_series):
            scale = 1.5 if label == "Bullish" else 0.5 if label == "Bearish" else 1.0
            scaled_returns.append(val * scale)
        scaled_returns = np.array(scaled_returns)
        sharpe = scaled_returns.mean() / (scaled_returns.std() + 1e-6)

        file_path = Path(__file__).parent / "known_bearish_periods.json"
        with file_path.open("r") as f:
            known_bearish_periods = json.load(f)

        # Convert regime_labels into a pandas Series aligned with aggregated_series.index
        label_series = pd.Series(regime_labels, index=aggregated_series.index)

        penalty = compute_penalty(
            label_series,
            aggregated_series,
            known_bearish_periods,
            penalty_strength=1.0,
            coverage_threshold=0.8,
        )
        score = sharpe - penalty
        return score

    return objective
