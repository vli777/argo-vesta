from typing import Dict, List, Tuple
import numpy as np
import optuna
import pandas as pd

from models.optimizer_utils import (
    strategy_composite_score,
    strategy_performance_metrics,
)
from utils.z_scores import calculate_robust_zscores
from utils.logger import logger


def optimize_robust_mean_reversion(
    returns_df: pd.DataFrame,
    objective_weights: dict,
    test_window_range: range = range(10, 31, 5),
    n_trials: int = 50,
    n_jobs: int = -1,
) -> Tuple[Dict[str, float], optuna.study.Study]:
    """
    Optimize the rolling window and z_threshold using Optuna.
    This version does not write group-level keys to the global cache.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        test_window_range (range): Range of window sizes.
        n_trials (int, optional): Number of trials. Defaults to 50.
        n_jobs (int, optional): Parallel jobs. Defaults to -1.
        reoptimize (bool): Force reoptimization.
        group_id (str): Identifier for the cluster (not used for caching here).

    Returns:
        Tuple[Dict[str, float], optuna.study.Study]: Best parameters and the study.
    """
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda trial: robust_mean_reversion_objective(
            trial,
            returns_df=returns_df,
            objective_weights=objective_weights,
            test_window_range=test_window_range,
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
    best_params = (
        study.best_trial.params
        if study.best_trial
        else {"window": 20, "z_threshold_positive": 1.5, "z_threshold_negative": 1.8}
    )
    return best_params, study


def robust_mean_reversion_objective(
    trial,
    returns_df: pd.DataFrame,
    objective_weights: dict,
    test_window_range: range = range(10, 31, 5),
) -> float:
    """
    Objective function for optimizing the robust mean reversion parameters.
    The trial suggests a rolling window size and a z_threshold.
    The resulting signals are used (with a one-day shift to avoid lookahead)
    to simulate a simple strategy; the cumulative return and sharpe ratio is maximized.
    """

    def suggest_window(trial, window_range: range):
        return trial.suggest_int(
            "window", window_range.start, window_range.stop - 1, step=window_range.step
        )

    window = suggest_window(trial, test_window_range)
    z_threshold_negative = trial.suggest_float(
        "z_threshold_negative", 1.0, 3.0, step=0.1
    )
    z_threshold_positive = trial.suggest_float(
        "z_threshold_positive", 1.0, 3.0, step=0.1
    )

    robust_z = calculate_robust_zscores(returns_df, window)
    # Generate signals:
    #  - If the z-score is below -z_threshold_negative, signal long (1).
    #  - If it is above z_threshold_positive, signal short (-1).
    #  - Otherwise, signal 0.
    # Compute continuous signals:
    signals = np.where(
        robust_z.values < -z_threshold_negative,
        (np.abs(robust_z.values) - z_threshold_negative) / z_threshold_negative,
        np.where(
            robust_z.values > z_threshold_positive,
            -((robust_z.values - z_threshold_positive) / z_threshold_positive),
            0,
        ),
    )

    signals_df = pd.DataFrame(signals, index=robust_z.index, columns=robust_z.columns)

    positions_df = signals_df.shift(1).fillna(0)
    valid_stocks = returns_df.dropna(axis=1, how="all").columns
    positions_df = positions_df[valid_stocks]
    aligned_returns = returns_df[valid_stocks].reindex(positions_df.index)

    metrics = strategy_performance_metrics(
        returns_df=aligned_returns,
        positions_df=positions_df,
        objective_weights=objective_weights,
    )

    return strategy_composite_score(metrics, objective_weights=objective_weights)
