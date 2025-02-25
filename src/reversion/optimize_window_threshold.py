from typing import Dict, List, Tuple
import numpy as np
import optuna
import pandas as pd

from models.optimizer_utils import (
    strategy_composite_score,
    strategy_performance_metrics,
)
from utils.z_scores import calculate_robust_zscores
from utils import logger


def optimize_robust_mean_reversion(
    returns_df: pd.DataFrame,
    objective_weights: dict,
    test_window_range: range = range(10, 31, 5),
    n_trials: int = 50,
    n_jobs: int = -1,
    refinement_margin: int = 5,
    refined_step: int = 1,
    refined_trials: int = 30,
) -> Tuple[Dict[str, float], optuna.study.Study]:
    """
    Optimize the rolling window and z_threshold using Optuna.
    This first performs a coarse grid search using the provided test_window_range before a refining search.

    Args:
        returns_df (pd.DataFrame): Log returns DataFrame.
        objective_weights (dict): Weights for objective metrics.
        test_window_range (range, optional): Coarse range of window sizes. Defaults to range(10, 31, 5).
        n_trials (int, optional): Number of trials for coarse search. Defaults to 50.
        n_jobs (int, optional): Parallel jobs. Defaults to -1.
        refinement_margin (int, optional): Margin around the best window for refined search. Defaults to 5.
        refined_step (int, optional): Step size for refined search. Defaults to 1.
        refined_trials (int, optional): Number of trials for refined search. Defaults to 30.

    Returns:
        Tuple[Dict[str, float], optuna.study.Study]: Best parameters and the study object.
    """
    # Coarse search
    coarse_study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    coarse_study.optimize(
        lambda trial: robust_mean_reversion_objective(
            trial,
            returns_df=returns_df,
            objective_weights=objective_weights,
            test_window_range=test_window_range,
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
    best_coarse_params = (
        coarse_study.best_trial.params
        if coarse_study.best_trial
        else {"window": 20, "z_threshold_positive": 1.5, "z_threshold_negative": 1.8}
    )

    # Adaptive refinement: refine the window parameter around the best coarse value
    best_window = best_coarse_params["window"]
    refined_lower = max(test_window_range.start, best_window - refinement_margin)
    # test_window_range.stop is exclusive so add 1 to include best_window + refinement_margin
    refined_upper = min(test_window_range.stop, best_window + refinement_margin + 1)
    refined_range = range(refined_lower, refined_upper, refined_step)

    # Create the refined study and seed it with the coarse best parameters.
    refined_study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    refined_study.enqueue_trial(best_coarse_params)

    refined_study.optimize(
        lambda trial: robust_mean_reversion_objective(
            trial,
            returns_df=returns_df,
            objective_weights=objective_weights,
            test_window_range=refined_range,
        ),
        n_trials=refined_trials,
        n_jobs=n_jobs,
    )

    # Choose the best parameters from the refined search if it improves upon the coarse result.
    if refined_study.best_value > (
        coarse_study.best_value if coarse_study else -float("inf")
    ):
        best_params = refined_study.best_trial.params
        return best_params, refined_study
    else:
        return best_coarse_params, coarse_study or refined_study


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
