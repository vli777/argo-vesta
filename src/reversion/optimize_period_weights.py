from typing import Dict, List, Tuple
import numpy as np
import optuna
import pandas as pd

from models.optimizer_utils import (
    strategy_composite_score,
    strategy_performance_metrics,
)
from utils.logger import logger


def find_optimal_weights(
    daily_signals_df: pd.DataFrame,
    weekly_signals_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    objective_weights: dict,
    n_trials: int = 50,
    n_jobs: int = -1,
    group_id: str = None,
) -> Dict[str, float]:
    """
    Run Optuna to find the optimal weighting of daily and weekly signals.
    Returns a dictionary with both weight_daily and weight_weekly, ensuring they sum to 1.
    """
    study = optuna.create_study(
        study_name=f"reversion_weights_optimization_{group_id}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=max(5, n_trials // 10)),
    )
    study.optimize(
        lambda trial: reversion_weights_objective(
            trial, daily_signals_df, weekly_signals_df, returns_df, objective_weights
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
    if study.best_trial is None:
        logger.error("No valid optimization results found.")
        best_weights = {"weight_daily": 0.7, "weight_weekly": 0.3}
    else:
        best_weights = study.best_trial.params
        best_weights["weight_weekly"] = round(1.0 - best_weights["weight_daily"], 1)
    return best_weights


def reversion_weights_objective(
    trial,
    daily_signals_df: pd.DataFrame,
    weekly_signals_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    objective_weights: dict,
) -> float:
    """
    Optimize the weight adjustment for mean reversion signals.

    The idea is to start with an equal-weight baseline allocation. The daily and weekly signals,
    which are continuous adjustment factors, are combined using a weight parameter.
    The combined signal is then normalized relative to the maximum observed signal value
    (using 1 as the baseline reference). This normalized signal is used to adjust the baseline allocation.
    The adjusted weights are normalized to sum to 1, and the strategy is simulated.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        daily_signals_df (pd.DataFrame): Daily signals DataFrame.
        weekly_signals_df (pd.DataFrame): Weekly signals DataFrame.
        returns_df (pd.DataFrame): Log returns DataFrame.

    Returns:
        float: Composite score from the simulated strategy.
    """
    # Let the optimizer choose the weight for the daily signal.
    weight_daily = trial.suggest_float("weight_daily", 0.1, 1.0, step=0.1)
    weight_weekly = 1.0 - weight_daily

    # Combine daily and weekly signals (using the last row, e.g. the most recent signals).
    combined_signal = (
        weight_daily * daily_signals_df.iloc[-1]
        + weight_weekly * weekly_signals_df.iloc[-1]
    )

    # Define a baseline allocation: equal weight among assets with valid returns.
    valid_stocks = returns_df.dropna(axis=1, how="all").columns
    n = len(valid_stocks)
    baseline_allocation = pd.Series(1.0 / n, index=valid_stocks)

    # We assume the baseline signal is 1 (i.e. no adjustment).
    # Use the maximum value of the combined signal as the reference.
    reference_max = combined_signal.max()
    if reference_max <= 1:
        normalized_signal = (
            combined_signal - 1
        )  # no upward adjustment possible if all signals ≤ 1
    else:
        normalized_signal = (combined_signal - 1) / (reference_max - 1)

    # Set a sensitivity factor (you may tune this if desired).
    sensitivity = 1.0

    # Compute the adjustment factor.
    adjustment_factor = 1 + sensitivity * normalized_signal
    adjustment_factor = adjustment_factor.clip(lower=0)  # ensure nonnegative factors

    # Adjust the baseline allocation by these factors.
    prelim_weights = baseline_allocation * adjustment_factor
    adjusted_weights = prelim_weights / prelim_weights.sum()  # Normalize to sum to 1

    # For simulation, construct a positions DataFrame that uses these adjusted weights for each day.
    positions_df = pd.DataFrame(
        [adjusted_weights] * len(returns_df),
        index=returns_df.index,
        columns=valid_stocks,
    )
    # Shift positions to avoid lookahead bias.
    positions_df = positions_df.shift(1).bfill()

    # Simulate the strategy.
    metrics = strategy_performance_metrics(
        returns_df=returns_df[valid_stocks].reindex(positions_df.index),
        positions_df=positions_df,
        objective_weights=objective_weights,
    )

    return strategy_composite_score(metrics, objective_weights=objective_weights)
