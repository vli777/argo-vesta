from typing import Dict, Tuple
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import IsolationForest
from reversion.reversion_utils import compute_spread, johansen_test
from models.optimizer_utils import (
    strategy_composite_score,
    strategy_performance_metrics,
)
from utils.z_scores import calculate_robust_zscores


def cointegration_mean_reversion_objective(
    trial,
    prices_df: pd.DataFrame,
    objective_weights: dict,
    test_window_range: range = range(10, 31, 5),
    contamination: float = 0.05,
    penalize_outlier: bool = True,
) -> float:
    """
    Objective function for tuning cointegration-based mean reversion parameters.
    The procedure:
      1. Suggest a rolling window and z-score thresholds.
      2. Computes the cointegrated spread from prices.
      3. Optionally determines if the spread is unusual (temporary event etc.)
      4. Calculates robust z-scores on the spread.
      5. Generates signals and computes a composite performance score.
    
    Args:
        trial (optuna.Trial): The trial object.
        prices_df (pd.DataFrame): Log-price DataFrame.
        objective_weights (dict): Weighting for performance metrics.
        test_window_range (range): Rolling z-score window range.
        contamination (float): Isolation Forest outlier fraction.
        penalize_outlier (bool): Whether to penalize if current spread is an outlier.

    Returns:
        float: Composite performance score or large penalty if invalid.
    """
    # Suggest parameters.
    window = trial.suggest_int(
        "window",
        test_window_range.start,
        test_window_range.stop - 1,
        step=test_window_range.step,
    )
    z_threshold_negative = trial.suggest_float(
        "z_threshold_negative", 1.0, 3.0, step=0.1
    )
    z_threshold_positive = trial.suggest_float(
        "z_threshold_positive", 1.0, 3.0, step=0.1
    )

    # Compute cointegrating vector.
    coint_result = johansen_test(prices_df)
    if not coint_result.cointegration_found:
        # Penalize if cointegration isn't found.
        return -1e6
    eigenvector = coint_result.eigenvector
    spread = compute_spread(prices_df, eigenvector)  # Spread as a Series
    spread_df = pd.DataFrame(spread, columns=["spread"])
    
    # Optional outlier detection
    if penalize_outlier and len(spread_df) > 20:
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(spread_df.values)
        latest_flag = iso.predict(spread_df.values[-1].reshape(1, -1))[0]
        if latest_flag == -1:
            return -1e5  # Penalize trial with outlier spread

    # Compute robust z-scores on the spread.
    robust_z = calculate_robust_zscores(spread_df, window)  # Expects a DataFrame

    # Generate signals:
    #   - Long signal when spread is significantly low.
    #   - Short signal when spread is significantly high.
    signals = np.where(
        robust_z["spread"].values < -z_threshold_negative,
        (np.abs(robust_z["spread"].values) - z_threshold_negative) / z_threshold_negative,
        np.where(
            robust_z["spread"].values > z_threshold_positive,
            -((robust_z["spread"].values - z_threshold_positive) / z_threshold_positive),
            0,
        ),
    )
    signals_df = pd.DataFrame(signals, index=robust_z.index, columns=["signal"])
    # Shift signals to avoid look-ahead bias.
    positions_df = signals_df.shift(1).fillna(0)

    # Compute spread returns difference
    spread_returns = spread_df["spread"].diff().dropna()
    positions = positions_df["signal"].reindex(spread_returns.index).fillna(0)

    # Evaluate performance using custom metrics.
    metrics = strategy_performance_metrics(
        returns_df=spread_returns.to_frame("spread"),
        positions_df=positions.to_frame("signal"),
        objective_weights=objective_weights,
    )
    return strategy_composite_score(metrics, objective_weights=objective_weights)


def optimize_cointegration_mean_reversion(
    prices_df: pd.DataFrame,
    objective_weights: dict,
    test_window_range: range = range(10, 31, 5),
    n_trials: int = 50,
    n_jobs: int = -1,
    refinement_margin: int = 5,
    refined_step: int = 1,
    refined_trials: int = 30,
) -> Tuple[Dict[str, float], optuna.study.Study]:
    """
    Optimize cointegration-based mean reversion parameters using Optuna.
    Performs a coarse search over the test_window_range, then refines the window parameter.

    Args:
        prices_df (pd.DataFrame): Adjusted close price data for the cluster.
        returns_df (pd.DataFrame): Returns data for the cluster (used for performance evaluation).
        objective_weights (dict): Weights for the objective metrics.
        test_window_range (range, optional): Coarse range of window sizes. Defaults to range(10, 31, 5).
        n_trials (int, optional): Number of trials for the coarse search. Defaults to 50.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        refinement_margin (int, optional): Margin around the best window for refined search. Defaults to 5.
        refined_step (int, optional): Step size for the refined search. Defaults to 1.
        refined_trials (int, optional): Number of trials for the refined search. Defaults to 30.

    Returns:
        Tuple[Dict[str, float], optuna.study.Study]: The best parameters and the corresponding Optuna study.
    """
    # Coarse search
    coarse_study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    coarse_study.optimize(
        lambda trial: cointegration_mean_reversion_objective(
            trial,
            prices_df=prices_df,
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

    best_window = best_coarse_params["window"]
    refined_lower = max(test_window_range.start, best_window - refinement_margin)
    refined_upper = min(test_window_range.stop, best_window + refinement_margin + 1)
    refined_range = range(refined_lower, refined_upper, refined_step)

    refined_study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    refined_study.enqueue_trial(best_coarse_params)
    refined_study.optimize(
        lambda trial: cointegration_mean_reversion_objective(
            trial,
            prices_df=prices_df,
            objective_weights=objective_weights,
            test_window_range=refined_range,
        ),
        n_trials=refined_trials,
        n_jobs=n_jobs,
    )

    if refined_study.best_value > (
        coarse_study.best_value
        if coarse_study.best_value is not None
        else -float("inf")
    ):
        best_params = refined_study.best_trial.params
        return best_params, refined_study
    else:
        return best_coarse_params, coarse_study
