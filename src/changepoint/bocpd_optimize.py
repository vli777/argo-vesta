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
    cache_filename = cache_path / "bocpd_optuna_params.pkl"

    cached_params = load_parameters_from_pickle(cache_filename) or {}
    required_keys = [
        "agg_window",
        "rolling_window",
        "hazard_inv",
        "mu0",
        "kappa0",
        "alpha0",
        "beta0",
    ]
    if not reoptimize and all(key in cached_params for key in required_keys):
        logger.info("Using cached BOCPD parameters.")
        return cached_params
    else:
        logger.info("Optimizing BOCPD parameters...")
        study = optuna.create_study(direction="maximize")
        objective = create_bocpd_objective(returns_df)
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        logger.info("Optimized parameters: %s", best_params)
        save_parameters_to_pickle(best_params, cache_filename)
        return best_params


def create_bocpd_objective(returns_df: pd.DataFrame) -> callable:
    """
    Returns an objective function that optimizes parameters for BOCPD.
    It aggregates the returns using an aggregation window (agg_window) and then
    runs a measure (e.g. Sharpe ratio) on scaled returns based on regime labels.
    """

    def objective(trial):
        # Optimize the aggregation window size (for converting the returns DF)
        agg_window = trial.suggest_int("agg_window", 3, 20)
        # Optimize the BOCPD internal window size (renamed as rolling_window for consistency)
        rolling_window = trial.suggest_int("rolling_window", 5, 50)
        hazard_inv = trial.suggest_float("hazard_inv", 5, 100, log=True)
        mu0 = trial.suggest_float("mu0", 0.0005, 0.002)
        kappa0 = trial.suggest_float("kappa0", 0.01, 0.1)
        alpha0 = trial.suggest_float("alpha0", 2.5, 4.0)
        beta0 = trial.suggest_float("beta0", 0.0003, 0.0008)

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
        regime_labels = compute_regime_labels(aggregated_series, R_matrix)
        # Scale returns based on regime labels.
        scaled_returns = []
        for label, val in zip(regime_labels, aggregated_series):
            scale = 1.5 if label == "Bullish" else 0.5 if label == "Bearish" else 1.0
            scaled_returns.append(val * scale)
        scaled_returns = np.array(scaled_returns)
        sharpe = scaled_returns.mean() / (scaled_returns.std() + 1e-6)
        return sharpe

    return objective
