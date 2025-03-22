import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from anomaly.isolation_forest import apply_isolation_forest
from anomaly.anomaly_utils import apply_fixed_zscore
from anomaly.kalman_filter import apply_kalman_filter
from utils.performance_metrics import kappa_ratio


def optimize_threshold_for_ticker(
    returns_series: pd.Series,
    weight_dict: dict,
    stock: str,
    method: str,
    contamination: float = "auto",
) -> dict:
    """
    Optimizes the anomaly detection threshold for a single ticker using Optuna.
    The objective function scores the threshold based on multiple factors (kappa,
    volatility penalties, and anomaly fraction). The search space is adjusted
    dynamically depending on the method:
        - IF: Isolation Forest (default range: 4.2 to 6.9)
        - KF: Kalman Filter (default range: 5.0 to 8.0)
        - Z-score: Fixed Z-score (default range: 2.0 to 4.0)

    Args:
        returns_series (pd.Series): Series of stock returns.
        weight_dict (dict): Weights for scoring components (e.g. "kappa", "stability").
        stock (str): Ticker symbol.
        method (str): One of "IF", "KF", or "Z-score".
        contamination (float): Contamination parameter for Isolation Forest.

    Returns:
        dict: Contains "stock", "threshold", "best_score", and "anomaly_fraction".
    """

    def objective(trial):
        # Adjust the search space and anomaly detection function based on method.
        if method == "IF":
            threshold = trial.suggest_float("threshold", 5.0, 8.0, step=0.1)
            anomaly_flags, _ = apply_isolation_forest(
                returns_series, threshold=threshold, contamination=contamination
            )
        elif method == "KF":
            threshold = trial.suggest_float("threshold", 5.0, 12.0, step=0.1)
            anomaly_flags, _ = apply_kalman_filter(returns_series, threshold=threshold)
        elif method == "Z-score":
            threshold = trial.suggest_float("threshold", 3.0, 10.0, step=0.1)
            anomaly_flags, _ = apply_fixed_zscore(returns_series, threshold=threshold)
        else:
            raise ValueError(f"Unsupported method: {method}")

        anomaly_fraction = anomaly_flags.mean()

        # Compute kappa ratio; if NaN, default to 0.
        kappa = kappa_ratio(returns_series, order=3)
        if np.isnan(kappa):
            kappa = 0.0

        # Compute EWMA volatility using a span equivalent to ~30 days.
        ewma_std_series = returns_series.ewm(span=30, min_periods=5).std()
        rolling_volatility = np.nanmean(ewma_std_series)
        median_volatility = np.nanmedian(ewma_std_series)
        if np.isnan(rolling_volatility):
            rolling_volatility = 0.0
        if np.isnan(median_volatility) or median_volatility == 0:
            median_volatility = 1.0

        # Rolling volatility computation (30-day window, min 5 observations).
        # rolling_std_series = returns_series.rolling(window=30, min_periods=5).std()
        # rolling_volatility = np.nanmean(rolling_std_series)
        # median_volatility = np.nanmedian(rolling_std_series)
        # if np.isnan(rolling_volatility):
        #     rolling_volatility = 0.0
        # if np.isnan(median_volatility) or median_volatility == 0:
        #     median_volatility = 1.0

        # Stability penalty: higher volatility reduces the score.
        stability_penalty = -rolling_volatility * 0.05

        # Extreme volatility penalty based on 95th percentile of daily pct changes.
        pct_changes = returns_series.pct_change().abs().dropna()
        if len(pct_changes) > 0:
            extreme_vol = np.percentile(pct_changes, 95)
            extreme_volatility_penalty = (
                -extreme_vol * 0.5 if not np.isnan(extreme_vol) else 0.0
            )
        else:
            extreme_volatility_penalty = 0.0

        # Meme stock penalty based on rolling z-scores.
        # Use EWMA for rolling mean and std for the z-score calculation.
        ewma_mean = returns_series.ewm(span=30, min_periods=5).mean()
        ewma_std = returns_series.ewm(span=30, min_periods=5).std()
        z_score_series = (
            ((returns_series - ewma_mean) / (ewma_std + 1e-8)).abs().fillna(0)
        )
        # rolling_mean = returns_series.rolling(window=30, min_periods=5).mean()
        # rolling_std = returns_series.rolling(window=30, min_periods=5).std()
        # z_score_series = (
        #     ((returns_series - rolling_mean) / (rolling_std + 1e-8)).abs().fillna(0)
        # )
        if len(z_score_series) > 0:
            perc_z = np.percentile(z_score_series, 95)
            meme_stock_penalty = -perc_z * 1.5 if not np.isnan(perc_z) else 0.0
        else:
            meme_stock_penalty = 0.0

        # Direct penalty for a higher anomaly fraction.
        anomaly_fraction_penalty = anomaly_fraction * 10

        # Composite score: higher is better.
        composite_score = (
            weight_dict["kappa"] * kappa
            + weight_dict["stability"] * stability_penalty
            + extreme_volatility_penalty
            + meme_stock_penalty
            - anomaly_fraction_penalty
        )

        # Normalize by volatility ratio.
        vol_penalty_ratio = (
            rolling_volatility / median_volatility if median_volatility > 0 else 1.0
        )
        composite_score /= vol_penalty_ratio + 1e-8

        print(
            f"[DEBUG] Stock: {stock}, Method: {method}, Threshold: {threshold:.2f}, "
            f"Score: {composite_score:.4f}, Anomaly Fraction: {anomaly_fraction:.4f}"
        )

        trial.set_user_attr("anomaly_fraction", anomaly_fraction)
        trial.report(composite_score, step=int(threshold * 10))
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return composite_score

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=50, timeout=60)

    # Use a default threshold if optimization fails.
    if not study.best_trial or study.best_value == -np.inf:
        default_threshold = 5.0 if method == "IF" else (7.0 if method == "KF" else 3.0)
        return {
            "stock": stock,
            "threshold": default_threshold,
            "best_score": 0.0,
            "anomaly_fraction": 1.0,
        }

    best_threshold = study.best_trial.params["threshold"]
    best_anomaly_fraction = study.best_trial.user_attrs.get("anomaly_fraction", 0.0)
    best_composite_score = study.best_value

    return {
        "stock": stock,
        "threshold": best_threshold,
        "best_score": best_composite_score,
        "anomaly_fraction": best_anomaly_fraction,
    }
