import optuna
import pandas as pd

from reversion.reversion_utils import (
    adjust_allocation_with_mean_reversion,
    propagate_signals_by_similarity,
)
from models.optimizer_utils import (
    strategy_composite_score,
    strategy_performance_metrics,
)


def tune_reversion_alpha(
    returns_df: pd.DataFrame,
    baseline_allocation: pd.Series,
    composite_signals: dict,
    group_mapping: dict,
    objective_weights: dict,
    n_trials: int = 50,
    patience: int = 10,  # Stop early if no improvement
    hv_window: int = 50,
) -> float:

    # Compute historical realized volatility
    historical_vol = returns_df.rolling(window=hv_window).std().mean().mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=patience),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        lambda trial: alpha_objective(
            trial,
            returns_df,
            historical_vol,
            baseline_allocation,
            composite_signals,
            group_mapping,
            objective_weights,
        ),
        n_trials=n_trials,
        n_jobs=-1,  # Parallelize across all cores
    )

    # Get the best base_alpha
    best_base_alpha = study.best_params["base_alpha"]
    print(f"Optimal base_alpha: {best_base_alpha}")

    return best_base_alpha


def alpha_objective(
    trial,
    returns_df: pd.DataFrame,
    historical_vol: float,
    baseline_allocation: pd.Series,
    composite_signals: dict,
    group_mapping: dict,
    objective_weights: dict,
    rebalance_period: int = 30,
    hv_window: int = 50,
) -> float:
    """
    Optuna objective function for tuning base_alpha.
    Now, at each rebalance, realized volatility is computed over hv_window periods
    and the base_alpha is scaled by 1/(1 + realized_vol) to yield an effective alpha.
    """
    # Set prior mean based on historical volatility.
    mean_alpha = min(0.1, max(0.2, 0.5 * historical_vol))
    low_alpha = max(0.01, 0.5 * mean_alpha)
    high_alpha = min(0.5, 2 * mean_alpha)

    # Sample base_alpha within a reasonable range.
    base_alpha = trial.suggest_float("base_alpha", low_alpha, high_alpha, log=True)

    positions_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
    final_allocation = baseline_allocation.copy()

    # Iterate through dates, rebalancing every rebalance_period.
    for i, date in enumerate(returns_df.index):
        if i % rebalance_period == 0:
            # Compute realized volatility over the past hv_window days.
            start_idx = max(0, i - hv_window)
            window_returns = returns_df.iloc[start_idx:i]
            if not window_returns.empty:
                realized_vol = window_returns.std().mean()
            else:
                realized_vol = historical_vol

            # Scale base_alpha by the current volatility.
            effective_alpha = base_alpha / (1 + realized_vol)

            updated_composite_signals = propagate_signals_by_similarity(
                composite_signals, group_mapping, returns_df
            )
            final_allocation = adjust_allocation_with_mean_reversion(
                baseline_allocation=baseline_allocation,
                composite_signals=updated_composite_signals,
                alpha=effective_alpha,
                allow_short=False,
            )

        positions_df.loc[date] = final_allocation

    metrics = strategy_performance_metrics(
        returns_df=returns_df,
        positions_df=positions_df,
        objective_weights=objective_weights,
    )

    return strategy_composite_score(metrics, objective_weights=objective_weights)
