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

# optuna.logging.set_verbosity(optuna.logging.ERROR)


def alpha_objective(
    trial,
    returns_df: pd.DataFrame,
    historical_vol: float,
    baseline_allocation: pd.Series,
    composite_signals: dict,
    group_mapping: dict,
    objective_weights: dict,
    rebalance_period: int = 30,
) -> float:
    """
    Optuna objective function for tuning base_alpha.
    Uses volatility-conditioned priors and rebalances dynamically.
    """
    # Set prior mean based on historical volatility
    mean_alpha = min(0.1, max(0.2, 0.5 * historical_vol))
    low_alpha = max(0.01, 0.5 * mean_alpha)
    high_alpha = min(0.5, 2 * mean_alpha)

    # Sample base_alpha within a reasonable range based on prior
    base_alpha = trial.suggest_float("base_alpha", low_alpha, high_alpha, log=True)

    # Expand allocations into a dynamic positions DataFrame
    positions_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)

    for i, date in enumerate(returns_df.index):
        if i % rebalance_period == 0:
            # Recompute mean reversion signals every rebalance period
            updated_composite_signals = propagate_signals_by_similarity(
                composite_signals, group_mapping, returns_df
            )
            # Adjust allocations dynamically
            final_allocation = adjust_allocation_with_mean_reversion(
                baseline_allocation=baseline_allocation,
                composite_signals=updated_composite_signals,
                alpha=base_alpha,
                allow_short=False,
            )

        # Store the latest allocation
        positions_df.loc[date] = final_allocation

    # Simulate strategy performance
    metrics = strategy_performance_metrics(
        returns_df=returns_df,
        positions_df=positions_df,
        objective_weights=objective_weights,
    )

    # Return composite performance score
    return strategy_composite_score(metrics, objective_weights=objective_weights)


def tune_reversion_alpha(
    returns_df: pd.DataFrame,
    baseline_allocation: pd.Series,
    composite_signals: dict,
    group_mapping: dict,
    objective_weights: dict,
    n_trials: int = 50,
    patience: int = 10,  # Stop early if no improvement
) -> float:

    # Compute historical realized volatility
    historical_vol = returns_df.rolling(window=180).std().mean().mean()

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
