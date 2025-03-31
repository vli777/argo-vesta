from pathlib import Path
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
from utils.caching_utils import (
    compute_ticker_hash,
    load_parameters_from_pickle,
    save_parameters_to_pickle,
)


def tune_reversion_alpha(
    returns_df: pd.DataFrame,
    baseline_allocation: pd.Series,
    composite_signals: dict,
    group_mapping: dict,
    objective_weights: dict,
    ticker_params: dict,
    cache_dir: str = "optuna_cache",
    n_trials: int = 50,
    patience: int = 10,
    hv_window: int = 50,
    rebalance_period: int = 30,
) -> float:
    hash = compute_ticker_hash(sorted(composite_signals.keys()))
    cache_file_path = str(Path(cache_dir) / f"reversion_cache_alpha_{hash}.pkl")
    alpha_cache = load_parameters_from_pickle(filename=cache_file_path)

    if "base_alpha" in alpha_cache:
        print(f"Using cached base_alpha: {alpha_cache['base_alpha']}")
        return alpha_cache["base_alpha"]

    historical_vol = returns_df.rolling(window=hv_window).std().mean().mean()
    precomputed_signals = precompute_composite_signals(
        returns_df, composite_signals, group_mapping, rebalance_period, ticker_params
    )

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
            rebalance_period=rebalance_period,
            hv_window=hv_window,
            precomputed_signals=precomputed_signals,
        ),
        n_trials=n_trials,
        n_jobs=-1,
    )

    best_base_alpha = study.best_params["base_alpha"]
    print(f"Optimal base_alpha: {best_base_alpha}")
    alpha_cache["base_alpha"] = best_base_alpha
    save_parameters_to_pickle(parameters=alpha_cache, filename=cache_file_path)
    return best_base_alpha


def precompute_composite_signals(
    returns_df, composite_signals, group_mapping, rebalance_period, ticker_params
):
    # Only propagate fallback signals.
    fallback_signals = {
        ticker: signal
        for ticker, signal in composite_signals.items()
        if ticker_params.get(ticker, {}).get("mode", "fallback") == "fallback"
    }
    precomputed_signals = {}
    for i, date in enumerate(returns_df.index):
        if i % rebalance_period == 0:
            precomputed_signals[date] = propagate_signals_by_similarity(
                fallback_signals, group_mapping, returns_df
            )
    return precomputed_signals


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
    precomputed_signals: dict = None,  # Pass precomputed signals here
) -> float:
    mean_alpha = min(0.1, max(0.2, 0.5 * historical_vol))
    low_alpha = max(0.01, 0.5 * mean_alpha)
    high_alpha = min(0.5, 2 * mean_alpha)
    base_alpha = trial.suggest_float("base_alpha", low_alpha, high_alpha, log=True)
    positions_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
    final_allocation = baseline_allocation.copy()

    for i, date in enumerate(returns_df.index):
        if i % rebalance_period == 0:
            start_idx = max(0, i - hv_window)
            window_returns = returns_df.iloc[start_idx:i]
            realized_vol = (
                window_returns.std().mean()
                if not window_returns.empty
                else historical_vol
            )
            effective_alpha = base_alpha / (1 + realized_vol)

            # Use precomputed signals if available; otherwise, compute them.
            if precomputed_signals is not None and date in precomputed_signals:
                updated_signals = precomputed_signals[date]
            else:
                updated_signals = propagate_signals_by_similarity(
                    composite_signals, group_mapping, returns_df
                )

            final_allocation = adjust_allocation_with_mean_reversion(
                baseline_allocation=baseline_allocation,
                composite_signals=updated_signals,
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
