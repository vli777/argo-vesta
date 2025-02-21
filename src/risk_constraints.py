from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
import optuna

from config import Config
from models.optimize_portfolio import (
    optimize_weights_objective,
    estimated_portfolio_volatility,
)
from utils.performance_metrics import conditional_var
from utils.portfolio_utils import convert_weights_to_series
from utils import logger


def adaptive_risk_constraints(
    config: Config, risk_estimates: Dict[str, np.ndarray], initial_weights: np.ndarray
) -> Optional[np.ndarray]:
    """
    Adjust portfolio risk constraints using Optuna to tune separate relaxation factors
    for volatility and CVaR. If a risk limit is not specified (None), that constraint is not applied.
    Additionally, if risk_priority is 'vol', the vol_limit is pinned and only CVaR is relaxed;
    if risk_priority is 'cvar', then the cvar_limit is pinned and only vol is relaxed.
    Returns the optimized portfolio weights or None if optimization fails.
    """
    # Extract optional risk limits: leave as None if not specified.
    max_vol: Optional[float] = (
        float(config.portfolio_max_vol)
        if config.portfolio_max_vol is not None
        else None
    )
    max_cvar: Optional[float] = (
        float(config.portfolio_max_cvar)
        if config.portfolio_max_cvar is not None
        else None
    )
    max_weight = float(config.max_weight) if config.max_weight is not None else 1.0
    allow_short: bool = config.allow_short
    risk_priority: str = config.portfolio_risk_priority
    optimization_objective: str = config.optimization_objective
    risk_free_rate: float = config.risk_free_rate

    trading_days_per_year = 252
    risk_free_rate_log_daily = np.log(1 + risk_free_rate) / trading_days_per_year

    if "returns" not in risk_estimates or risk_estimates["returns"].empty:
        target = risk_free_rate_log_daily
    else:
        simulated_returns = risk_estimates["returns"]
        target = max(
            np.percentile(simulated_returns.to_numpy().flatten(), 30),
            risk_free_rate_log_daily,
        )

    def portfolio_cumulative_return(w, returns, T):
        # Assumes returns is a DataFrame of daily log returns.
        daily_return = np.sum(w * returns)  # weighted daily log return
        return T * daily_return

    def objective(trial):
        """
        Optuna objective function for tuning relaxation factors.
        Depending on risk_priority, one of the limits is pinned while the other is relaxed.
        """
        # Determine relaxation factors based on risk_priority.
        if risk_priority == "vol":
            # Pin vol_limit and relax only CVaR.
            relax_factor_vol = 1.0
            relax_factor_cvar = trial.suggest_float(
                "relax_factor_cvar", 1.0, 1.5, step=0.1
            )
        elif risk_priority == "cvar":
            # Pin cvar_limit and relax only volatility.
            relax_factor_cvar = 1.0
            relax_factor_vol = trial.suggest_float(
                "relax_factor_vol", 1.0, 1.5, step=0.1
            )
        else:  # "both" or other: relax both limits.
            relax_factor_vol = trial.suggest_float(
                "relax_factor_vol", 1.0, 1.5, step=0.1
            )
            relax_factor_cvar = trial.suggest_float(
                "relax_factor_cvar", 1.0, 1.5, step=0.1
            )

        vol_limit_adj = max_vol * relax_factor_vol if max_vol is not None else None
        cvar_limit_adj = max_cvar * relax_factor_cvar if max_cvar is not None else None

        try:
            final_w = optimize_weights_objective(
                cov=risk_estimates["cov"],
                mu=risk_estimates["mu"],
                returns=risk_estimates["returns"],
                objective=optimization_objective,
                order=3,
                target=target,
                max_weight=max_weight,
                allow_short=allow_short,
                target_sum=1.0,
                vol_limit=vol_limit_adj,
                cvar_limit=cvar_limit_adj,
                # Only set min_return if vol_limit is provided.
                min_return=(
                    risk_estimates["mu"].mean() * (vol_limit_adj / max_vol)
                    if max_vol is not None
                    else 0.0
                ),
                alpha=0.05,
                solver_method="SLSQP",
                initial_guess=initial_weights,
                apply_constraints=True,
            )

            port_vol = estimated_portfolio_volatility(final_w, risk_estimates["cov"])
            computed_cvar = conditional_var(
                pd.Series(risk_estimates["returns"] @ final_w), 0.05
            )

            vol_loss = max(0, port_vol - max_vol) if max_vol is not None else 0
            cvar_loss = max(0, max_cvar - computed_cvar) if max_cvar is not None else 0

            T = trading_days_per_year
            k = 0.5  # sensitivity parameter for return penalty
            port_cum_return = portfolio_cumulative_return(
                final_w, risk_estimates["returns"], T
            )
            cumulative_target = T * target
            adjusted_cumulative_target = cumulative_target * (
                1 - k * (relax_factor_vol - 1)
            )
            return_penalty = max(0, adjusted_cumulative_target - port_cum_return)

            loss = 5 * vol_loss + 20 * cvar_loss + 10 * return_penalty
            return loss

        except ValueError as e:
            logger.warning(f"Optimization trial failed: {e}")
            return float("inf")

    study = optuna.create_study(direction="minimize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=10)

    # Extract the best relaxation factors based on risk_priority.
    if risk_priority == "vol":
        best_relax_factor_vol = 1.0
        best_relax_factor_cvar = study.best_params["relax_factor_cvar"]
    elif risk_priority == "cvar":
        best_relax_factor_cvar = 1.0
        best_relax_factor_vol = study.best_params["relax_factor_vol"]
    else:
        best_relax_factor_vol = study.best_params["relax_factor_vol"]
        best_relax_factor_cvar = study.best_params["relax_factor_cvar"]

    vol_limit_final = max_vol * best_relax_factor_vol if max_vol is not None else None
    cvar_limit_final = (
        max_cvar * best_relax_factor_cvar if max_cvar is not None else None
    )

    try:
        final_w = optimize_weights_objective(
            cov=risk_estimates["cov"],
            mu=risk_estimates["mu"],
            returns=risk_estimates["returns"],
            objective=optimization_objective,
            order=3,
            target=target,
            max_weight=max_weight,
            allow_short=allow_short,
            target_sum=1.0,
            vol_limit=vol_limit_final,
            cvar_limit=cvar_limit_final,
            min_return=(
                risk_estimates["mu"].mean() * (vol_limit_final / max_vol)
                if max_vol is not None
                else 0.0
            ),
            alpha=0.05,
            solver_method="SLSQP",
            initial_guess=initial_weights,
            apply_constraints=True,
        )
        return final_w
    except ValueError as e:
        logger.error(f"Final optimization failed: {e}")
        return None


def adjust_constraints(
    max_vol: Optional[float],
    max_cvar: Optional[float],
    relax_factor: float,
    risk_priority: str,
) -> tuple:
    """
    This function is retained for backward compatibility.
    When using separate relaxation factors, it is not used in the main optimization.
    """
    if risk_priority == "vol":
        new_vol = max_vol  # pinned
        new_cvar = max_cvar * relax_factor if max_cvar is not None else None
        return new_vol, new_cvar
    elif risk_priority == "cvar":
        new_vol = max_vol * relax_factor if max_vol is not None else None
        new_cvar = max_cvar  # pinned
        return new_vol, new_cvar
    else:  # "both"
        new_vol = max_vol * relax_factor if max_vol is not None else None
        new_cvar = max_cvar * relax_factor if max_cvar is not None else None
        return new_vol, new_cvar


def apply_risk_constraints(
    initial_weights: pd.Series, risk_estimates: dict, config: Config
) -> pd.Series:
    """
    Given a merged (unconstrained) set of weights, re-optimize using risk constraints.
    Uses Optuna to adaptively adjust constraints based on `config.risk_priority`.

    Args:
        initial_weights (pd.Series): Initial portfolio weights.
        risk_estimates (dict): Dictionary of risk measures (`cov`, `mu`, `returns`).
        config (Config): Configuration object.

    Returns:
        pd.Series: Optimized weights.
    """
    cov_assets = risk_estimates["cov"].index
    initial_weights = initial_weights.reindex(cov_assets, fill_value=0)
    initial_weights_np = initial_weights.values

    logger.info(
        f"Applying risk constraints: vol_limit={config.portfolio_max_vol}, cvar_limit={config.portfolio_max_cvar}"
    )

    final_w = adaptive_risk_constraints(config, risk_estimates, initial_weights_np)

    if final_w is None:
        logger.warning("Final optimization failed. Returning initial weights.")
        return initial_weights

    return convert_weights_to_series(final_w, index=risk_estimates["cov"].index)
