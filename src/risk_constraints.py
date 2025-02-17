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
    Adjusts portfolio risk constraints using Optuna to find an optimal relaxation factor.
    Returns the optimized portfolio weights or None if optimization fails.
    """

    # Extract and explicitly typecast key config values
    max_vol: float = float(config.portfolio_max_vol)
    max_cvar: float = float(config.portfolio_max_cvar)
    max_weight: float = float(config.max_weight)
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
        # Set target (τ) dynamically based on simulated returns (30th percentile threshold)
        target = max(
            np.percentile(simulated_returns.to_numpy().flatten(), 30),
            risk_free_rate_log_daily,
        )

    def portfolio_cumulative_return(w, returns, T):
        # Here, we assume returns is a DataFrame of daily log returns.
        # The expected daily return is given by the weighted average.
        daily_return = np.sum(w * returns)  # Or compute from returns
        return T * daily_return

    def objective(trial):
        """
        Optuna objective function for tuning constraint relaxations.
        Computes portfolio loss based on adjusted volatility and CVaR constraints.
        """

        # Suggest relaxation factor in steps of 0.1
        relax_factor = trial.suggest_float("relax_factor", 1.0, 1.5, step=0.1)

        # Adjust constraints based on priority
        vol_limit_adj, cvar_limit_adj = adjust_constraints(
            max_vol, max_cvar, relax_factor, risk_priority
        )

        try:
            # Optimize portfolio with adjusted constraints
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
                min_return=risk_estimates["mu"].mean() * (vol_limit_adj / max_vol),
                alpha=0.05,
                solver_method="SLSQP",
                initial_guess=initial_weights,
                apply_constraints=True,
            )

            # Compute risk metrics
            port_vol = estimated_portfolio_volatility(final_w, risk_estimates["cov"])
            computed_cvar = conditional_var(
                pd.Series(risk_estimates["returns"] @ final_w), 0.05
            )

            # Calculate constraint violations (loss function)
            vol_loss = max(
                0, port_vol - max_vol
            )  # Positive if it exceeds allowed volatility
            cvar_loss = max(
                0, max_cvar - computed_cvar
            )  # Positive if it exceeds allowed CVaR

            # Compute cumulative return penalty:
            trading_days_per_year = 252
            k = 0.5  # sensitivity parameter

            port_cumulative_return = portfolio_cumulative_return(
                final_w, risk_estimates["returns"], T=trading_days_per_year
            )
            # If portfolio cumulative return is below the adjusted target, penalize.
            cumulative_target = (
                trading_days_per_year * target
            )  # Expected cumulative log return without adjustment
            # Adjust the cumulative target based on the relaxation factor:
            adjusted_cumulative_target = cumulative_target * (
                1 - k * (relax_factor - 1)
            )
            return_penalty = max(0, adjusted_cumulative_target - port_cumulative_return)

            # Combine the penalties: adjust weights as needed.
            loss = 5 * vol_loss + 20 * cvar_loss + 10 * return_penalty

            return loss

        except ValueError:
            return float("inf")  # Penalize infeasible solutions

    # Optimize relaxation factor
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    best_relax_factor = study.best_params["relax_factor"]

    # Compute final adjusted constraints
    final_vol_limit, final_cvar_limit = adjust_constraints(
        max_vol, max_cvar, best_relax_factor, risk_priority
    )

    try:
        # Final optimization using the best relaxation factor
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
            vol_limit=final_vol_limit,
            cvar_limit=final_cvar_limit,
            min_return=risk_estimates["mu"].mean() * (final_vol_limit / max_vol),
            alpha=0.05,
            solver_method="SLSQP",
            initial_guess=initial_weights,
            apply_constraints=True,
        )
        return final_w

    except ValueError:
        logger.error("Optimization failed even after relaxing constraints.")
        return None


def adjust_constraints(
    max_vol: float, max_cvar: float, relax_factor: float, risk_priority: str
) -> tuple:
    """
    Adjusts volatility and CVaR limits based on relaxation factor and risk priority.
    Returns (vol_limit, cvar_limit).
    """
    if risk_priority == "vol":
        return max_vol * relax_factor, max_cvar  # Relax volatility, keep CVaR fixed
    elif risk_priority == "cvar":
        return max_vol, max_cvar * relax_factor  # Keep volatility fixed, relax CVaR
    else:  # "both"
        return max_vol * relax_factor, max_cvar * relax_factor  # Relax both constraints


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
