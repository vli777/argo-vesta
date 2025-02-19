import numpy as np
import pandas as pd
from typing import Optional, Union, Dict
from scipy.optimize import minimize
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition, SolverStatus
from models.pyomo_objective_models import (
    build_omega_model,
    build_sharpe_model,
)
from utils import logger
from utils.performance_metrics import conditional_var


def empirical_lpm(portfolio_returns, target=0, order=3):
    """
    Compute the empirical lower partial moment (LPM) of a return series.

    Parameters:
        portfolio_returns : array-like, historical portfolio returns.
        target          : target return level.
        order           : order of the LPM (default is 3).

    Returns:
        The LPM of the specified order.
    """
    diff = np.maximum(target - portfolio_returns, 0)
    return np.mean(diff**order)


def estimated_portfolio_volatility(w: np.ndarray, cov: np.ndarray) -> float:
    """Computes portfolio volatility (standard deviation) given weights and covariance."""
    return np.sqrt(w.T @ cov @ w)


def optimize_weights_objective(
    cov: pd.DataFrame,
    mu: Optional[Union[pd.Series, np.ndarray]] = None,
    returns: Optional[pd.DataFrame] = None,
    objective: str = "sharpe",
    order: int = 3,
    target: float = 0.0,
    min_weight: Optional[float] = None,
    max_weight: Optional[float] = None,
    allow_short: bool = False,
    target_sum: float = 1.0,
    # Constraint: maximum allowable portfolio volatility e.g., 0.15 for 15% vol
    vol_limit: Optional[float] = None,
    # Constraint: CVaR cannot exceed this threshold e.g., -0.02 for -2% CVaR
    cvar_limit: Optional[float] = None,
    alpha: float = 0.05,  # Tail probability for CVaR, default 5%
    min_return: float = 0.0,  # Constraint: minimum allowable cumulative return
    solver_method: str = "SLSQP",
    initial_guess: Optional[np.ndarray] = None,
    apply_constraints: bool = False,
) -> np.ndarray:
    """
    Optimize portfolio weights using a unified, robust interface.

    For 'sharpe', expected returns (mu) and covariance (cov) are used.
    For objectives such as 'omega', historical returns (returns) are required.

    Args:
        cov (pd.DataFrame): Covariance matrix of asset returns.
        mu (Optional[Union[pd.Series, np.ndarray]]): Expected returns.
        returns (Optional[pd.DataFrame]): Historical returns (T x n), where T is time.
        objective (str): Optimization objective (default 'sharpe').
        target (float): Target return (default 0.0).
        min_weight (Optional[float]): Minimum weight per asset (for long-only). If None, defaults to 0.0.
        max_weight (float): Maximum weight per asset (default 1.0).
        allow_short (bool): Allow short positions (default False).
        target_sum (float): Sum of weights (default 1.0).
        vol_limit (float): Maximum acceptable portfolio volatility (default 30%).
        cvar_limit (float): Maximum acceptable CVaR value (default -2%).
        alpha (float): Tail probability for CVaR constraint.
        min_return (float): Minimum acceptable portfolio return.
        solver_method (str): Solver method for use with scipy minimize (default "SQSLP").
        initial_guess (np.ndarray): Initial estimates for the optimal portfolio weights.
        apply_constraints (bool): Whether to add risk constraints (volatility and CVaR).


    Returns:
        np.ndarray: Optimized portfolio weights.
    """
    n = cov.shape[0]
    # Default max_weight to 1.0 if not provided.
    if max_weight is None:
        max_weight = 1.0
    else:
        max_weight = float(max_weight)
    # Ensure max_weight is at least equal to equal allocation
    max_weight = max(1.0 / n, max_weight)
    # Set lower bound based on whether shorting is allowed.
    if allow_short:
        lower_bound = -max_weight
    else:
        # If user provided a min_weight, use that; otherwise default to 0.
        lower_bound = float(min_weight) if min_weight is not None else 0.0

    bounds = [(lower_bound, max_weight)] * n

    # Define CVaR constraint using the conditional_var function.
    def cvar_constraint(w):
        port_returns = returns @ w  # returns is (T x n) and w is (n,)
        cvar_value = conditional_var(pd.Series(port_returns), alpha)
        # Constraint: cvar_limit - computed CVaR must be >= 0.
        return cvar_limit - cvar_value

    # Define the volatility constraint with a small slack to ease numerical issues.
    def vol_constraint(w):
        return vol_limit - estimated_portfolio_volatility(w, cov)

    def gross_exposure(w):
        return 1.3 - np.sum(np.abs(w))

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - target_sum},
    ]

    # Add CVaR and max volatility
    if apply_constraints:
        if cvar_limit is not None:
            constraints.append({"type": "ineq", "fun": cvar_constraint})
        if vol_limit is not None:
            constraints.append({"type": "ineq", "fun": vol_constraint})

    # Add gross exposure for long/short
    if allow_short:
        constraints.append({"type": "ineq", "fun": gross_exposure})

    # We'll assign the selected objective function to chosen_obj.
    chosen_obj = None
    if objective.lower() == "sharpe":
        if returns is None or mu is None:
            raise ValueError(
                "Both historical returns and expected returns (mu) must be provided for Sharpe optimization."
            )
        if n < 50:
            logger.info("Optimizing for max Sharpe...")

            def obj(w: np.ndarray) -> float:
                port_return = w @ mu
                port_vol = estimated_portfolio_volatility(w, cov)
                return -port_return / port_vol if port_vol > 0 else 1e6

            chosen_obj = obj
        else:
            logger.info("Optimizing for max Sharpe with pyomo...")
            model_pyomo = build_sharpe_model(
                cov=cov,
                mu=mu,
                returns=returns,
                target_sum=target_sum,
                max_weight=max_weight,
                allow_short=allow_short,
                vol_limit=vol_limit if apply_constraints else None,
                cvar_limit=cvar_limit if apply_constraints else None,
                min_return=min_return if apply_constraints else None,
                alpha=alpha,
            )
            solver = pyo.SolverFactory(
                "ipopt",
                executable="H:/Solvers/Ipopt-3.14.17-win64-msvs2022-md/bin/ipopt.exe",
            )
            solver.solve(model_pyomo)
            weights_pyomo = np.array(
                [pyo.value(model_pyomo.w[i]) for i in model_pyomo.assets]
            )
            # sharpe_pyomo = -pyo.value(model_pyomo.obj)
            return weights_pyomo

    elif objective.lower() == "omega":
        """
        For the 'omega' objective, historical returns are required. This function
        implements a robust version of the Omega ratio optimization using a linear-fractional
        programming formulation (which is then converted to a linear program). It uses robust
        estimates for expected returns (via a trimmed mean) and enforces constraints to control
        individual weights.

        The Omega ratio is defined as:

            Omega(θ) = [w^T E(r) - θ] / E[(θ - w^T r)_+] + 1

        which can be transformed into the following linear program:

        max_{y,q,z} y^T E(r) - θ z
        s.t.
            y^T E(r) >= θ z,
            sum(q) = 1,
            sum(y) = z,
            q_j >= θ z - y^T r_j,   for all j,
            (if no shorts:) y >= 0,
            y <= max_weight * z,
            z >= 0.

        After solving, the portfolio weights are recovered as w = y / z (and then normalized
        to sum to target_sum).
        """
        if returns is None or returns.empty:
            raise ValueError(
                "Historical returns must be provided for Omega optimization."
            )

        model = build_omega_model(
            cov=cov,
            returns=returns,
            target=target,
            target_sum=target_sum,
            max_weight=max_weight,
            allow_short=allow_short,
            vol_limit=vol_limit if apply_constraints else None,
            cvar_limit=cvar_limit if apply_constraints else None,
            alpha=alpha,
        )
        # Since the transformed omega formulation is linear, you could use an LP solver like CBC.
        solver = pyo.SolverFactory(
            "cbc",
            executable="H:/Solvers/Cbc-releases.2.10.12-w64-msvc17-md/bin/cbc.exe",
        )
        results = solver.solve(model, tee=False)
        if (results.solver.status != SolverStatus.ok) or (
            results.solver.termination_condition != TerminationCondition.optimal
        ):
            raise RuntimeError(
                "Solver did not converge! Status: {results.solver.status}, Termination: {results.solver.termination_condition}"
            )

        # Recover weights from y and z: w = y/z.
        z_val = pyo.value(model.z)
        if z_val is None or abs(z_val) < 1e-8:
            raise RuntimeError("Invalid scaling value in Omega optimization.")
        weights = np.array([pyo.value(model.y[i]) for i in model.assets]) / z_val
        # Normalize to sum to target_sum.
        weights = weights / np.sum(weights) * target_sum

        return weights

    elif objective.lower() == "aggro":
        if returns is None or mu is None:
            raise ValueError(
                "Both historical returns and expected returns (mu) must be provided for aggro optimization."
            )

        def obj(w: np.ndarray) -> float:
            port_returns = returns.values @ w
            cumulative_return = np.prod(1 + port_returns) - 1
            port_mean = np.mean(port_returns)
            port_vol = np.sqrt(w.T @ cov @ w)
            lpm = empirical_lpm(port_returns, target=target, order=order)
            kappa_val = (
                (port_mean - target) / (lpm ** (1.0 / order)) if lpm > 1e-8 else -1e6
            )
            sharpe_val = port_mean / port_vol if port_vol > 0 else -1e6
            combined = (
                (1 / 3) * cumulative_return + (1 / 3) * sharpe_val + (1 / 3) * kappa_val
            )
            return -combined

        chosen_obj = obj

    else:

        def obj(w: np.ndarray) -> float:
            port_return = w @ mu
            port_vol = np.sqrt(w.T @ cov @ w)
            return -port_return / port_vol if port_vol > 0 else 1e6

        chosen_obj = obj

    # Set initial weights (equal allocation)
    init_weights = (
        initial_guess if initial_guess is not None else np.full(n, target_sum / n)
    )

    # Check feasibility of initial weights; adjust if necessary.
    if (
        vol_limit is not None
        and estimated_portfolio_volatility(init_weights, cov) > vol_limit
    ):
        # Slightly dampen the weights if initial volatility is too high.
        init_weights *= 0.95

    result = minimize(
        chosen_obj,
        init_weights,
        method=solver_method,
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    return result.x
