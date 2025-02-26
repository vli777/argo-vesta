import numpy as np
import pandas as pd
from typing import Any, Callable, Optional, Union, Dict
from scipy.optimize import minimize
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition, SolverStatus

from models.pyomo_objective_models import (
    build_omega_model,
    build_sharpe_model,
)
from models.simulated_annealing import multi_seed_dual_annealing
from models.stochastic_diffusion import multi_seed_diffusion
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
    portfolio_returns = np.asarray(portfolio_returns)
    diff = np.maximum(target - portfolio_returns, 0)
    return np.mean(diff**order)


def estimated_portfolio_volatility(w: np.ndarray, cov: np.ndarray) -> float:
    """Computes portfolio volatility (standard deviation) given weights and covariance."""
    cov_arr = cov.to_numpy() if isinstance(cov, pd.DataFrame) else cov
    return float(np.sqrt(w.T @ cov_arr @ w))


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
    max_gross_exposure: float = 1.3,
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
    use_annealing: bool = False,
    use_diffusion: bool = False,
    penalty_weight: float = 1e6,  
    callback: Optional[Callable[[np.ndarray, float, Any], bool]] = None,
) -> np.ndarray:
    """
    Optimize portfolio weights using a unified, robust interface.
    If use_annealing is True, dual annealing (global optimization) is used
    with a penalized version of the objective function.

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
        max_gross_exposure (float): Maximum gross exposure when short positions are allowed.
        target_sum (float): Sum of weights (default 1.0).
        vol_limit (float): Maximum acceptable portfolio volatility (default 30%).
        cvar_limit (float): Maximum acceptable CVaR value (default -2%).
        alpha (float): Tail probability for CVaR constraint.
        min_return (float): Minimum acceptable portfolio return.
        solver_method (str): Solver method for use with scipy minimize (default "SQSLP").
        initial_guess (np.ndarray): Initial estimates for the optimal portfolio weights.
        apply_constraints (bool): Whether to add risk constraints (volatility and CVaR).
        callback (callable): Optional callback (for recording optimization search path).

    Returns:
        np.ndarray: Optimized portfolio weights.
    """
    n = cov.shape[0]
    max_weight = float(max_weight) if max_weight is not None else 1.0
    # Ensure max_weight is at least equal to equal allocation
    max_weight = max(1.0 / n, max_weight)
    # Set lower bound based on whether shorting is allowed.
    lower_bound = (
        -max_weight
        if allow_short
        else (float(min_weight) if min_weight is not None else 0.0)
    )
    bounds = [(lower_bound, max_weight)] * n

    # Precompute numpy arrays from cov and returns to avoid repeated conversions.
    cov_arr = cov.to_numpy() if isinstance(cov, pd.DataFrame) else cov
    if returns is not None:
        returns_arr = (
            returns.to_numpy() if isinstance(returns, pd.DataFrame) else returns
        )
    else:
        returns_arr = None
    if mu is not None:
        mu_arr = mu.to_numpy() if isinstance(mu, pd.Series) else mu
    else:
        mu_arr = None

    # Define CVaR constraint using the conditional_var function.
    def cvar_constraint(w):
        # Ensure returns is a NumPy array before matrix multiplication:
        port_returns = returns_arr @ w
        cvar_value = conditional_var(pd.Series(port_returns), alpha)
        # Constraint: cvar_limit - computed CVaR must be >= 0.
        # Convert to float if needed:
        cvar_value = float(cvar_value)
        return cvar_limit - cvar_value

    def vol_constraint(w):
        port_vol = estimated_portfolio_volatility(w, cov_arr)
        return vol_limit - port_vol

    def gross_exposure(w):
        return float(max_gross_exposure - np.sum(np.abs(w)))

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - target_sum}]

    # Add CVaR and max volatility
    if apply_constraints:
        if cvar_limit is not None:
            constraints.append({"type": "ineq", "fun": cvar_constraint})
        if vol_limit is not None:
            constraints.append({"type": "ineq", "fun": vol_constraint})
    if allow_short:
        constraints.append({"type": "ineq", "fun": gross_exposure})

    # Define the objective function to be optimized.
    chosen_obj = None
    if objective.lower() == "sharpe":
        if returns_arr is None or mu_arr is None:
            raise ValueError(
                "Both historical returns and expected returns (mu) must be provided for Sharpe optimization."
            )
        if n < 50:

            def obj(w: np.ndarray) -> float:
                port_return = float(np.dot(w, mu_arr))
                port_vol = estimated_portfolio_volatility(w, cov_arr)
                return -port_return / port_vol if port_vol > 0 else 1e6

            chosen_obj = obj
        else:
            model_pyomo = build_sharpe_model(
                cov=cov,
                mu=mu,
                returns=returns,
                target_sum=target_sum,
                max_weight=max_weight,
                allow_short=allow_short,
                gross_target=max_gross_exposure,
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
        if returns_arr is None or returns.empty:
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
        # Since the transformed omega formulation is linear, use an LP solver like CBC.
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
        if returns_arr is None or returns.empty:
            raise ValueError(
                "Historical returns must be provided for Omega optimization."
            )

        def obj(w: np.ndarray) -> float:
            port_returns = returns_arr @ w
            cumulative_return = np.prod(1 + port_returns) - 1
            port_mean = np.mean(port_returns)
            port_vol = estimated_portfolio_volatility(w, cov_arr)
            lpm = empirical_lpm(port_returns, target=target, order=order)
            kappa_val = (
                ((port_mean - target) / (lpm ** (1.0 / order))) if lpm > 1e-8 else -1e6
            )
            sharpe_val = port_mean / port_vol if port_vol > 0 else -1e6
            combined = (cumulative_return + sharpe_val + kappa_val) / 3.0
            return -combined

        chosen_obj = obj

    else:

        def obj(w: np.ndarray) -> float:
            port_return = float(np.dot(w, mu_arr))
            port_vol = estimated_portfolio_volatility(w, cov_arr)
            return -port_return / port_vol if port_vol > 0 else 1e6

        chosen_obj = obj

    # Check for infeasibility and adjust weight constraints if needed
    if min_weight is not None and min_weight * n > target_sum:
        logger.warning(
            f"Relaxing min_weight from {min_weight} to {target_sum / n:.4f} to meet feasibility."
        )
        min_weight = target_sum / n

    if max_weight is not None and max_weight * n < target_sum:
        logger.warning(
            f"Relaxing max_weight from {max_weight} to {target_sum / n:.4f} to meet feasibility."
        )
        max_weight = target_sum / n

    # Initialize weights considering relaxed min_weight and max_weight constraints
    if initial_guess is not None:
        # Clip initial guess to ensure it is within bounds
        init_weights = np.clip(initial_guess, lower_bound, max_weight)
    else:
        if allow_short:
            # Randomly initialize weights allowing for short positions
            init_weights = np.random.uniform(lower_bound, max_weight, size=n)
            # Normalize to sum to target_sum, maintaining potential short positions
            init_weights /= np.sum(np.abs(init_weights))
            init_weights *= target_sum
        else:
            # Uniform initialization for long-only portfolios
            init_weights = np.full(n, target_sum / n)
            # Ensure weights are within specified bounds
            if min_weight is not None:
                init_weights = np.maximum(init_weights, min_weight)
            init_weights = np.minimum(init_weights, max_weight)

    # Check feasibility of initial weights; adjust if necessary.
    if (
        vol_limit is not None
        and estimated_portfolio_volatility(init_weights, cov) > vol_limit
    ):
        init_weights *= 0.95

    # --- Dual Annealing Branch ---
    if use_annealing:
        # Define a penalty for constraint violations.
        def penalty(w) -> float:
            pen = penalty_weight * abs(np.sum(w) - target_sum)
            for con in constraints:
                if con["type"] == "ineq":
                    val = con["fun"](w)
                    if val < 0:
                        pen += penalty_weight * abs(val)
            return pen

        def penalized_obj(w) -> float:
            return chosen_obj(w) + penalty(w)

        cb = callback if callback is not None else (lambda x, f, context: False)

        # Call the multi-seed dual annealing function
        result = multi_seed_dual_annealing(
            penalized_obj,
            bounds=bounds,
            num_runs=20,  # Run the optimizer 20 times with different seeds
            maxiter=10000,
            initial_temp=10000,
            visit=10,
            accept=-10.0,
            callback=cb,
        )

        if not result.success:
            raise ValueError("Dual annealing optimization failed: " + result.message)

        return result.x

    if use_diffusion:
        # Define a penalty for constraint violations.
        def penalty(w) -> float:
            pen = penalty_weight * abs(np.sum(w) - target_sum)
            for con in constraints:
                if con["type"] == "ineq":
                    val = con["fun"](w)
                    if val < 0:
                        pen += penalty_weight * abs(val)
            return pen

        def penalized_obj(w) -> float:
            return chosen_obj(w) + penalty(w)

        cb = callback if callback is not None else (lambda x, convergence: False)

        # Call the multi-seed stochastic diffusion function with a progress bar
        result = multi_seed_diffusion(
            penalized_obj,
            bounds=bounds,
            num_runs=20,  # Number of random seeds
            popsize=15,  # Population size
            maxiter=1000,  # Number of generations
            mutation=(0.5, 1),  # Mutation range
            recombination=0.7,  # Recombination rate
            callback=cb,
        )

        if not result.success:
            raise ValueError(
                "Stochastic diffusion optimization failed: " + result.message
            )

        return result.x
    # --- Standard Local Solver Branch ---
    result = minimize(
        chosen_obj,
        init_weights,
        method=solver_method,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9, "eps": 1e-8},
    )
    if not result.success:
        raise ValueError("Optimization failed: " + result.message)
    return result.x
