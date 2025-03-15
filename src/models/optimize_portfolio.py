import numpy as np
import pandas as pd
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
from scipy.optimize import minimize
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition, SolverStatus
from functools import partial

from models.pyomo_objective_models import (
    build_omega_model,
    build_sharpe_model,
)
from models.simulated_annealing import multi_seed_dual_annealing
from models.stochastic_diffusion import multi_seed_diffusion
from models.scipy_objective_models import (
    aggro_objective,
    cvar_constraint_fn,
    estimated_portfolio_volatility,
    gross_exposure_fn,
    omega_objective,
    penalized_objective,
    penalty,
    sharpe_objective,
    sum_constraint_fn,
    vol_constraint_fn,
)

from utils import logger


def build_constraints_and_bounds(
    cov: pd.DataFrame,
    returns: Optional[pd.DataFrame],
    mu: Optional[Union[pd.Series, np.ndarray]],
    target_sum: float,
    cvar_limit: Optional[float],
    vol_limit: Optional[float],
    allow_short: bool,
    max_gross_exposure: float,
    apply_constraints: bool,
    min_weight: Optional[float],
    max_weight: Optional[float],
    alpha: float,
) -> Tuple[
    List[Tuple[float, float]],
    List[Dict],
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    n = cov.shape[0]
    max_weight = float(max_weight) if max_weight is not None else 1.0
    # Ensure max_weight is at least equal to equal allocation.
    max_weight = max(1.0 / n, max_weight)
    lower_bound = (
        -max_weight
        if allow_short
        else (float(min_weight) if min_weight is not None else 0.0)
    )
    bounds = [(lower_bound, max_weight)] * n

    cov_arr = cov.to_numpy() if isinstance(cov, pd.DataFrame) else cov
    returns_arr = (
        returns.to_numpy()
        if (returns is not None and isinstance(returns, pd.DataFrame))
        else returns
    )
    mu_arr = mu.to_numpy() if (mu is not None and isinstance(mu, pd.Series)) else mu

    constraints = [
        {
            "type": "eq",
            "fun": partial(sum_constraint_fn, target_sum=target_sum),
        }
    ]

    if apply_constraints:
        if cvar_limit is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": partial(
                        cvar_constraint_fn,
                        returns_arr=returns_arr,
                        alpha=alpha,
                        cvar_limit=cvar_limit,
                    ),
                }
            )
        if vol_limit is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": partial(
                        vol_constraint_fn, cov_arr=cov_arr, vol_limit=vol_limit
                    ),
                }
            )
    if allow_short:
        constraints.append(
            {
                "type": "ineq",
                "fun": partial(
                    gross_exposure_fn, max_gross_exposure=max_gross_exposure
                ),
            }
        )
    return bounds, constraints, cov_arr, returns_arr, mu_arr


def run_global_optimization(
    chosen_obj: Callable,
    init_weights: np.ndarray,
    bounds: List[Tuple[float, float]],
    constraints: List[Dict],
    solver_method: str,
    target_sum: float,
    penalty_weight: float,
    use_annealing: bool,
    use_diffusion: bool,
    callback: Optional[Callable] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    local_result = minimize(
        chosen_obj,
        init_weights,
        method=solver_method,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9, "eps": 1e-8},
    )
    if local_result.success:
        candidate = local_result.x
    else:
        logger.warning(
            "Local solver did not converge; using init_weights as candidate."
        )
        candidate = init_weights

    # Configure the penalized objective.
    penalty_function = partial(
        penalty,
        penalty_weight=penalty_weight,
        constraints=constraints,
        target_sum=target_sum,
    )
    penalized_obj = partial(
        penalized_objective,
        chosen_obj=chosen_obj,
        penalty=penalty_function,
    )

    if use_annealing:
        cb = callback if callback is not None else (lambda x, f, context: False)
        result = multi_seed_dual_annealing(
            penalized_obj,
            bounds=bounds,
            num_runs=10,
            maxiter=10000,
            initial_temp=10000,
            visit=10,
            accept=-10.0,
            callback=cb,
            initial_candidate=candidate,
            perturb_scale=0.3,
            target_sum=target_sum,
        )
        if result.success:
            return result.x, candidate
        else:
            logger.warning("Dual annealing optimization failed: " + result.message)
    elif use_diffusion:
        cb = callback if callback is not None else (lambda x, convergence: False)
        result = multi_seed_diffusion(
            penalized_obj,
            bounds=bounds,
            num_runs=10,
            popsize=15,
            maxiter=10000,
            mutation=(0.5, 1),
            recombination=0.7,
            callback=cb,
            # initial_candidate=candidate, # Diffusion doesn't require explicit seeding.
        )
        if result.success:
            return result.x, candidate
        else:
            logger.warning(
                "Stochastic diffusion optimization did not converge: " + result.message
            )

    # Global method was selected but did not succeed.
    return None, candidate


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
    # Build bounds, constraints, and precomputed arrays.
    bounds, constraints, cov_arr, returns_arr, mu_arr = build_constraints_and_bounds(
        cov=cov,
        returns=returns,
        mu=mu,
        target_sum=target_sum,
        cvar_limit=cvar_limit,
        vol_limit=vol_limit,
        allow_short=allow_short,
        max_gross_exposure=max_gross_exposure,
        apply_constraints=apply_constraints,
        min_weight=min_weight,
        max_weight=max_weight,
        alpha=alpha,
    )
    lower_bound = bounds[0][0]

    # Define the objective function to be optimized.
    chosen_obj = None
    if objective.lower() == "omega":
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

        if n < 50:
            chosen_obj = partial(omega_objective, returns_arr=returns_arr, theta=target)
        else:
            try:
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
                results_pyomo = solver.solve(model, tee=False)
                if (
                    results_pyomo.solver.status != pyo.SolverStatus.ok
                    or results_pyomo.solver.termination_condition
                    != pyo.TerminationCondition.optimal
                ):
                    raise RuntimeError(
                        f"Solver did not converge! Status: {results_pyomo.solver.status}, Termination: {results_pyomo.solver.termination_condition}"
                    )
                # Recover weights from y and z: w = y/z.
                z_val = pyo.value(model.z)
                if z_val is None or abs(z_val) < 1e-8:
                    raise RuntimeError("Invalid scaling value in Omega optimization.")
                weights = (
                    np.array([pyo.value(model.y[i]) for i in model.assets]) / z_val
                )
                # Normalize to sum to target_sum.
                weights = weights / np.sum(weights) * target_sum
                return weights
            except Exception as e:
                logger.warning(
                    "Pyomo optimization for Omega objective failed: "
                    + str(e)
                    + ". Falling back to local solver approach."
                )
                chosen_obj = partial(
                    omega_objective, returns_arr=returns_arr, theta=target
                )

    elif objective.lower() == "aggro":
        if returns_arr is None or returns.empty:
            raise ValueError(
                "Historical returns must be provided for Omega optimization."
            )

        chosen_obj = partial(
            aggro_objective,
            returns_arr=returns_arr,
            cov_arr=cov_arr,
            target_return=target,
            order=order,
        )

    else:  # default to sharpe optimization
        if returns_arr is None or mu_arr is None:
            raise ValueError(
                "Both historical returns and expected returns (mu) must be provided for Sharpe optimization."
            )
        if n < 50:
            chosen_obj = partial(sharpe_objective, mu_arr=mu_arr, cov_arr=cov_arr)
        else:
            try:
                model_pyomo = build_sharpe_model(
                    cov=cov,
                    mu=mu,
                    returns=returns,
                    target_sum=target_sum,
                    min_weight=min_weight,
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
                results_pyomo = solver.solve(model_pyomo, tee=False)
                # Check if the solver status is OK and termination condition is optimal.
                if (
                    results_pyomo.solver.status != pyo.SolverStatus.ok
                    or results_pyomo.solver.termination_condition
                    != pyo.TerminationCondition.optimal
                ):
                    raise RuntimeError(
                        f"Solver did not converge! Status: {results_pyomo.solver.status}, "
                        f"Termination: {results_pyomo.solver.termination_condition}"
                    )
                weights_pyomo = np.array(
                    [pyo.value(model_pyomo.w[i]) for i in model_pyomo.assets]
                )
                weights_pyomo = weights_pyomo / np.sum(weights_pyomo) * target_sum
                return weights_pyomo
            except Exception as e:
                logger.warning(
                    "Pyomo optimization failed (possibly due to infeasibility): "
                    f"{e}. Falling back to local solver approach."
                )
                chosen_obj = partial(sharpe_objective, mu_arr=mu_arr, cov_arr=cov_arr)

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
        init_weights = np.clip(initial_guess, lower_bound, max_weight)
    else:
        if allow_short:
            init_weights = np.random.uniform(lower_bound, max_weight, size=n)
            init_weights /= np.sum(np.abs(init_weights))
            init_weights *= target_sum
        else:
            init_weights = np.full(n, target_sum / n)
            if min_weight is not None:
                init_weights = np.maximum(init_weights, min_weight)
            init_weights = np.minimum(init_weights, max_weight)
            init_weights = init_weights / init_weights.sum() * target_sum

    # Check feasibility of initial weights; adjust if necessary.
    if (
        vol_limit is not None
        and estimated_portfolio_volatility(init_weights, cov) > vol_limit
    ):
        init_weights *= 0.95
        init_weights = init_weights / init_weights.sum() * target_sum

    # --- Global Optimization Branch ---
    if use_annealing or use_diffusion:
        global_result, candidate = run_global_optimization(
            chosen_obj=chosen_obj,
            init_weights=init_weights,
            bounds=bounds,
            constraints=constraints,
            solver_method=solver_method,
            target_sum=target_sum,
            penalty_weight=penalty_weight,
            use_annealing=use_annealing,
            use_diffusion=use_diffusion,
            callback=callback,
        )
        if global_result is not None:
            return global_result
        else:
            # Fall back to the candidate from local minimization if global fails.
            init_weights = candidate

    # --- Standard Local Solver Branch ---
    result = minimize(
        chosen_obj,
        init_weights,
        method=solver_method,
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={"maxiter": 1000, "ftol": 1e-9, "eps": 1e-8},
    )
    if not result.success:
        raise ValueError("Optimization failed: " + result.message)
    return result.x
