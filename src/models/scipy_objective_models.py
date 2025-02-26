from typing import Any, Callable, Dict, List
import numpy as np
import pandas as pd

from utils.performance_metrics import conditional_var


def estimated_portfolio_volatility(w: np.ndarray, cov: np.ndarray) -> float:
    """Computes portfolio volatility (standard deviation) given weights and covariance."""
    cov_arr = cov.to_numpy() if isinstance(cov, pd.DataFrame) else cov
    return float(np.sqrt(w.T @ cov_arr @ w))


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


def sharpe_objective(w: np.ndarray, mu_arr: np.ndarray, cov_arr: np.ndarray) -> float:
    """
    Computes the negative Sharpe ratio given weights, expected returns, and covariance.

    Args:
        w (np.ndarray): Portfolio weights, shape (n,).
        mu_arr (np.ndarray): Expected returns, shape (n,).
        cov_arr (np.ndarray): Covariance matrix, shape (n, n).

    Returns:
        float: Negative Sharpe ratio.
    """
    port_return = np.dot(w, mu_arr)
    port_vol = np.sqrt(np.dot(w, np.dot(cov_arr, w)))
    return -(port_return / port_vol if port_vol > 0 else 1e6)


def omega_objective(w: np.ndarray, returns_arr: np.ndarray, theta: float) -> float:
    """
    Computes the negative Omega ratio using historical returns and a threshold.

    Args:
        w (np.ndarray): Portfolio weights, shape (n,).
        returns_arr (np.ndarray): Historical returns, shape (T, n).
        theta (float): Target threshold for the Omega ratio.

    Returns:
        float: Negative Omega ratio.
    """
    port_returns = returns_arr.dot(w)
    expected_return = np.mean(port_returns)
    shortfall = np.maximum(theta - port_returns, 0)
    expected_shortfall = np.mean(shortfall)
    if expected_shortfall < 1e-8:
        expected_shortfall = 1e-8
    omega = (expected_return - theta) / expected_shortfall + 1
    return -omega


def aggro_objective(
    w: np.ndarray,
    returns_arr: np.ndarray,
    cov_arr: np.ndarray,
    target: float,
    order: int,
) -> float:
    """
    Computes a combined objective function for aggressive strategies.

    Args:
        w (np.ndarray): Portfolio weights, shape (n,).
        returns_arr (np.ndarray): Historical returns, shape (T, n).
        cov_arr (np.ndarray): Covariance matrix, shape (n, n).
        target (float): Target return.
        order (int): Order of the lower partial moment (LPM).

    Returns:
        float: Negative combined objective value.
    """
    port_returns = returns_arr @ w
    cumulative_return = np.prod(1 + port_returns) - 1
    port_mean = np.mean(port_returns)
    port_vol = estimated_portfolio_volatility(w, cov_arr)
    lpm = empirical_lpm(port_returns, target=target, order=order)
    kappa_val = ((port_mean - target) / (lpm ** (1.0 / order))) if lpm > 1e-8 else -1e6
    sharpe_val = port_mean / port_vol if port_vol > 0 else -1e6
    combined = (cumulative_return + sharpe_val + kappa_val) / 3.0
    return -combined


def cvar_constraint_fn(
    w: np.ndarray, returns_arr: np.ndarray, alpha: float, cvar_limit: float
) -> float:
    port_returns = returns_arr @ w
    cvar_value = float(conditional_var(pd.Series(port_returns), alpha))
    return cvar_limit - cvar_value


def vol_constraint_fn(w: np.ndarray, cov_arr: np.ndarray, vol_limit: float) -> float:
    port_vol = estimated_portfolio_volatility(w, cov_arr)
    return vol_limit - port_vol


def gross_exposure_fn(w: np.ndarray, max_gross_exposure: float) -> float:
    return float(max_gross_exposure - np.sum(np.abs(w)))


def sum_constraint_fn(w: np.ndarray, target_sum: float) -> float:
    return float(np.sum(w)) - target_sum


def penalized_objective(
    w: np.ndarray,
    chosen_obj: Callable[[np.ndarray], float],
    penalty: Callable[[np.ndarray], float],
) -> float:
    """Module-level function that wraps the chosen objective with a penalty term."""
    return chosen_obj(w) + penalty(w)


def penalty(
    w: np.ndarray,
    penalty_weight: float,
    constraints: List[Dict[str, Any]],
    target_sum: float,
) -> float:
    """
    Computes the penalty for violating constraints during optimization.

    Args:
        w (np.ndarray): Portfolio weights, shape (n,).
        penalty_weight (float): Penalty weight for constraint violations.
        constraints (List[Dict[str, Any]]): List of constraint dictionaries.
        target_sum (float): The target sum for portfolio weights.

    Returns:
        float: The computed penalty value.
    """
    # Enforce the sum of weights to be close to the target sum
    pen = penalty_weight * abs(np.sum(w) - target_sum)

    # Apply penalty for inequality constraints
    for con in constraints:
        if con["type"] == "ineq" and callable(con["fun"]):
            val = con["fun"](w)
            if val < 0:
                pen += penalty_weight * abs(val)

    return pen
