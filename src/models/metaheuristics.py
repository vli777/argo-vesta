# global_optim.py
import numpy as np
from scipy.optimize import dual_annealing


# global_optim.py
import numpy as np
from scipy.optimize import dual_annealing


def global_optimize(
    chosen_obj,
    bounds,
    target_sum,
    apply_constraints=False,
    cvar_constraint=None,
    vol_constraint=None,
    penalty_weight=1e6,
    maxiter=1000,
):
    """
    Global optimization using dual annealing, with penalty functions to enforce constraints.

    Parameters:
      - chosen_obj: callable
          The base objective function that takes a weight vector (w) and returns a scalar.
      - bounds: list of tuples
          Bounds for each variable, e.g., [(lower, upper), ...].
      - target_sum: float
          The required sum of weights (equality constraint: sum(w) == target_sum).
      - apply_constraints: bool
          Whether to enforce additional inequality constraints (CVaR, volatility).
      - cvar_constraint: callable (optional)
          Function that takes w and returns a scalar; feasible if >= 0.
      - vol_constraint: callable (optional)
          Function that takes w and returns a scalar; feasible if >= 0.
      - penalty_weight: float
          Multiplier for constraint violations.
      - maxiter: int
          Maximum number of iterations for dual annealing.

    Returns:
      - numpy.ndarray: Optimized weight vector.
    """

    def penalty_function(w):
        pen = 0.0
        # Enforce equality constraint: sum(w) == target_sum.
        pen += penalty_weight * abs(np.sum(w) - target_sum)

        # Apply CVaR and volatility constraints if enabled.
        if apply_constraints:
            if cvar_constraint is not None:
                cvar_val = cvar_constraint(w)
                if cvar_val < 0:
                    pen += penalty_weight * abs(cvar_val)
            if vol_constraint is not None:
                vol_val = vol_constraint(w)
                if vol_val < 0:
                    pen += penalty_weight * abs(vol_val)
        return pen

    def global_objective(w):
        return chosen_obj(w) + penalty_function(w)

    result = dual_annealing(global_objective, bounds=bounds, maxiter=maxiter)
    if not result.success:
        raise ValueError(
            "Global optimization via dual annealing failed: " + result.message
        )
    return result.x


# Example usage
"""
import numpy as np
from global_optim import global_optimize

# Example: Define your base objective (e.g., negative Sharpe ratio).
def my_obj(w):
    # Example: dummy objective; replace with your real function.
    # Suppose higher return and lower risk is better.
    # Here, just use a placeholder computation.
    return -np.dot(w, np.array([0.1, 0.15, 0.12]))

# Define any constraint functions. For instance:
def cvar_constraint(w):
    # Dummy constraint: cvar_value must be >= 0.
    # Replace with your actual computation.
    return 0.05 - np.var(w)

def vol_constraint(w):
    # Dummy volatility constraint.
    return 0.2 - np.std(w)

# Define bounds for each asset weight, e.g. for three assets.
bounds = [(0, 0.2)] * 3
target_sum = 1.0

# Now call the global optimizer.
optimal_weights = global_optimize(
    chosen_obj=my_obj,
    bounds=bounds,
    target_sum=target_sum,
    apply_constraints=True,
    cvar_constraint=cvar_constraint,
    vol_constraint=vol_constraint,
    penalty_weight=1e6,
    maxiter=1000,
)

"""
