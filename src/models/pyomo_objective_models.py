# File: objectives.py
import pyomo.environ as pyo
import numpy as np
import pandas as pd


def build_sharpe_model(
    cov: pd.DataFrame,
    mu: np.ndarray,
    target_sum: float,
    max_weight: float = 0.2,
    allow_short: bool = False,
    vol_limit: float = None,  # Optional volatility constraint
    cvar_limit: float = None,  # Optional CVaR constraint (upper bound on CVaR)
    min_return: float = None,  # Optional min returns
    alpha: float = 0.05,  # Tail probability for CVaR (e.g., 5%)
    returns: pd.DataFrame = None,  # Historical returns (scenarios) needed if using CVaR
):
    """
    Build a Pyomo model to maximize the Sharpe ratio with optional volatility & CVaR constraints.

    The CVaR (Conditional Value-at-Risk) is modeled in its standard form. In each scenario j,
    define the loss as the negative portfolio return:
         loss_j = - sum_i (w_i * returns.iloc[j, i])
    Then, introducing auxiliary variables q[j] >= 0 and a variable z (interpreted as VaR),
    we require for every scenario:
         q[j] >= loss_j - z
    The CVaR is then given by:
         CVaR = z + (1/(alpha * T)) * sum_j q[j]
    and we enforce CVaR <= cvar_limit.

    Args:
        cov (pd.DataFrame): Covariance matrix.
        mu (np.ndarray): Expected returns (1D array).
        target_sum (float): Sum of portfolio weights (usually 1).
        max_weight (float): Maximum weight per asset.
        allow_short (bool): Allow short selling.
        vol_limit (float): Optional volatility constraint.
        cvar_limit (float): Optional CVaR limit.
        min_return (float): Optional minimum return.
        alpha (float): Tail probability for CVaR (default 5%).
        returns (pd.DataFrame): Historical returns (scenarios), required if `cvar_limit` is set.

    Returns:
        Pyomo optimization model.
    """
    model = pyo.ConcreteModel()
    n = len(mu)
    assets = list(range(n))
    model.assets = pyo.Set(initialize=assets)

    # Decision Variables: Portfolio Weights
    def weight_bounds(model, i):
        return (-max_weight, max_weight) if allow_short else (0, max_weight)

    model.w = pyo.Var(model.assets, domain=pyo.Reals, bounds=weight_bounds)

    # Constraint: Net Exposure - Weights Sum to Target
    model.weight_sum = pyo.Constraint(
        expr=sum(model.w[i] for i in model.assets) == target_sum
    )

    # Constraint: Gross Exposure if allowing short positions
    if allow_short:
        model.z = pyo.Var(model.assets, domain=pyo.NonNegativeReals)

        # Enforce z_i >= |w_i|
        def abs_constraint_rule(model, i):
            return model.z[i] >= model.w[i]

        model.abs_constraint_pos = pyo.Constraint(
            model.assets, rule=abs_constraint_rule
        )

        def abs_constraint_rule_neg(model, i):
            return model.z[i] >= -model.w[i]

        model.abs_constraint_neg = pyo.Constraint(
            model.assets, rule=abs_constraint_rule_neg
        )
        # Now, enforce gross exposure constraint (e.g., 1.3)
        gross_target = 1.3
        model.gross_exposure = pyo.Constraint(
            expr=sum(model.z[i] for i in model.assets) <= gross_target
        )

    # Portfolio Return Expression (using expected returns)
    model.port_return = pyo.Expression(
        expr=sum(model.w[i] * mu.iloc[i] for i in model.assets)
    )

    # Portfolio Variance Expression
    model.port_variance = pyo.Expression(
        expr=sum(
            model.w[i] * cov.iloc[i, j] * model.w[j]
            for i in model.assets
            for j in model.assets
        )
    )

    # Portfolio Volatility Expression
    model.port_vol = pyo.Expression(expr=pyo.sqrt(model.port_variance + 1e-8))

    # Objective: Maximize Sharpe Ratio (minimizing negative Sharpe ratio)
    model.obj = pyo.Objective(
        expr=-model.port_return / model.port_vol, sense=pyo.minimize
    )

    # Optional Returns Constraint, assuming mu is a pd.Series:
    if min_return is not None:
        mu_array = mu.to_numpy()
        model.min_return = pyo.Constraint(
            expr=sum(model.w[i] * mu_array[i] for i in model.assets)
            >= float(min_return)
        )

    # Optional Volatility Constraint
    if vol_limit is not None:
        model.vol_constraint = pyo.Constraint(expr=model.port_vol <= vol_limit)

    # Optional CVaR Constraint
    if cvar_limit is not None and returns is not None:
        T = returns.shape[0]
        model.obs = pyo.Set(initialize=range(T))
        # q[j] are slack variables (excess loss over z) and must be nonnegative
        model.q = pyo.Var(model.obs, domain=pyo.NonNegativeReals)
        # z represents the VaR (can be any real number)
        model.z = pyo.Var(domain=pyo.Reals)
        # For each scenario j, ensure:
        #    q[j] >= (- sum_i w[i]*returns.iloc[j, i]) - z
        # where - sum_i w[i]*returns.iloc[j, i] is the loss in scenario j.
        model.q_constraints = pyo.ConstraintList()
        for j in model.obs:
            model.q_constraints.add(
                model.q[j]
                >= -sum(model.w[i] * returns.iloc[j, i] for i in model.assets) - model.z
            )
        #    CVaR = z + (1/(alpha * T)) * sum_j q[j] <= cvar_limit
        model.cvar_constraint = pyo.Constraint(
            expr=model.z + (1 / (alpha * T)) * sum(model.q[j] for j in model.obs)
            <= cvar_limit
        )

    return model


def build_omega_model(
    cov: pd.DataFrame,
    returns: pd.DataFrame,
    target: float,
    target_sum: float,
    max_weight: float,
    allow_short: bool,
    vol_limit: float = None,  # Optional volatility constraint
    cvar_limit: float = None,  # Optional CVaR constraint (upper bound on CVaR)
    alpha: float = 0.05,  # Tail probability for CVaR (default 5%)
):
    """
    Build a Pyomo model for the Omega ratio using the linear-fractional formulation.

    The formulation uses additional variables y, z, and q:
       max_{y,q,z}  y^T E(r) - target * z
       s.t.
         y^T E(r) >= target * z,
         sum(y) = z,
         sum(q) = 1,
         q[j] >= target*z - sum_i (y[i]*r[j,i])  for each observation j,
         (if no shorting) y >= 0,
         y <= max_weight * z,
         z >= 0.

    After solving, portfolio weights are recovered as w = y / z.

    Optional risk constraints:
      - If vol_limit is provided, enforce that the portfolio volatility (computed from w)
        does not exceed vol_limit.
      - If cvar_limit is provided, enforce that the computed CVaR (in the homogeneous space)
        is less than or equal to cvar_limit.

    Additionally, if allow_short is True, a gross exposure constraint is added:
      sum(|y[i]|) <= gross_target * z, which ensures that the gross exposure of w = y/z
      is controlled (e.g. gross_target=1.3).

    Args:
        cov (pd.DataFrame): Covariance matrix.
        returns (pd.DataFrame): Historical returns (T x n).
        target (float): Target return (for the Omega formulation).
        target_sum (float): Sum of portfolio weights (usually 1).
        max_weight (float): Maximum weight per asset.
        allow_short (bool): Allow short selling.
        vol_limit (float): Optional volatility constraint.
        cvar_limit (float): Optional CVaR constraint.
        alpha (float): Tail probability for CVaR.

    Returns:
        Pyomo optimization model.
    """
    model = pyo.ConcreteModel()
    T, n = returns.shape
    assets = list(range(n))
    obs = list(range(T))
    model.assets = pyo.Set(initialize=assets)
    model.obs = pyo.Set(initialize=obs)

    # Use sample mean of returns as robust expected returns.
    mu_robust = returns.mean(axis=0).values

    # Decision variables:
    model.y = pyo.Var(
        model.assets, domain=pyo.Reals
    )  # homogeneous representation variable
    model.z = pyo.Var(domain=pyo.NonNegativeReals)
    model.q = pyo.Var(model.obs, domain=pyo.NonNegativeReals)

    # Scaling constraint: sum(y) == z.
    model.scaling_constraint = pyo.Constraint(
        expr=sum(model.y[i] for i in model.assets) == model.z
    )

    # Optional: if shorting is not allowed, force y >= 0.
    if not allow_short:

        def nonnegative_rule(model, i):
            return model.y[i] >= 0

        model.nonnegative = pyo.Constraint(model.assets, rule=nonnegative_rule)
    else:
        # If shorting is allowed, we add a gross exposure constraint on y.
        # Introduce auxiliary variable for |y[i]|
        model.gross_y = pyo.Var(model.assets, domain=pyo.NonNegativeReals)

        def abs_constraint_pos(model, i):
            return model.gross_y[i] >= model.y[i]

        model.abs_constraint_pos = pyo.Constraint(model.assets, rule=abs_constraint_pos)

        def abs_constraint_neg(model, i):
            return model.gross_y[i] >= -model.y[i]

        model.abs_constraint_neg = pyo.Constraint(model.assets, rule=abs_constraint_neg)
        # Enforce gross exposure: sum(|y[i]|) <= gross_target * z.
        gross_target = 1.3  # adjust as desired
        model.gross_exposure = pyo.Constraint(
            expr=sum(model.gross_y[i] for i in model.assets) <= gross_target * model.z
        )

    # Upper bound constraint: y[i] <= max_weight * z.
    def upper_bound_rule(model, i):
        return model.y[i] <= max_weight * model.z

    model.upper_bound = pyo.Constraint(model.assets, rule=upper_bound_rule)

    # Define portfolio variance and volatility (w = y/z).
    model.port_variance = pyo.Expression(
        expr=sum(
            (model.y[i] / model.z) * cov.iloc[i, j] * (model.y[j] / model.z)
            for i in model.assets
            for j in model.assets
        )
    )
    model.port_vol = pyo.Expression(expr=pyo.sqrt(model.port_variance + 1e-8))

    # Normalize tail loss auxiliary variables: sum(q) == 1.
    model.q_norm = pyo.Constraint(expr=sum(model.q[j] for j in model.obs) == 1)

    # Omega shortfall constraint: for each observation j:
    def obs_constraint_rule(model, j):
        return model.q[j] >= target * model.z - sum(
            model.y[i] * returns.iloc[j, i] for i in model.assets
        )

    model.obs_constraints = pyo.Constraint(model.obs, rule=obs_constraint_rule)

    # Optional Volatility Constraint: if vol_limit is provided, enforce portfolio volatility <= vol_limit.
    if vol_limit is not None:
        model.vol_constraint = pyo.Constraint(expr=model.port_vol <= vol_limit)

    # Optional CVaR Constraint:
    if cvar_limit is not None and returns is not None:
        T_obs = returns.shape[0]
        # Reinitialize obs set to ensure it covers T_obs.
        model.obs = pyo.Set(initialize=range(T_obs))
        # Auxiliary variables for CVaR.
        model.q_cvar = pyo.Var(model.obs, domain=pyo.NonNegativeReals)
        model.eta = pyo.Var(domain=pyo.Reals)  # VaR-like variable

        def omega_cvar_rule(model, j):
            unscaled_loss = -sum(model.y[i] * returns.iloc[j, i] for i in model.assets)
            return model.q_cvar[j] >= unscaled_loss - model.eta

        model.cvar_q_constraints = pyo.Constraint(model.obs, rule=omega_cvar_rule)
        # Enforce: CVaR = eta + (1/(alpha*T_obs))*sum_j q_cvar[j] <= cvar_limit * model.z
        model.cvar_constraint = pyo.Constraint(
            expr=model.eta
            + (1 / (alpha * T_obs)) * sum(model.q_cvar[j] for j in model.obs)
            <= cvar_limit * model.z
        )

    # Objective: maximize (y^T mu_robust - target*z) which is equivalent to maximizing the Omega ratio.
    # We express this as minimizing its negative.
    model.obj = pyo.Objective(
        expr=-(sum(model.y[i] * mu_robust[i] for i in model.assets) - target * model.z),
        sense=pyo.minimize,
    )

    return model
