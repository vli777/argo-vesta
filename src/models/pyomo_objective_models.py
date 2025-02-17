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
    cvar_limit: float = None,  # Optional CVaR constraint
    alpha: float = 0.05,  # Default CVaR threshold
):
    """
    Build a Pyomo model for the Omega ratio using the linear-fractional formulation.

    The formulation uses additional variables y, z, and q:

       max_{y,q,z}  y^T E(r) - target * z
       s.t.
         y^T E(r) >= target * z,
         sum(y) = z,
         sum(q) = 1,
         q[j] >= target * z - y^T r_j  for each observation j,
         (if no shorting) y >= 0,
         y <= max_weight * z,
         z >= 0.

    After solving, the portfolio weights are recovered as w = y / z.
    """
    model = pyo.ConcreteModel()
    T, n = returns.shape
    assets = list(range(n))
    obs = list(range(T))
    model.assets = pyo.Set(initialize=assets)
    model.obs = pyo.Set(initialize=obs)

    # For a robust estimate of expected returns, here we simply use the sample mean.
    # (You could substitute a trimmed mean if desired.)
    mu_robust = returns.mean(axis=0).values

    # Decision variables:
    model.y = pyo.Var(model.assets, domain=pyo.Reals)
    model.z = pyo.Var(domain=pyo.NonNegativeReals)
    model.q = pyo.Var(model.obs, domain=pyo.NonNegativeReals)

    # Expected Return Constraint: y^T mu_robust >= target * z.
    # model.exp_return_constraint = pyo.Constraint(
    #     expr=sum(model.y[i] * mu_robust[i] for i in model.assets) >= target * model.z
    # )

    # Scaling constraint: sum(y) == z.
    model.scaling_constraint = pyo.Constraint(
        expr=sum(model.y[i] for i in model.assets) == model.z
    )

    # **Portfolio Variance**
    model.port_variance = pyo.Expression(
        expr=sum(
            (model.y[i] / model.z) * cov.iloc[i, j] * (model.y[j] / model.z)
            for i in model.assets
            for j in model.assets
        )
    )

    # **Portfolio Volatility**
    model.port_vol = pyo.Expression(expr=pyo.sqrt(model.port_variance + 1e-8))

    # Normalize tail loss auxiliary variables: sum(q) == 1.
    model.q_norm = pyo.Constraint(expr=sum(model.q[j] for j in model.obs) == 1)

    # Omega Shortfall Constraint: For each historical observation, enforce:
    #   q[j] >= target*z - sum(y[i]*r[j, i])
    def obs_constraint_rule(model, j):
        return model.q[j] >= target * model.z - sum(
            model.y[i] * returns.iloc[j, i] for i in model.assets
        )

    model.obs_constraints = pyo.Constraint(model.obs, rule=obs_constraint_rule)

    # Upper Bound the y variables (weights).
    def upper_bound_rule(model, i):
        return model.y[i] <= max_weight * model.z

    model.upper_bound = pyo.Constraint(model.assets, rule=upper_bound_rule)

    if not allow_short:

        def nonnegative_rule(model, i):
            return model.y[i] >= 0

        model.nonnegative = pyo.Constraint(model.assets, rule=nonnegative_rule)

    # **Portfolio Volatility Constraint**
    if vol_limit:
        model.vol_constraint = pyo.Constraint(expr=model.port_vol <= vol_limit)

    # **CVaR Constraint (Only if `returns` is provided)**
    if cvar_limit is not None and returns is not None:
        T = returns.shape[0]
        model.obs = pyo.Set(initialize=range(T))

        # Auxiliary variables for CVaR in the Omega model
        model.q_cvar = pyo.Var(model.obs, domain=pyo.NonNegativeReals)
        model.eta = pyo.Var(domain=pyo.Reals)  # VaR-like variable

        # For each scenario j: enforce q[j] >= (unscaled loss) - eta.
        def omega_cvar_rule(model, j):
            unscaled_loss = -sum(model.y[i] * returns.iloc[j, i] for i in model.assets)
            return model.q_cvar[j] >= unscaled_loss - model.eta

        model.cvar_q_constraints = pyo.Constraint(model.obs, rule=omega_cvar_rule)

        # CVaR definition in the homogeneous (y,z) space:
        # When divided by z, CVaR = eta/z + (1/(alpha*T)) * sum_j q[j]/z.
        # Multiply through by z to get:
        model.cvar_constraint = pyo.Constraint(
            expr=model.eta + (1 / (alpha * T)) * sum(model.q_cvar[j] for j in model.obs)
            <= cvar_limit * model.z
        )

    # Note: The recovered weights will be w = y / z.
    # Objective: maximize y^T mu_robust - target * z.
    # We express this as minimizing its negative.
    model.obj = pyo.Objective(
        expr=-(sum(model.y[i] * mu_robust[i] for i in model.assets) - target * model.z),
        sense=pyo.minimize,
    )
    return model
