from functools import cached_property
from typing import Optional
import numpy as np
import pandas as pd
import optuna
from scipy.optimize import minimize, minimize_scalar
from sklearn.cluster import KMeans
import torch
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Param,
    Var,
    Constraint,
    Objective,
    SolverFactory,
    maximize,
    Reals,
    exp,
    sqrt,
    log,
    value,
)
from pyomo.opt import SolverStatus, TerminationCondition

from utils import logger
from stat_arb.graph_autoencoder import construct_graph, train_gae
from utils.performance_metrics import sharpe_ratio


class MultiAssetReversion:
    def __init__(
        self,
        prices_df: pd.DataFrame,
        asset_cluster_map: dict[str, int] = None,
        hidden_channels=32,
        num_epochs=200,
        learning_rate=0.01,
        p_value_threshold=0.05,
        n_clusters=42,
        use_adf_filter: bool = False,
        adf_window: int = 9,
        adf_pvalue: float = 0.05,
        use_trend_filter: bool = False,
        short_ma_window: int = 20,
        long_ma_window: int = 42,
    ):
        """
        Multi-asset mean reversion strategy using a hybrid GNN + Johansen approach,
        with OU dynamics and natural-unit scaling for signal generation.

        Args:
            prices_df (pd.DataFrame): Price data with each column as a ticker.
            asset_cluster_map (dict[str, int], optional): Mapping from asset tickers to cluster labels.
                Assets labeled as -1 are considered noise and are excluded.
            hidden_channels (int): Hidden dimension size for the GNN.
            num_epochs (int): Number of epochs for GNN training.
            learning_rate (float): Learning rate for GNN training.
            p_value_threshold (float): P-value threshold for constructing the asset graph.
            n_clusters (int): Number of clusters to segment assets.
            use_adf_filter (bool): If True, use a rolling ADF test to filter non-stationary periods.
            adf_window (int): Window size for the rolling ADF test.
            adf_pvalue (float): p-value threshold for the ADF test.
            use_trend_filter (bool): If True, skip signals that go against a short/long moving average trend.
            short_ma_window (int): Window size for the short moving average.
            long_ma_window (int): Window size for the long moving average.
        """
        self.use_adf_filter = use_adf_filter
        self.adf_window = adf_window
        self.adf_pvalue = adf_pvalue
        self.use_trend_filter = use_trend_filter
        self.short_ma_window = short_ma_window
        self.long_ma_window = long_ma_window

        # Validate and preprocess price data.
        if (prices_df <= 0).any().any():
            raise ValueError("Price data must be strictly positive.")
        start_date = max(prices_df.apply(lambda col: col.first_valid_index()))
        end_date = min(prices_df.apply(lambda col: col.last_valid_index()))
        if start_date is None or end_date is None or start_date > end_date:
            raise ValueError("No overlapping period found among tickers.")
        prices_df = prices_df.loc[start_date:end_date].dropna(axis=0, how="any")

        # Convert to log prices and compute log returns.
        self.prices_df = np.log(prices_df)
        self.returns_df = self.prices_df.diff().dropna()

        # --- Cointegration Basket Construction ---
        if asset_cluster_map is not None:
            filtered_asset_cluster_map = self.filter_asset_cluster_map(
                asset_cluster_map
            )
            self.hedge_ratios, self.spread_series = (
                self.construct_cointegrated_baskets_from_map(filtered_asset_cluster_map)
            )
        else:
            self.hedge_ratios, self.spread_series = self.construct_cointegrated_baskets(
                hidden_channels,
                num_epochs,
                learning_rate,
                p_value_threshold,
                n_clusters,
            )

        # Center the spread.
        self.spread_series = self.spread_series - self.spread_series.mean()

        # Estimate OU parameters (kappa, mu, sigma) on the basket spread.
        self.ou_kappa, self.ou_mu, self.ou_sigma = self.estimate_ou_parameters()

        # Compute a scaling factor to move into natural units:
        #   x_scaled = sqrt(ou_kappa) / ou_sigma * (spread - ou_mu)
        self.scale = np.sqrt(self.ou_kappa) / self.ou_sigma

        # Set a flag for mean reversion.
        self.mean_reverting = self.ou_kappa > 0 and self.ou_sigma > 0
        
        # Allocation computations (Kelly, risk parity, etc.)
        self.kelly_fractions = self.compute_dynamic_kelly()
        self.risk_parity_weights = self.compute_risk_parity_weights()
        self.optimal_params = self.optimize_kelly_risk_parity()

    def filter_asset_cluster_map(
        self, asset_cluster_map: dict[str, int]
    ) -> dict[str, int]:
        valid_assets = set(self.prices_df.columns)
        filtered_map = {
            asset: cluster_id
            for asset, cluster_id in asset_cluster_map.items()
            if asset in valid_assets and cluster_id != -1
        }
        logger.info(
            f"Filtered {len(filtered_map)} assets from {len(asset_cluster_map)} in the original map."
        )
        return filtered_map

    def construct_cointegrated_baskets_from_map(
        self, asset_cluster_map: dict[str, int]
    ):
        hedge_ratios = {}
        basket_spread = pd.Series(0, index=self.prices_df.index)
        clusters = {}
        for asset, cluster_label in asset_cluster_map.items():
            if cluster_label == -1:
                continue
            clusters.setdefault(cluster_label, []).append(asset)
        for cluster_label, assets in clusters.items():
            valid_assets = [
                asset for asset in assets if asset in self.prices_df.columns
            ]
            if len(valid_assets) < 2:
                logger.debug(f"Skipping cluster {cluster_label}")
                continue
            cluster_prices = self.prices_df[valid_assets]
            if cluster_prices.shape[1] < 2:
                hedge_ratios[valid_assets[0]] = 1.0
                basket_spread += cluster_prices.iloc[:, 0]
            else:
                try:
                    result = coint_johansen(cluster_prices, det_order=0, k_ar_diff=1)
                    cointegration_vector = result.evec[:, 0]
                    cointegration_vector /= np.sum(np.abs(cointegration_vector))
                    for asset, weight in zip(valid_assets, cointegration_vector):
                        hedge_ratios[asset] = weight
                    basket_spread += cluster_prices.dot(cointegration_vector)
                except Exception as e:
                    logger.info(f"Cluster {cluster_label} Johansen test error: {e}")
        basket_spread = basket_spread - basket_spread.mean()
        return hedge_ratios, basket_spread

    def construct_cointegrated_baskets(
        self, hidden_channels, num_epochs, learning_rate, p_value_threshold, n_clusters
    ):
        data = construct_graph(self.prices_df, self.returns_df, p_value_threshold)
        gae_model = train_gae(
            data,
            hidden_channels=hidden_channels,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
        )
        gae_model.eval()
        with torch.no_grad():
            latent = gae_model.encode(data.x, data.edge_index).cpu().numpy()
        n_samples = latent.shape[0]
        if n_samples < 2:
            clusters = np.zeros(n_samples, dtype=int)
        else:
            n_clusters = min(n_clusters, n_samples)
            clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(
                latent
            )
        hedge_ratios = {}
        basket_spread = pd.Series(0, index=self.prices_df.index)
        for cluster_id in np.unique(clusters):
            asset_indices = np.where(clusters == cluster_id)[0]
            cluster_assets = self.prices_df.columns[asset_indices]
            cluster_prices = self.prices_df[cluster_assets]
            if cluster_prices.shape[1] < 2:
                hedge_ratios[cluster_assets[0]] = 1.0
                basket_spread += cluster_prices.iloc[:, 0]
            else:
                try:
                    result = coint_johansen(cluster_prices, det_order=0, k_ar_diff=1)
                    cointegration_vector = result.evec[:, 0]
                    cointegration_vector /= np.sum(np.abs(cointegration_vector))
                    for asset, weight in zip(cluster_assets, cointegration_vector):
                        hedge_ratios[asset] = weight
                    basket_spread += cluster_prices.dot(cointegration_vector)
                except Exception as e:
                    logger.info(f"Cluster {cluster_id} Johansen test error: {e}")
        basket_spread = basket_spread - basket_spread.mean()
        return hedge_ratios, basket_spread

    def estimate_ou_parameters(self):
        """
        Estimate OU parameters on the basket spread using OLS:
          Δx = α + β x_(t-1) + ε
        and compute kappa = -β, μ = α/kappa, and σ based on the residuals.
        A fallback is used if the regression sigma is too small compared to the overall spread volatility.
        """
        x = self.spread_series.dropna()
        delta_x = x.diff().dropna()
        x_lag = x.shift(1).dropna()
        delta_x = delta_x.loc[x_lag.index]
        beta, alpha = np.polyfit(x_lag, delta_x, 1)
        kappa = -beta  # assume dt=1
        mu = alpha / kappa if kappa != 0 else 0
        residuals = delta_x - (alpha + beta * x_lag)
        reg_sigma = (
            np.std(residuals) * np.sqrt(2 * kappa / (1 - np.exp(-2 * kappa)))
            if kappa != 0
            else np.std(x)
        )
        overall_sigma = np.std(x)
        sigma = reg_sigma if reg_sigma > 0.1 * overall_sigma else overall_sigma
        return kappa, mu, sigma

    def composite_score(self, metrics):
        """
        Compute a composite score to evaluate strategy performance.
        Here we use Sharpe Ratio * Win Rate * log(Total Trades + 1)
        """
        if metrics["Total Trades"] == 0:
            return 0
        return (
            metrics["Sharpe Ratio"]
            * metrics["Win Rate"]
            * np.log(metrics["Total Trades"] + 1)
        )

    def compute_dynamic_kelly(self, risk_free_rate=0.0):
        kelly_allocations = {}
        for asset in self.returns_df.columns:
            mean_return = self.returns_df[asset].mean() - risk_free_rate
            vol = self.returns_df[asset].std()
            if vol == 0 or np.isnan(vol):
                kelly_allocations[asset] = 0.0
            else:
                raw_kelly = mean_return / (vol**2)
                rolling_vol = (
                    self.returns_df[asset]
                    .rolling(window=30, min_periods=5)
                    .std()
                    .dropna()
                )
                market_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else vol
                adaptive_kelly = raw_kelly / (1 + market_vol)
                kelly_allocations[asset] = max(0, min(adaptive_kelly, 1))
        return pd.Series(kelly_allocations)

    def compute_risk_parity_weights(self):
        volatilities = self.returns_df.std().replace(0, 1e-6)
        inv_vol_weights = 1 / volatilities
        total = inv_vol_weights.sum()
        if total == 0 or np.isnan(total):
            return pd.Series(
                np.repeat(1 / len(inv_vol_weights), len(inv_vol_weights)),
                index=inv_vol_weights.index,
            )
        return inv_vol_weights / total

    def optimize_kelly_risk_parity(self):
        def objective(trial):
            kelly_scaling = trial.suggest_float("kelly_scaling", 0.1, 1.0)
            risk_parity_scaling = trial.suggest_float("risk_parity_scaling", 0.1, 1.0)
            kelly_allocations = self.kelly_fractions * kelly_scaling
            risk_parity_allocations = self.risk_parity_weights * risk_parity_scaling
            final_allocations = kelly_allocations + risk_parity_allocations
            total = final_allocations.sum()
            if total == 0 or np.isnan(total):
                return -1e6
            final_allocations /= total
            portfolio_returns = (self.returns_df * final_allocations).sum(axis=1)
            if portfolio_returns.std() == 0 or np.isnan(portfolio_returns.std()):
                return -1e6
            s = sharpe_ratio(portfolio_returns)
            return s if not np.isnan(s) else -1e6

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=50)
        return study.best_params

    def compute_rolling_adf(
        self, series: pd.Series, window: int, pval_threshold: float
    ) -> pd.Series:
        is_stationary = pd.Series(False, index=series.index)
        for i in range(window, len(series)):
            window_data = series.iloc[i - window : i].dropna()
            if len(window_data) < window:
                continue
            try:
                adf_result = adfuller(window_data, autolag="AIC")
                p_value = adf_result[1]
                is_stationary.iloc[i] = p_value < pval_threshold
            except:
                is_stationary.iloc[i] = False
        return is_stationary

    # @cached_property
    # def calculate_optimal_bounds(self):
    #     """
    #     Solve the free-boundary problem via the method of heat potentials by discretizing the
    #     associated Volterra integral equation on an inhomogeneous grid using Pyomo with IPOPT.

    #     In natural (dimensionless) units the problem is:
    #         E(υ) = f(υ) + ∫_0^υ K(υ, s; b) E(s) ds, 0 ≤ υ ≤ Υ,
    #     with the smooth-fit condition dE/dυ|_(υ=Υ) = 0.

    #     We use the exact expressions:
    #         Π(υ; b) = sqrt(1 - 2*υ) * (b - θ),
    #         K(υ, s; b) = 1/sqrt(2*pi) * ((Π(υ; b) - Π(s; b))/(upsilon - s + eps)) * exp(-((Π(υ; b) - Π(s; b))**2)/(2*(upsilon-s+eps))),
    #         f(υ) = 2*pi*log((1-2*υ)/(1-2*Υ)) + 2*(Π(υ; b) + θ)*log(1-2*Υ).

    #     Here we assume θ = ou_mu and the natural-unit scaling is given by:
    #         scale = sqrt(ou_kappa)/ou_sigma,
    #         x_scaled = scale*(spread - ou_mu).

    #     The free-boundary in original units is then:
    #         stop_loss = ou_mu - b*/scale,  take_profit = ou_mu + b*/scale,
    #     where b* is the optimal boundary (in natural units) found by solving the above system.

    #     NOTE: To avoid math domain errors, T_val is chosen so that Υ = 1 - exp(-2*T_val) < 0.5.
    #     """
    #     # Constants from OU estimation and transformation.
    #     theta = self.ou_mu
    #     scale = np.sqrt(self.ou_kappa) / self.ou_sigma

    #     # Choose a finite horizon T_val small enough so that Υ < 0.5.
    #     T_val = 0.2  # Adjust as needed.
    #     Upsilon = 1 - np.exp(-2 * T_val)

    #     # A small epsilon to avoid division by zero.
    #     eps = 1e-6

    #     # --- Step 1. Build an inhomogeneous grid on υ in [0, Upsilon].
    #     N = 100  # number of grid points
    #     u_uniform = np.linspace(0, 1, N)
    #     c = 5.0  # clustering parameter
    #     grid = Upsilon * (np.exp(u_uniform * c) - 1) / (np.exp(c) - 1)

    #     delta = np.empty(N)
    #     delta[0] = grid[1] - grid[0]
    #     for i in range(1, N):
    #         delta[i] = grid[i] - grid[i - 1]

    #     # --- Step 2. Define the exact functions using Pyomo functions.
    #     def Pi_expr(upsilon, b):
    #         return sqrt(1 - 2 * upsilon) * (b - theta)

    #     def K_expr(upsilon, s, b):
    #         # To avoid division by zero, use (upsilon-s+eps)
    #         denom = upsilon - s + eps
    #         return (
    #             1.0
    #             / sqrt(2 * np.pi)
    #             * ((Pi_expr(upsilon, b) - Pi_expr(s, b)) / denom)
    #             * exp(-((Pi_expr(upsilon, b) - Pi_expr(s, b)) ** 2) / (2 * denom))
    #         )

    #     def f_expr(upsilon, b):
    #         return 2 * np.pi * log((1 - 2 * upsilon) / (1 - 2 * Upsilon)) + 2 * (
    #             Pi_expr(upsilon, b) + theta
    #         ) * log(1 - 2 * Upsilon)

    #     # --- Step 3. Build the Pyomo model.
    #     model = ConcreteModel()
    #     model.I = RangeSet(1, N)

    #     grid_dict = {i: grid[i - 1] for i in model.I}
    #     model.upsilon = Param(model.I, initialize=grid_dict, within=Reals)
    #     delta_dict = {i: delta[i - 1] for i in range(2, N + 1)}
    #     model.delta = Param(RangeSet(2, N), initialize=delta_dict, within=Reals)

    #     # Free-boundary variable b (in natural units).
    #     model.b = Var(bounds=(0.0, grid[-1]), initialize=grid[-1], domain=Reals)

    #     # Unknown function E(υ) at grid points.
    #     model.E = Var(model.I, domain=Reals)

    #     # --- Step 4. Discretize the Volterra integral equation.
    #     def volterra_rule(model, i):
    #         upsilon_i = model.upsilon[i]
    #         b_val = model.b
    #         expr = f_expr(upsilon_i, b_val)
    #         if i == 1:
    #             integral = 0
    #         else:
    #             integral = 0
    #             for j in range(1, i):
    #                 if j < i:
    #                     d = (
    #                         model.delta[j + 1]
    #                         if (j + 1) in model.delta
    #                         else model.delta[j]
    #                     )
    #                     # Use the average of kernel values at upsilon[j] and upsilon[j+1]
    #                     k1 = K_expr(upsilon_i, model.upsilon[j], b_val)
    #                     k2 = K_expr(
    #                         upsilon_i,
    #                         (
    #                             model.upsilon[j + 1]
    #                             if (j + 1) in model.upsilon
    #                             else model.upsilon[j]
    #                         ),
    #                         b_val,
    #                     )
    #                     integral += 0.5 * (k1 * model.E[j] + k2 * model.E[j]) * d
    #         return model.E[i] == expr + integral

    #     model.VolterraConstraint = Constraint(model.I, rule=volterra_rule)

    #     # --- Step 5. Impose the smooth-fit condition at υ = Upsilon.
    #     def smooth_fit_rule(model):
    #         return (model.E[N] - model.E[N - 1]) / (
    #             model.upsilon[N] - model.upsilon[N - 1] + eps
    #         ) == 0.0

    #     model.SmoothFit = Constraint(rule=smooth_fit_rule)

    #     # --- Step 6. Define the objective.
    #     model.obj = Objective(expr=model.E[N], sense=maximize)

    #     # --- Step 7. Solve the model using IPOPT.
    #     solver = SolverFactory("ipopt")
    #     results = solver.solve(model, tee=False)
    #     if (results.solver.status != SolverStatus.ok) or (
    #         results.solver.termination_condition != TerminationCondition.optimal
    #     ):
    #         logger.error(
    #             "Solver did not converge to an optimal solution. Falling back to default bounds."
    #         )
    #         b_star = 2.0  # fallback value in natural units
    #     else:
    #         b_star = value(model.b)

    #     # --- Step 8. Transform back to original units.
    #     stop_loss_unscaled = theta - b_star / scale
    #     take_profit_unscaled = theta + b_star / scale
    #     return stop_loss_unscaled, take_profit_unscaled
    
    # @cached_property
    # def calculate_optimal_bounds(self):
    #     """
    #     Numerically solve the free-boundary problem (in natural units) as derived in Lipton and López de Prado.

    #     In natural (dimensionless) units the problem is:
    #         E(υ) = f(υ; b) + ∫_0^υ K(υ, s; b) E(s) ds,   0 ≤ υ ≤ Υ,
    #     with the smooth-fit condition dE/dυ |_(υ=Υ) = 0.

    #     The functions are defined as:
    #         Π(υ; b) = sqrt(1 - 2*υ) * (b - θ),
    #         K(υ, s; b) = 1/sqrt(2*pi) * ((Π(υ; b) - Π(s; b))/(υ-s+eps)) * exp(-((Π(υ; b) - Π(s; b))**2)/(2*(υ-s+eps))),
    #         f(υ; b) = 2*pi*log((1-2*υ)/(1-2*Υ)) + 2*(Π(υ; b) + θ)*log(1-2*Υ).

    #     Here, θ is ou_mu and Υ = 1 - exp(-2*T_val) for a chosen T_val such that Υ < 0.5.

    #     We solve the integral equation via forward iteration on an inhomogeneous grid and then compute the
    #     backward finite difference derivative at υ = Υ. The free-boundary condition is enforced by minimizing
    #     the absolute derivative.

    #     Finally, we transform the optimal free-boundary (in natural units) back into the original units via:
    #         stop_loss = ou_mu - b*/scale,  take_profit = ou_mu + b*/scale,
    #     where scale = sqrt(ou_kappa)/ou_sigma.
    #     """
    #     # Constants from OU estimation and natural-unit transformation.
    #     theta = self.ou_mu
    #     scale = np.sqrt(self.ou_kappa) / self.ou_sigma  # natural scaling factor
    #     eps = 1e-6  # safeguard against division by zero

    #     # Choose a finite horizon T_val so that Υ < 0.5.
    #     T_val = 0.2  # Adjust as needed
    #     Upsilon = 1 - np.exp(-2 * T_val)

    #     # Define the exact functions:
    #     def Pi(upsilon, b):
    #         return np.sqrt(max(0, 1 - 2 * upsilon)) * (b - theta)

    #     def K(upsilon, s, b):
    #         denom = upsilon - s + eps
    #         return (
    #             1.0
    #             / np.sqrt(2 * np.pi)
    #             * ((Pi(upsilon, b) - Pi(s, b)) / denom)
    #             * np.exp(-((Pi(upsilon, b) - Pi(s, b)) ** 2) / (2 * denom))
    #         )

    #     def f_func(upsilon, b):
    #         return 2 * np.pi * np.log((1 - 2 * upsilon) / (1 - 2 * Upsilon)) + 2 * (
    #             Pi(upsilon, b) + theta
    #         ) * np.log(1 - 2 * Upsilon)

    #     # Build an inhomogeneous grid in υ ∈ [0, Upsilon] using an exponential transformation.
    #     N = 200  # number of grid points; increase for higher accuracy
    #     u_uniform = np.linspace(0, 1, N)
    #     c = 5.0  # clustering parameter
    #     upsilon_grid = Upsilon * (np.exp(u_uniform * c) - 1) / (np.exp(c) - 1)

    #     # Compute grid spacings.
    #     delta = np.diff(upsilon_grid)  # length N-1

    #     # Given a candidate free-boundary b (in natural units), solve for E(upsilon) on the grid by forward iteration.
    #     def solve_E(b):
    #         E = np.zeros(N)
    #         # For each grid point i, approximate:
    #         #   E(upsilon_i) = f(upsilon_i; b) + ∫_0^(upsilon_i) K(upsilon_i, s; b) E(s) ds.
    #         for i in range(N):
    #             upsilon_i = upsilon_grid[i]
    #             # Compute the integral using trapezoidal rule.
    #             if i == 0:
    #                 integral = 0.0
    #             else:
    #                 # Use trapezoidal rule over indices 0 to i.
    #                 s_vals = upsilon_grid[: i + 1]
    #                 E_vals = E[: i + 1]
    #                 # For j=0,...,i, evaluate kernel at upsilon_i, s.
    #                 K_vals = np.array([K(upsilon_i, s, b) for s in s_vals])
    #                 # Compute the integral approximately.
    #                 integral = np.trapz(K_vals * E_vals, s_vals)
    #             E[i] = f_func(upsilon_i, b) + integral
    #         return E

    #     # Define the free-boundary error function.
    #     # We enforce the smooth-fit condition at the last grid point: dE/dυ ≈ (E[N-1] - E[N-2])/(upsilon_grid[N-1]-upsilon_grid[N-2]) == 0.
    #     def free_boundary_error(b):
    #         E = solve_E(b)
    #         dE = (E[-1] - E[-2]) / (upsilon_grid[-1] - upsilon_grid[-2] + eps)
    #         return np.abs(dE)

    #     # Use SciPy's minimize_scalar to find b* that minimizes free_boundary_error.
    #     res = minimize_scalar(
    #         free_boundary_error, bounds=(0.1, upsilon_grid[-1]), method="bounded"
    #     )
    #     if not res.success:
    #         logger.error(
    #             "Free-boundary optimization did not converge. Using default b* = 2.0"
    #         )
    #         b_star = 2.0
    #     else:
    #         b_star = res.x
    #         logger.info(f"Optimized free-boundary in natural units: b* = {b_star}")

    #     # Transform b_star back to original units:
    #     stop_loss_unscaled = theta - b_star / scale
    #     take_profit_unscaled = theta + b_star / scale
    #     return stop_loss_unscaled, take_profit_unscaled
    @cached_property
    def calculate_optimal_bounds(self):
        """
        Optimize stop-loss and take-profit thresholds based on OU dynamics.
        Here the thresholds are defined in terms of deviations from the OU mean.
        For example, bounds might be set at -2σ for long entry and +2σ for short entry.
        """

        def objective(bounds):
            stop_loss, take_profit = bounds
            signals = self.generate_trading_signals(stop_loss, take_profit)
            _, metrics = self.simulate_strategy(signals)
            if metrics["Total Trades"] == 0 or np.isnan(s):
                return 1e6
            s = metrics["Sharpe Ratio"]
            return -s if not np.isnan(s) else np.inf

        # Use bounds scaled by the estimated OU sigma
        bounds = [(-2 * self.ou_sigma, 0), (0, 2 * self.ou_sigma)]

        result = minimize(
            objective, x0=[-1 * self.ou_sigma, 1 * self.ou_sigma], bounds=bounds
        )
        return result.x

    def generate_trading_signals(
        self, stop_loss: float = None, take_profit: float = None
    ) -> pd.DataFrame:
        """
        Generate trading signals based on OU dynamics.
        We first compute the scaled deviation:
            x_scaled = scale * (spread - ou_mu)
        and then compare it to the optimal thresholds (stop_loss and take_profit) which are in the same units.
        Optional ADF and trend filters are applied.
        """
        if not self.mean_reverting:
            # logger.info("Asset is not mean reverting. Skipping signal generation.")
            return pd.DataFrame(index=self.prices_df.index, columns=["Position", "Ticker", "Entry Price", "Exit Price"])
        
        if stop_loss is None or take_profit is None:
            stop_loss, take_profit = self.calculate_optimal_bounds

        # Compute scaled OU deviation.
        x_scaled = self.scale * (self.spread_series - self.ou_mu)

        # Apply stationarity filter if enabled.
        if self.use_adf_filter:
            is_stationary = self.compute_rolling_adf(
                self.spread_series, self.adf_window, self.adf_pvalue
            )
        else:
            is_stationary = pd.Series(True, index=self.spread_series.index)

        # Trend filter based on moving averages of the scaled deviation.
        if self.use_trend_filter:
            short_ma = x_scaled.rolling(
                self.short_ma_window, min_periods=self.short_ma_window
            ).mean()
            long_ma = x_scaled.rolling(
                self.long_ma_window, min_periods=self.long_ma_window
            ).mean()
        else:
            short_ma = pd.Series(0.0, index=x_scaled.index)
            long_ma = pd.Series(0.0, index=x_scaled.index)

        signals = pd.DataFrame(
            index=self.spread_series.index,
            columns=["Position", "Ticker", "Entry Price", "Exit Price"],
        )
        signals[:] = np.nan
        signals["Ticker"] = ", ".join(self.prices_df.columns)
        signals["Position"] = 0
        signals["Entry Price"] = np.nan
        signals["Exit Price"] = np.nan

        position = 0  # 0 = no position, 1 = long, -1 = short.
        entry_value = np.nan
        # A small buffer (epsilon) to avoid premature exit.
        epsilon = 0.1

        for t in signals.index:
            if np.isnan(x_scaled.loc[t]) or not is_stationary.loc[t]:
                signals.loc[t, "Position"] = 0
                continue

            dev = x_scaled.loc[t]

            # Trend filter: if short_ma > long_ma, avoid short; if short_ma < long_ma, avoid long.
            can_short = True
            can_long = True
            if self.use_trend_filter:
                if short_ma.loc[t] > long_ma.loc[t]:
                    can_short = False
                elif short_ma.loc[t] < long_ma.loc[t]:
                    can_long = False

            if position == 0:
                if (dev < stop_loss) and can_long:
                    position = 1
                    entry_value = dev
                    signals.loc[t, "Position"] = 1
                    signals.loc[t, "Entry Price"] = dev
                    signals.loc[t, "Exit Price"] = dev
                elif (dev > take_profit) and can_short:
                    position = -1
                    entry_value = dev
                    signals.loc[t, "Position"] = -1
                    signals.loc[t, "Entry Price"] = dev
                    signals.loc[t, "Exit Price"] = dev
                else:
                    signals.loc[t, "Position"] = 0
            elif position == 1:
                if dev >= -epsilon:  # exit long near the mean
                    signals.loc[t, "Position"] = 0
                    signals.loc[t, "Exit Price"] = dev
                    position = 0
                    entry_value = np.nan
                else:
                    signals.loc[t, "Position"] = 1
                    signals.loc[t, "Entry Price"] = entry_value
                    signals.loc[t, "Exit Price"] = dev
            elif position == -1:
                if dev <= epsilon:  # exit short near the mean
                    signals.loc[t, "Position"] = 0
                    signals.loc[t, "Exit Price"] = dev
                    position = 0
                    entry_value = np.nan
                else:
                    signals.loc[t, "Position"] = -1
                    signals.loc[t, "Entry Price"] = entry_value
                    signals.loc[t, "Exit Price"] = dev

        signals["Entry Price"] = signals["Entry Price"].ffill()
        signals["Exit Price"] = signals["Exit Price"].ffill()
        return signals

    def simulate_strategy(self, signals: pd.DataFrame) -> tuple:
        """
        Simulate performance using numeric signals.
        Compute returns as the change in the basket spread multiplied by the lagged position.
        """
        strat_returns = self.spread_series.diff().dropna() * signals["Position"].shift(
            1
        ).fillna(0)
        strat_returns = strat_returns.dropna()
        metrics = {
            "Total Trades": (signals["Position"].diff().abs() > 0).sum(),
            "Sharpe Ratio": sharpe_ratio(strat_returns),
            "Win Rate": (strat_returns > 0).mean(),
            "Avg Return": strat_returns.mean(),
        }
        return strat_returns, metrics

    def optimize_and_trade(self):
        stop_loss, take_profit = self.calculate_optimal_bounds
        logger.info(
            f"Optimal stop_loss: {stop_loss:.4f}, take_profit: {take_profit:.4f}"
        )
        signals = self.generate_trading_signals(stop_loss, take_profit)
        returns, metrics = self.simulate_strategy(signals)
        hedge_ratios_series = pd.Series(
            self.hedge_ratios, index=self.prices_df.columns
        ).fillna(0)
        return {
            "Optimal Stop-Loss": stop_loss,
            "Optimal Take-Profit": take_profit,
            "Metrics": metrics,
            "Signals": signals,
            "Hedge Ratios": hedge_ratios_series.to_dict(),
        }
