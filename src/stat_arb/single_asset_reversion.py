from functools import cached_property
from typing import Optional, Union
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
import plotly.graph_objects as go

from utils import logger
from utils.performance_metrics import sharpe_ratio


class SingleAssetReversion:
    """
    Single-Asset Mean-Reversion Strategy using an OU (Ornstein-Uhlenbeck) model.

    This class implements:
    1. OU parameter estimation (kappa, mu, sigma).
    2. A free-boundary (stop-loss & take-profit) determination based on
       a heat-potential approach for the OU process, as discussed in
       standard references on the 'method of heat potentials' or 'Green’s function'
       for the heat equation applied to hitting times / payoff functionals of
       the OU process.
    3. Generating trading signals and simulating strategy performance.
    """

    def __init__(
        self,
        prices: pd.Series,
        returns_df: pd.DataFrame,
        dt: float = 1.0,
        T: int = 10,
        max_leverage: float = 1.0,
    ):
        """
        Initialize the strategy with price data and returns DataFrame.

        Args:
            prices (pd.Series): Time series of stock prices.
            returns_df (pd.DataFrame): DataFrame of asset returns.
            dt (float): Time step (default 1 day).
            T (int): Lookback period for OU estimation.
            max_leverage (float): Maximum allowed leverage for position sizing.
        """
        self.prices = prices
        self.returns_df = returns_df
        self.dt = dt
        self.T = T
        self.max_leverage = max_leverage
        self.kappa, self.mu, self.sigma = self.estimate_ou_parameters()
        self.valid_ou = self.kappa > 0 and self.sigma > 0

    def estimate_ou_parameters(self):
        """
        Estimate Ornstein-Uhlenbeck (OU) process parameters using OLS regression on log prices.

        We assume the log-price follows an OU process:
            dX_t = kappa*(mu - X_t) dt + sigma dW_t.

        In discrete form, one can regress (X_{t+1} - X_t) on X_t to estimate kappa and mu.

        Returns:
            tuple: Estimated (kappa, mu, sigma).
        """
        log_prices = np.log(self.prices)
        delta_x = log_prices.diff().dropna()
        X_t = log_prices.shift(1).dropna()

        # Align response variable Y_t with X_t
        Y_t = delta_x.loc[X_t.index]

        # OLS fit: Y_t = alpha + beta * X_t
        beta, alpha = np.polyfit(X_t, Y_t, 1)

        # From the discrete-time OU regression relationships:
        #   kappa = -beta / dt
        #   mu = alpha / kappa
        kappa = -beta / self.dt
        mu = alpha / kappa if kappa != 0 else 0

        # Residual-based sigma estimate
        residuals = Y_t - (alpha + beta * X_t)
        sigma = (
            np.std(residuals) * np.sqrt(2 * kappa / (1 - np.exp(-2 * kappa * self.dt)))
            if kappa > 0
            else 0.0
        )

        return kappa, mu, sigma

    # def calculate_optimal_bounds(self):
    #     """
    #     Optimize stop-loss and take-profit levels to maximize composite score.
    #     Returns:
    #         tuple: (optimal_stop_loss, optimal_take_profit)
    #     """

    #     def objective(bounds):
    #         stop_loss, take_profit = bounds
    #         signals = self.generate_trading_signals(stop_loss, take_profit)
    #         _, metrics = self.simulate_strategy(signals)
    #         return -self.composite_score(metrics)

    #     bounds = [(-2 * self.sigma, 0), (0, 2 * self.sigma)]
    #     opt_result = minimize(objective, x0=[-1, 1], bounds=bounds)
    #     return opt_result.x

    @cached_property
    def calculate_optimal_bounds(self):
        """
        We consider an OU process X(t) with drift kappa*(mu - X), and want to find optimal
        boundaries (stop-loss and take-profit) in log-price space. The solution can be written
        via a *heat potential* (Green’s function of the 1D heat equation) which leads to an
        integral equation of the form:

            E(upsilon) = f(upsilon; b) + ∫ K(upsilon, s; b) E(s) ds,   0 ≤ upsilon ≤ Upsilon,

        with boundary condition (smooth-fit) dE/d(upsilon) at upsilon=Upsilon = 0.

        Here, upsilon is a dimensionless variable (0 <= upsilon <= Upsilon < 0.5), and K is the
        heat kernel. The unknown b* is found by ensuring the boundary condition is met.

        Once b* is found in dimensionless units, we transform back to the original price scale:
            stop_loss = mu - b* / scale,
            take_profit = mu + b* / scale,
        where scale = sqrt(kappa) / sigma.

        -----------------------------------------------------------------------------------------
        Returns:
            (stop_loss, take_profit): The optimal boundaries in original log-price scale.
        """
        theta = self.mu
        # If parameters are invalid, force scale=1 and log a warning.
        if self.valid_ou:
            scale = np.sqrt(self.kappa) / self.sigma
        else:
            scale = 1.0
            # logger.warning(
            #     f"Invalid OU parameters (kappa={self.kappa}, sigma={self.sigma}); setting scale=1."
            # )

        eps = 1e-6
        # Choose a dimensionless upper bound for upsilon
        T_val = 0.2  # for demonstration, smaller T_val => smaller domain
        Upsilon = 1 - np.exp(
            -2 * T_val
        )  # must be < 0.5 for sqrt(1-2*upsilon) to be real

        # Dimensionless transform for log-price deviation
        # Pi(upsilon, b) = sqrt(1 - 2 upsilon)*(b - theta)
        def Pi(upsilon, b):
            return np.sqrt(max(0.0, 1 - 2 * upsilon)) * (b - theta)

        # Heat kernel K(upsilon, s; b).
        # The integral equation approach is akin to a convolution with the
        # fundamental solution of the heat equation (a 'heat potential').
        def heat_kernel(upsilon, s, b):
            denom = upsilon - s + eps
            return (
                1.0
                / np.sqrt(2 * np.pi)
                * ((Pi(upsilon, b) - Pi(s, b)) / denom)
                * np.exp(-((Pi(upsilon, b) - Pi(s, b)) ** 2) / (2 * denom))
            )

        # Source function f(upsilon, b), representing immediate 'payoff' or cost
        # in the integral equation.
        def f_func(upsilon, b):
            return 2 * np.pi * np.log((1 - 2 * upsilon) / (1 - 2 * Upsilon)) + 2 * (
                Pi(upsilon, b) + theta
            ) * np.log(1 - 2 * Upsilon)

        # Discretize 0..Upsilon in a non-uniform grid for numerical stability
        N = 200
        u_uniform = np.linspace(0, 1, N)
        c = 5.0  # exponent scale
        upsilon_grid = Upsilon * (np.exp(u_uniform * c) - 1) / (np.exp(c) - 1)

        def solve_E(b):
            """
            Solve the integral equation for E(upsilon) for a fixed b using
            a simple forward integration + trapezoidal rule for the integral.
            """
            E = np.zeros(N)
            for i in range(N):
                upsilon_i = upsilon_grid[i]
                if i == 0:
                    # Start from E(0) = f(0; b), no integral portion
                    integral = 0.0
                else:
                    # Numerically approximate the integral ∫_0^upsilon_i K(upsilon_i, s; b)*E(s) ds
                    s_vals = upsilon_grid[: i + 1]
                    E_vals = E[: i + 1]
                    K_vals = np.array([heat_kernel(upsilon_i, s, b) for s in s_vals])
                    integral = np.trapz(K_vals * E_vals, s_vals)
                E[i] = f_func(upsilon_i, b) + integral
            return E

        def free_boundary_error(b):
            """
            For the free-boundary problem, we enforce a smooth-fit condition at upsilon=Upsilon:
            dE/d(upsilon)|_{upsilon=Upsilon} = 0.
            We approximate it with a finite difference.
            """
            E = solve_E(b)
            # Approximate derivative near the boundary (last two points)
            dE = (E[-1] - E[-2]) / (upsilon_grid[-1] - upsilon_grid[-2] + eps)
            return np.abs(dE)

        # Minimize the boundary mismatch to find b*
        res = minimize_scalar(
            free_boundary_error, bounds=(0.1, upsilon_grid[-1]), method="bounded"
        )
        if not res.success:
            logger.warning(
                "Free-boundary optimization did not converge. Using default b* = 2.0."
            )
            b_star = 2.0
        else:
            b_star = res.x
            # logger.info(
            #     f"Optimized free-boundary in dimensionless units: b* = {b_star}"
            # )

        # Map b_star back to log-price scale
        stop_loss_unscaled = theta - b_star / scale
        take_profit_unscaled = theta + b_star / scale
        return stop_loss_unscaled, take_profit_unscaled

    def generate_trading_signals(
        self, stop_loss: Optional[float] = None, take_profit: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate trading signals using a stateful loop.
        - 1 for BUY, -1 for SELL, and 0 for NO_SIGNAL.
        - The very first day is always neutral (0).

        Args:
            stop_loss (Optional[float]): Lower deviation threshold to enter a long position.
            take_profit (Optional[float]): Upper deviation threshold to exit the position.

        Returns:
            pd.DataFrame: Signals with "Position", "Entry Price", and "Exit Price".
        """
        # Return no signals if the asset is not well described by an OU process.
        if not self.valid_ou:
            # logger.info("Asset does not fit the OU model well; no signals generated.")
            return pd.DataFrame(
                index=self.prices.index,
                columns=["Position", "Entry Price", "Exit Price"],
            )

        # Compute optimal bounds if not provided
        if stop_loss is None or take_profit is None:
            stop_loss, take_profit = self.calculate_optimal_bounds

        # Compute log prices and deviations from the mean (or μ)
        log_prices = np.log(self.prices)
        deviations = log_prices - self.mu

        # Initialize signals DataFrame
        signals = pd.DataFrame(
            index=self.prices.index, columns=["Position", "Entry Price", "Exit Price"]
        )
        signals[:] = np.nan

        in_position = False
        entry_price = None

        for idx, date in enumerate(self.prices.index):
            price = self.prices.iloc[idx]
            dev = deviations.iloc[idx]

            # Always mark the first day as neutral (0)
            if idx == 0:
                signals.loc[date, "Position"] = 0
                continue

            if not in_position:
                # Look for entry: if deviation is sufficiently low, enter long.
                if dev < stop_loss:
                    in_position = True
                    entry_price = price
                    signals.loc[date, "Position"] = 1  # BUY
                    signals.loc[date, "Entry Price"] = price
                else:
                    signals.loc[date, "Position"] = 0  # NO_SIGNAL
            else:
                # When in position, by default mark as neutral.
                signals.loc[date, "Position"] = 0  # NO_SIGNAL
                # Check exit condition: if deviation exceeds take_profit, exit.
                if dev > take_profit:
                    signals.loc[date, "Position"] = -1  # SELL
                    signals.loc[date, "Entry Price"] = entry_price
                    signals.loc[date, "Exit Price"] = price
                    in_position = False
                    entry_price = None

        return signals

    def simulate_strategy(self, signals: pd.DataFrame) -> tuple:
        """
        Simulate strategy performance using numeric signals.
        - 1 for BUY, -1 for SELL, and 0 for NO_SIGNAL.
        """
        trades = signals.dropna(subset=["Entry Price", "Exit Price"])
        if trades.empty:
            # Return a zero-filled series over the same index as prices
            zero_returns = pd.Series(0, index=self.prices.index)
            metrics = {
                "Total Trades": 0,
                "Sharpe Ratio": 0,
                "Win Rate": 0,
            }
            return zero_returns, metrics

        # Force conversion to float in case the types are mixed
        entry_prices = np.array(trades["Entry Price"].values, dtype=float)
        exit_prices = np.array(trades["Exit Price"].values, dtype=float)
        returns = np.log(exit_prices / entry_prices)
        returns_series = pd.Series(returns, index=trades.index)

        # Calculate performance metrics
        win_rate = (returns > 0).mean()
        sharpe_r = sharpe_ratio(
            returns_series, entries_per_year=252, risk_free_rate=0.0
        )
        metrics = {
            "Total Trades": len(trades),
            "Sharpe Ratio": sharpe_r,
            "Win Rate": win_rate,
        }
        return returns_series, metrics

    def composite_score(self, metrics):
        """
        Compute a composite score to evaluate strategy performance.
        """
        if metrics["Total Trades"] == 0:
            return 0
        return (
            metrics["Sharpe Ratio"]
            * metrics["Win Rate"]
            * np.log(metrics["Total Trades"] + 1)
        )

    def run_strategy(self):
        """
        Run the complete strategy: optimize bounds, generate signals, simulate strategy.
        Returns:
            tuple: (signals DataFrame, metrics dictionary)
        """
        if not self.valid_ou:
            # logger.info("Asset does not fit the OU model well; strategy not run.")
            return None, {"Message": "Asset does not fit the OU model well."}

        stop_loss, take_profit = self.calculate_optimal_bounds
        logger.info(
            f"Optimal stop_loss: {stop_loss:.4f}, take_profit: {take_profit:.4f}"
        )
        signals = self.generate_trading_signals(stop_loss, take_profit)
        _, metrics = self.simulate_strategy(signals)
        logger.info("Strategy Metrics:", metrics)
        return signals, metrics

    def compute_optimal_kelly_risk_parity(self, n_trials: int = 50):
        """
        Optimize Kelly Fraction and Risk Parity Allocation jointly using Optuna.
        Also adjusts leverage dynamically based on market volatility.

        Args:
            n_trials (int): Number of Optuna trials.

        Returns:
            dict: Optimized parameters.
        """

        def objective(trial):
            kelly_fraction = trial.suggest_float("kelly_fraction", 0.01, 1.0, log=True)
            risk_parity_scaling = trial.suggest_float("risk_parity_scaling", 0.1, 2.0)
            rolling_volatility = (
                self.returns_df.rolling(window=30, min_periods=10).std().iloc[-1]
            )
            dynamic_kelly_fraction = kelly_fraction / (1 + rolling_volatility.mean())
            rolling_vols = (
                self.returns_df.rolling(window=30, min_periods=10).std().iloc[-1]
            )
            inv_vol_weights = 1 / rolling_vols
            risk_parity_allocation = inv_vol_weights / inv_vol_weights.sum()
            adjusted_allocation = (
                dynamic_kelly_fraction * risk_parity_scaling * risk_parity_allocation
            )
            adjusted_allocation /= adjusted_allocation.sum()
            stop_loss, take_profit = self.calculate_optimal_bounds
            signals = self.generate_trading_signals(stop_loss, take_profit)
            _, metrics = self.simulate_strategy(signals)
            return self.composite_score(metrics)

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)

        best_params = study.best_params
        logger.info(f"Optimized Kelly & Risk Parity: {best_params}")
        return best_params
