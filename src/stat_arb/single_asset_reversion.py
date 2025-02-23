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
        # Check if the asset is mean reverting (OU process must have positive kappa and sigma)
        self.mean_reverting = self.kappa > 0 and self.sigma > 0
        # logger.debug(
        #     f"\nEstimated parameters: kappa={self.kappa:.4f}, mu={self.mu:.4f}, sigma={self.sigma:.4f}"
        # )

    def estimate_ou_parameters(self):
        """
        Estimate Ornstein-Uhlenbeck (OU) process parameters using OLS regression on log prices.

        Returns:
            tuple: Estimated (kappa, mu, sigma).
        """
        log_prices = np.log(self.prices)
        delta_x = log_prices.diff().dropna()
        X_t = log_prices.shift(1).dropna()
        # Align response variable Y_t with X_t
        Y_t = delta_x.loc[X_t.index]
        beta, alpha = np.polyfit(X_t, Y_t, 1)
        kappa = -beta / self.dt
        mu = alpha / kappa if kappa != 0 else 0
        residuals = Y_t - (alpha + beta * X_t)
        sigma = np.std(residuals) * np.sqrt(
            2 * kappa / (1 - np.exp(-2 * kappa * self.dt))
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
        Numerically solve the free-boundary problem (in natural units) for the SingleAssetReversion model.

        In natural (dimensionless) units the problem is:
            E(υ) = f(υ; b) + ∫_0^υ K(υ, s; b) E(s) ds,  0 ≤ υ ≤ Υ,
        with the smooth-fit condition dE/dυ|_(υ=Υ) = 0.

        The functions are defined as:
            Π(υ; b) = sqrt(1 - 2*υ) * (b - θ),
            K(υ, s; b) = 1/sqrt(2*pi) * ((Π(υ; b) - Π(s; b))/(υ-s+eps)) * exp(-((Π(υ; b) - Π(s; b))**2)/(2*(υ-s+eps))),
            f(υ; b) = 2*pi*log((1-2*υ)/(1-2*Υ)) + 2*(Π(υ; b) + θ)*log(1-2*Υ).

        Here, θ = self.mu and Υ = 1 - exp(-2*T_val) with T_val chosen so that Υ < 0.5.

        The free-boundary in original units is then:
            stop_loss = self.mu - b*/scale,  take_profit = self.mu + b*/scale,
        where scale = sqrt(self.kappa)/self.sigma.
        """
        theta = self.mu
        if self.kappa > 0 and self.sigma > 0:
            scale = np.sqrt(self.kappa) / self.sigma
        else:
            logger.warning(
                f"Invalid OU parameters (kappa = {self.kappa}, sigma = {self.sigma}); setting scale=1."
            )
            scale = 1.0

        eps = 1e-6
        T_val = 0.2
        Upsilon = 1 - np.exp(-2 * T_val)

        def Pi(upsilon, b):
            return np.sqrt(max(0, 1 - 2 * upsilon)) * (b - theta)

        def K(upsilon, s, b):
            denom = upsilon - s + eps
            return (
                1.0
                / np.sqrt(2 * np.pi)
                * ((Pi(upsilon, b) - Pi(s, b)) / denom)
                * np.exp(-((Pi(upsilon, b) - Pi(s, b)) ** 2) / (2 * denom))
            )

        def f_func(upsilon, b):
            return 2 * np.pi * np.log((1 - 2 * upsilon) / (1 - 2 * Upsilon)) + 2 * (
                Pi(upsilon, b) + theta
            ) * np.log(1 - 2 * Upsilon)

        N = 200
        u_uniform = np.linspace(0, 1, N)
        c = 5.0
        upsilon_grid = Upsilon * (np.exp(u_uniform * c) - 1) / (np.exp(c) - 1)
        delta = np.diff(upsilon_grid)

        def solve_E(b):
            E = np.zeros(N)
            for i in range(N):
                upsilon_i = upsilon_grid[i]
                if i == 0:
                    integral = 0.0
                else:
                    s_vals = upsilon_grid[: i + 1]
                    E_vals = E[: i + 1]
                    K_vals = np.array([K(upsilon_i, s, b) for s in s_vals])
                    integral = np.trapz(K_vals * E_vals, s_vals)
                E[i] = f_func(upsilon_i, b) + integral
            return E

        def free_boundary_error(b):
            E = solve_E(b)
            dE = (E[-1] - E[-2]) / (upsilon_grid[-1] - upsilon_grid[-2] + eps)
            return np.abs(dE)

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
            logger.info(f"Optimized free-boundary in natural units: b* = {b_star}")

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
        if not self.mean_reverting:
            # logger.info("Asset is not mean reverting. Skipping signal generation.")
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

        # Create a Series using the trades' index
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
        if not self.mean_reverting:
            # logger.info("Asset is not mean reverting. Skipping strategy.")
            return None, {"Message": "Not mean reverting"}

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
