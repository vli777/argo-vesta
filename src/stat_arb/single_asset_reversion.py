from typing import Optional, Union
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go

from utils import logger
from utils.performance_metrics import sharpe_ratio


class OUHeatPotential:
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

    def calculate_optimal_bounds(self):
        """
        Optimize stop-loss and take-profit levels to maximize composite score.
        Returns:
            tuple: (optimal_stop_loss, optimal_take_profit)
        """

        def objective(bounds):
            stop_loss, take_profit = bounds
            signals = self.generate_trading_signals(stop_loss, take_profit)
            _, metrics = self.simulate_strategy(signals)
            return -self.composite_score(metrics)

        bounds = [(-2 * self.sigma, 0), (0, 2 * self.sigma)]
        opt_result = minimize(objective, x0=[-1, 1], bounds=bounds)
        return opt_result.x

    def generate_trading_signals(
        self,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Generate trading signals using a stateful loop.
        A BUY signal is generated only when entering a new position.
        A SELL signal is generated only when exiting a position.
        Otherwise, "NO_SIGNAL" is used to avoid signal propagation.
        The very first day is always neutral (NO_SIGNAL).

        Args:
            stop_loss (Optional[float]): Lower deviation threshold to enter a long position.
            take_profit (Optional[float]): Upper deviation threshold to exit the position.

        Returns:
            pd.DataFrame: Signals with "Position", "Entry Price", and "Exit Price".
        """
        if stop_loss is None or take_profit is None:
            stop_loss, take_profit = self.calculate_optimal_bounds()

        log_prices = np.log(self.prices)
        deviations = log_prices - self.mu
        signals = pd.DataFrame(
            index=self.prices.index, columns=["Position", "Entry Price", "Exit Price"]
        )
        signals[:] = np.nan

        in_position = False
        entry_price = None

        for idx, date in enumerate(self.prices.index):
            price = self.prices.iloc[idx]
            dev = deviations.iloc[idx]
            # Always mark the first day as neutral
            if idx == 0:
                signals.loc[date, "Position"] = "NO_SIGNAL"
                continue

            if not in_position:
                if dev < stop_loss:
                    in_position = True
                    entry_price = price
                    signals.loc[date, "Position"] = "BUY"
                    signals.loc[date, "Entry Price"] = price
                else:
                    signals.loc[date, "Position"] = "NO_SIGNAL"
            else:
                # While holding a position, remain neutral unless exit is triggered
                signals.loc[date, "Position"] = "NO_SIGNAL"
                if dev > take_profit:
                    signals.loc[date, "Position"] = "SELL"
                    signals.loc[date, "Entry Price"] = entry_price
                    signals.loc[date, "Exit Price"] = price
                    in_position = False
                    entry_price = None

        return signals

    def simulate_strategy(self, signals):
        """
        Simulate strategy performance.

        Args:
            signals (pd.DataFrame): Trading signals.

        Returns:
            tuple: (trade returns series, metrics dictionary)
        """
        trades = signals.dropna(subset=["Entry Price", "Exit Price"])
        if trades.empty:
            # Return a zero-filled series over the same index as prices
            zero_returns = pd.Series(0, index=self.prices.index)
            metrics = {
                "Total Trades": 0,
                "Sharpe Ratio": 0,
                "Win Rate": 0,
                "Optimized Kelly Fraction": 0,
                "Risk Parity Allocation": {},
            }
            return zero_returns, metrics

        # Force conversion to float in case the types are mixed
        entry_prices = np.array(trades["Entry Price"].values, dtype=float)
        exit_prices = np.array(trades["Exit Price"].values, dtype=float)
        returns = np.log(exit_prices / entry_prices)
        
        # Create a Series using the trades' index
        returns_series = pd.Series(returns, index=trades.index)

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
        stop_loss, take_profit = self.calculate_optimal_bounds()
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
            stop_loss, take_profit = self.calculate_optimal_bounds()
            signals = self.generate_trading_signals(stop_loss, take_profit)
            _, metrics = self.simulate_strategy(signals)
            return self.composite_score(metrics)

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)

        best_params = study.best_params
        logger.info(f"Optimized Kelly & Risk Parity: {best_params}")
        return best_params

   