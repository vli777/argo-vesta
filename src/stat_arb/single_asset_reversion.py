from typing import Optional, Union
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go

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
        # print(
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
        Vectorized trading signal generation using NumPy and Pandas indexing.

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

        # Entry (BUY signal) when deviation < stop_loss
        buy_mask = (deviations < stop_loss) & (signals["Position"].isna())
        signals.loc[buy_mask, "Position"] = "BUY"
        signals.loc[buy_mask, "Entry Price"] = self.prices[buy_mask]

        # Exit (SELL signal) when deviation > take_profit
        sell_mask = (deviations > take_profit) & (signals["Position"].shift(1) == "BUY")
        signals.loc[sell_mask, "Position"] = "SELL"
        signals.loc[sell_mask, "Exit Price"] = self.prices[sell_mask]

        return signals

    def simulate_strategy(self, signals):
        """
        Simulate strategy performance.

        Args:
            signals (pd.DataFrame): Trading signals.

        Returns:
            tuple: (trade returns array, metrics dictionary)
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

        returns = np.log(trades["Exit Price"].values / trades["Entry Price"].values)
        returns_series = pd.Series(returns).reset_index(drop=True)

        win_rate = (returns > 0).mean()
        sharpe_r = sharpe_ratio(
            returns_series, entries_per_year=252, risk_free_rate=0.0
        )
        metrics = {
            "Total Trades": len(trades),
            "Sharpe Ratio": sharpe_r,
            "Win Rate": win_rate,
        }
        return returns, metrics

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
        print(f"Optimal stop_loss: {stop_loss:.4f}, take_profit: {take_profit:.4f}")
        signals = self.generate_trading_signals(stop_loss, take_profit)
        _, metrics = self.simulate_strategy(signals)
        print("Strategy Metrics:", metrics)
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
        print(f"Optimized Kelly & Risk Parity: {best_params}")
        return best_params

    def plot_signals(self, title: str = "OU Process Trading Signals"):
        """
        Create an interactive Plotly chart that overlays buy/sell signals on the price series.

        Args:
            title (str): Title of the plot.
        """
        signals, _ = self.run_strategy()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.prices.index, y=self.prices.values, mode="lines", name="Price"
            )
        )
        buy_signals = signals[signals["Position"] == "BUY"]
        sell_signals = signals[signals["Position"] == "SELL"]
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=self.prices.loc[buy_signals.index],
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", color="green", size=10),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=self.prices.loc[sell_signals.index],
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", color="red", size=10),
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_white",
        )
        fig.show()
