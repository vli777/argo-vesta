from typing import Optional
import numpy as np
import pandas as pd
import optuna
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import torch

from stat_arb.cointegration import CointegrationAnalyzer
from utils.performance_metrics import sharpe_ratio

class MultiAssetReversion:
    def __init__(self, prices_df: pd.DataFrame, det_order=0, k_ar_diff=1):
        """
        Multi-asset mean reversion strategy based on cointegration.
        This version handles tickers with different history lengths by selecting
        the maximum overlapping period.

        Args:
            prices_df (pd.DataFrame): Price data DataFrame (regular prices) where
                                      each column is a ticker.
            det_order (int): Deterministic trend order for Johansen test.
            k_ar_diff (int): Number of lag differences.
        """
        # Ensure prices are strictly positive.
        if (prices_df <= 0).any().any():
            raise ValueError("Price data must be strictly positive.")

        # Determine maximum overlapping period by finding the latest start and earliest end.
        start_date = max(prices_df.apply(lambda col: col.first_valid_index()))
        end_date = min(prices_df.apply(lambda col: col.last_valid_index()))
        if start_date is None or end_date is None or start_date > end_date:
            raise ValueError("No overlapping period found among tickers.")

        # Restrict to the overlapping period.
        prices_df = prices_df.loc[start_date:end_date]

        # Drop any remaining rows with missing data.
        prices_df = prices_df.dropna(axis=0, how="any")

        # Convert to log prices.
        self.prices_df = np.log(prices_df)
        self.returns_df = self.prices_df.diff().dropna()

        # Initialize cointegration analyzer.
        self.cointegration_analyzer = CointegrationAnalyzer(
            self.prices_df, det_order, k_ar_diff
        )
        # Reindex the spread to match our prices and fill missing values.
        self.spread_series = (
            self.cointegration_analyzer.spread_series.reindex(self.prices_df.index)
            .ffill()
            .bfill()
        )
        self.hedge_ratios = self.cointegration_analyzer.get_hedge_ratios()

        # Compute allocations.
        self.kelly_fractions = self.compute_dynamic_kelly()
        self.risk_parity_weights = self.compute_risk_parity_weights()

        # Optimize the combined allocations.
        self.optimal_params = self.optimize_kelly_risk_parity()

    def compute_dynamic_kelly(self, risk_free_rate=0.0):
        """
        Compute dynamic Kelly fractions for each asset.
        """
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
        """
        Computes Risk Parity weights based on inverse volatility.
        """
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
        """
        Jointly optimizes Kelly sizing and Risk Parity scaling to maximize a composite performance metric.
        """

        def objective(trial):
            kelly_scaling = trial.suggest_float("kelly_scaling", 0.1, 1.0)
            risk_parity_scaling = trial.suggest_float("risk_parity_scaling", 0.1, 1.0)
            kelly_allocations = self.kelly_fractions * kelly_scaling
            risk_parity_allocations = self.risk_parity_weights * risk_parity_scaling
            final_allocations = kelly_allocations + risk_parity_allocations
            total = final_allocations.sum()
            if total == 0 or np.isnan(total):
                return -np.inf
            final_allocations /= total

            portfolio_returns = (self.returns_df * final_allocations).sum(axis=1)
            if portfolio_returns.std() == 0 or np.isnan(portfolio_returns.std()):
                return -np.inf
            s = sharpe_ratio(portfolio_returns)
            if np.isnan(s):
                return -np.inf
            return s

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=50)

        return study.best_params

    def calculate_optimal_bounds(self):
        """
        Finds optimal stop-loss and take-profit levels based on maximizing (negative) the Sharpe ratio.
        A penalty (np.inf) is returned if no trades are generated.
        """

        def objective(bounds):
            stop_loss, take_profit = bounds
            signals = self.generate_trading_signals(stop_loss, take_profit)
            _, metrics = self.simulate_strategy(signals)
            if metrics["Total Trades"] == 0:
                return np.inf
            s = metrics["Sharpe Ratio"]
            if np.isnan(s):
                return np.inf
            return -s

        std_spread = self.spread_series.std()
        if std_spread == 0 or np.isnan(std_spread):
            std_spread = 1e-6
        bounds = [(-2 * std_spread, 0), (0, 2 * std_spread)]
        result = minimize(objective, x0=[-0.5, 0.5], bounds=bounds)
        return result.x

    def generate_trading_signals(
        self, stop_loss: Optional[float] = None, take_profit: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generates basket-level buy/sell signals based on the spread using a stateful approach.
        Once a position is entered, it is held until the exit condition is met.

        Returns:
            pd.DataFrame: Signals with columns "Position", "Ticker", "Entry Price", "Exit Price".
        """
        if stop_loss is None or take_profit is None:
            stop_loss, take_profit = self.calculate_optimal_bounds()

        deviations = self.spread_series - self.spread_series.mean()
        signals = pd.DataFrame(
            index=self.spread_series.index,
            columns=["Position", "Ticker", "Entry Price", "Exit Price"],
        )
        signals["Ticker"] = ", ".join(self.prices_df.columns)
        signals["Position"] = 0
        signals["Entry Price"] = np.nan
        signals["Exit Price"] = np.nan

        position = 0  # 0: no position, 1: long, -1: short
        for t in signals.index:
            dev = deviations.loc[t]
            current_price = self.spread_series.loc[t]
            if position == 0:
                if dev < stop_loss:
                    position = 1
                    signals.loc[t, "Position"] = 1
                    signals.loc[t, "Entry Price"] = current_price
                elif dev > take_profit:
                    position = -1
                    signals.loc[t, "Position"] = -1
                    signals.loc[t, "Entry Price"] = current_price
                else:
                    signals.loc[t, "Position"] = 0
            elif position == 1:
                if dev >= 0:
                    signals.loc[t, "Position"] = 0
                    signals.loc[t, "Exit Price"] = current_price
                    position = 0
                else:
                    signals.loc[t, "Position"] = 1
            elif position == -1:
                if dev <= 0:
                    signals.loc[t, "Position"] = 0
                    signals.loc[t, "Exit Price"] = current_price
                    position = 0
                else:
                    signals.loc[t, "Position"] = -1

        signals["Position"] = signals["Position"].ffill().fillna(0)
        return signals

    def simulate_strategy(self, signals):
        """
        Backtests the basket strategy using the generated signals.
        Computes returns by applying the previous period's Position
        to the percentage change in the spread.
        """
        returns = signals["Position"].shift(1) * self.spread_series.pct_change()
        returns = returns.dropna()

        s = sharpe_ratio(returns)
        metrics = {
            "Total Trades": (signals["Position"] != 0).sum(),
            "Sharpe Ratio": s,
            "Win Rate": (returns > 0).mean(),
            "Avg Return": returns.mean(),
        }
        return returns, metrics

    def optimize_and_trade(self):
        """
        Full pipeline:
          1. Optimize entry/exit bounds.
          2. Generate trading signals.
          3. Simulate strategy and compute metrics.
          4. Return the hedge ratios along with signals and metrics.
        """
        stop_loss, take_profit = self.calculate_optimal_bounds()
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
