from typing import Optional
import numpy as np
import pandas as pd
import optuna
from scipy.optimize import minimize
import torch

from stat_arb.graph_autoencoder import construct_graph, train_gae
from utils.performance_metrics import sharpe_ratio


class MultiAssetReversion:
    def __init__(
        self,
        prices_df: pd.DataFrame,
        hidden_channels=32,
        num_epochs=200,
        learning_rate=0.01,
        p_value_threshold=0.05,
    ):
        """
        Multi-asset mean reversion strategy using a Graph Autoencoder (GAE) for cointegration.
        Handles tickers with different history lengths by selecting the maximum overlapping period.
        The GAE-derived latent embeddings are used to compute hedge ratios and the stationary spread.

        Args:
            prices_df (pd.DataFrame): Price data DataFrame (regular prices) with each column as a ticker.
            det_order (int): Deterministic trend order (retained for compatibility).
            k_ar_diff (int): Number of lag differences (retained for compatibility).
            hidden_channels (int): Hidden dimension size for the GAE.
            num_epochs (int): Number of training epochs for the GAE.
            learning_rate (float): Learning rate for GAE training.
            p_value_threshold (float): P-value threshold for establishing cointegration edges.
        """
        # Ensure prices are strictly positive.
        if (prices_df <= 0).any().any():
            raise ValueError("Price data must be strictly positive.")

        # Determine maximum overlapping period.
        start_date = max(prices_df.apply(lambda col: col.first_valid_index()))
        end_date = min(prices_df.apply(lambda col: col.last_valid_index()))
        if start_date is None or end_date is None or start_date > end_date:
            raise ValueError("No overlapping period found among tickers.")

        # Restrict to the overlapping period and drop rows with missing data.
        prices_df = prices_df.loc[start_date:end_date].dropna(axis=0, how="any")

        # Convert to log prices.
        self.prices_df = np.log(prices_df)
        self.returns_df = self.prices_df.diff().dropna()

        # --- GAE-Based Cointegration Analysis ---
        # Construct the asset graph using log-prices and returns.
        data = construct_graph(
            self.prices_df, self.returns_df, p_value_threshold=p_value_threshold
        )
        # Train the Graph Autoencoder.
        self.gae_model = train_gae(
            data,
            hidden_channels=hidden_channels,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
        )
        self.gae_model.eval()
        with torch.no_grad():
            # Obtain latent embeddings for each asset.
            latent = self.gae_model.encode(data.x, data.edge_index).cpu().numpy()
        self.latent_embeddings = latent
        # Derive hedge ratios from the first latent dimension.
        raw_hedge_ratios = latent[:, 0]
        norm_factor = np.sum(np.abs(raw_hedge_ratios))
        if norm_factor == 0:
            norm_factor = 1e-6
        self.hedge_ratios = raw_hedge_ratios / norm_factor
        # Compute the basket spread as the weighted sum of log-prices.
        self.spread_series = self.prices_df.dot(self.hedge_ratios)

        # --- Allocation Computations ---
        self.kelly_fractions = self.compute_dynamic_kelly()
        self.risk_parity_weights = self.compute_risk_parity_weights()
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

    def generate_trading_signals(self, stop_loss=None, take_profit=None):
        """
        Generates basket-level trading signals based on the spread using a stateful approach.
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
