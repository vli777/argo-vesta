from typing import Optional
import numpy as np
import pandas as pd
import optuna
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import torch

from statsmodels.tsa.vector_ar.vecm import coint_johansen
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
    ):
        """
        Multi-asset mean reversion strategy using a hybrid GNN + Johansen approach.
        This version handles tickers with different history lengths by selecting the maximum overlapping period.

        Args:
            prices_df (pd.DataFrame): Price data DataFrame (regular prices) with each column as a ticker.
            asset_cluster_map (dict[str, int], optional): Mapping from asset tickers to cluster labels.
                Assets labeled as -1 are considered noise and are excluded.
            hidden_channels (int): Hidden dimension size for the GNN.
            num_epochs (int): Number of epochs for GNN training.
            learning_rate (float): Learning rate for GNN training.
            p_value_threshold (float): P-value threshold for constructing the asset graph.
            n_clusters (int): Number of clusters to segment assets.
        """
        # Validate and preprocess price data
        if (prices_df <= 0).any().any():
            raise ValueError("Price data must be strictly positive.")
        start_date = max(prices_df.apply(lambda col: col.first_valid_index()))
        end_date = min(prices_df.apply(lambda col: col.last_valid_index()))
        if start_date is None or end_date is None or start_date > end_date:
            raise ValueError("No overlapping period found among tickers.")
        prices_df = prices_df.loc[start_date:end_date].dropna(axis=0, how="any")

        # Convert to log prices and compute returns
        self.prices_df = np.log(prices_df)
        self.returns_df = self.prices_df.diff().dropna()

        # --- Cointegration Basket Construction ---
        if asset_cluster_map is not None:
            # Filter the asset cluster map to only include relevant assets
            filtered_asset_cluster_map = self.filter_asset_cluster_map(
                asset_cluster_map
            )
            # Use externally provided cluster mapping.
            self.hedge_ratios, self.spread_series = (
                self.construct_cointegrated_baskets_from_map(filtered_asset_cluster_map)
            )
        # --- Hybrid GNN + Johansen Step ---
        self.hedge_ratios, self.spread_series = self.construct_cointegrated_baskets(
            hidden_channels, num_epochs, learning_rate, p_value_threshold, n_clusters
        )

        # Allocation computations (Kelly, risk parity, etc.)
        self.kelly_fractions = self.compute_dynamic_kelly()
        self.risk_parity_weights = self.compute_risk_parity_weights()
        self.optimal_params = self.optimize_kelly_risk_parity()

    def filter_asset_cluster_map(
        self, asset_cluster_map: dict[str, int]
    ) -> dict[str, int]:
        """
        Filters the asset cluster map to include only assets present in prices_df.

        Args:
            asset_cluster_map (dict[str, int]): The full asset cluster map.

        Returns:
            dict[str, int]: A filtered map with only relevant assets.
        """
        valid_assets = set(self.prices_df.columns)
        filtered_map = {
            asset: cluster_id
            for asset, cluster_id in asset_cluster_map.items()
            if asset in valid_assets and cluster_id != -1
        }

        # Debug: Check filtered asset distribution
        logger.info(
            f"Filtered {len(filtered_map)} assets from {len(asset_cluster_map)} in the original map."
        )

        return filtered_map

    def construct_cointegrated_baskets_from_map(
        self, asset_cluster_map: dict[str, int]
    ):
        """
        Constructs cointegrated baskets using an externally provided asset_cluster_map.
        Assets labeled as -1 (noise) are ignored.

        Args:
            asset_cluster_map (dict[str, int]): Mapping from asset tickers to cluster labels.

        Returns:
            tuple: (hedge_ratios dictionary, aggregated basket spread series)
        """
        hedge_ratios = {}
        basket_spread = pd.Series(0, index=self.prices_df.index)

        # Group assets by their cluster labels (ignore noise labeled as -1).
        clusters = {}
        for asset, cluster_label in asset_cluster_map.items():
            if cluster_label == -1:
                continue
            clusters.setdefault(cluster_label, []).append(asset)

        # For each cluster, run Johansen test to obtain cointegration vectors.
        for cluster_label, assets in clusters.items():
            # Filter out assets that are not present in prices_df
            valid_assets = [
                asset for asset in assets if asset in self.prices_df.columns
            ]

            if len(valid_assets) < 2:
                logger.debug(f"Skipping cluster {cluster_label}")
                continue

            # Use valid_assets instead of assets to avoid KeyError
            cluster_prices = self.prices_df[valid_assets]

            if cluster_prices.shape[1] < 2:
                # Single asset: assign full weight.
                hedge_ratios[valid_assets[0]] = 1.0
                basket_spread += cluster_prices.iloc[:, 0]
            else:
                try:
                    result = coint_johansen(cluster_prices, det_order=0, k_ar_diff=1)
                    cointegration_vector = result.evec[:, 0]
                    cointegration_vector /= np.sum(
                        np.abs(cointegration_vector)
                    )  # Normalize
                    for asset, weight in zip(valid_assets, cointegration_vector):
                        hedge_ratios[asset] = weight
                    basket_spread += cluster_prices.dot(cointegration_vector)
                except Exception as e:
                    logger.info(f"Cluster {cluster_label} Johansen test error: {e}")

        return hedge_ratios, basket_spread

    def construct_cointegrated_baskets(
        self, hidden_channels, num_epochs, learning_rate, p_value_threshold, n_clusters
    ):
        """
        Constructs cointegrated baskets using a GNN-based clustering followed by Johansen testing.
        Returns a dictionary of hedge ratios and an aggregated basket spread series.
        """
        # Construct asset graph and train GNN
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

        # Cluster latent embeddings to segment assets
        # Use the number of latent samples (number of assets) to adjust n_clusters if needed.
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

        # Run Johansen test within each cluster to get cointegration vectors
        for cluster_id in np.unique(clusters):
            asset_indices = np.where(clusters == cluster_id)[0]
            cluster_assets = self.prices_df.columns[asset_indices]
            cluster_prices = self.prices_df[cluster_assets]

            if cluster_prices.shape[1] < 2:
                # Single asset: assign full weight
                hedge_ratios[cluster_assets[0]] = 1.0
                basket_spread += cluster_prices.iloc[:, 0]
            else:
                try:
                    result = coint_johansen(cluster_prices, det_order=0, k_ar_diff=1)
                    # For simplicity, use the first eigenvector if it passes the 5% critical value test
                    cointegration_vector = result.evec[:, 0]
                    cointegration_vector /= np.sum(
                        np.abs(cointegration_vector)
                    )  # Normalize
                    for asset, weight in zip(cluster_assets, cointegration_vector):
                        hedge_ratios[asset] = weight
                    # Compute basket spread for this cluster
                    basket_spread += cluster_prices.dot(cointegration_vector)
                except Exception as e:
                    logger.info(f"Cluster {cluster_id} Johansen test error: {e}")

        return hedge_ratios, basket_spread

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
                return -np.inf
            final_allocations /= total
            portfolio_returns = (self.returns_df * final_allocations).sum(axis=1)
            if portfolio_returns.std() == 0 or np.isnan(portfolio_returns.std()):
                return -np.inf
            s = sharpe_ratio(portfolio_returns)
            return s if not np.isnan(s) else -np.inf

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=50)
        return study.best_params

    def calculate_optimal_bounds(self):
        def objective(bounds):
            stop_loss, take_profit = bounds
            signals = self.generate_trading_signals(stop_loss, take_profit)
            _, metrics = self.simulate_strategy(signals)
            if metrics["Total Trades"] == 0:
                return np.inf
            s = metrics["Sharpe Ratio"]
            return -s if not np.isnan(s) else np.inf

        std_spread = self.spread_series.std()
        if std_spread == 0 or np.isnan(std_spread):
            std_spread = 1e-6
        bounds = [(-2 * std_spread, 0), (0, 2 * std_spread)]
        result = minimize(objective, x0=[-0.5, 0.5], bounds=bounds)
        return result.x

    def generate_trading_signals(self, stop_loss=None, take_profit=None):
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

        position = 0
        entry_price = np.nan

        for t in signals.index:
            dev = deviations.loc[t]
            current_price = self.spread_series.loc[t]

            if position == 0:
                # No current position, so check for entry signals.
                if dev < stop_loss:
                    position = 1
                    entry_price = current_price
                    signals.loc[t, "Position"] = 1
                    signals.loc[t, "Entry Price"] = entry_price
                    signals.loc[t, "Exit Price"] = (
                        current_price  # set initial exit price
                    )
                elif dev > take_profit:
                    position = -1
                    entry_price = current_price
                    signals.loc[t, "Position"] = -1
                    signals.loc[t, "Entry Price"] = entry_price
                    signals.loc[t, "Exit Price"] = (
                        current_price  # set initial exit price
                    )
                else:
                    signals.loc[t, "Position"] = 0
                    # Leave Entry and Exit prices as NaN when no position.
            elif position == 1:
                # Long position is open.
                if dev >= 0:
                    # Exit condition met for a long.
                    signals.loc[t, "Position"] = 0
                    signals.loc[t, "Exit Price"] = current_price
                    position = 0
                    entry_price = np.nan
                else:
                    signals.loc[t, "Position"] = 1
                    signals.loc[t, "Entry Price"] = (
                        entry_price  # keep the original entry price
                    )
                    signals.loc[t, "Exit Price"] = (
                        current_price  # update exit price continuously
                    )
            elif position == -1:
                # Short position is open.
                if dev <= 0:
                    # Exit condition met for a short.
                    signals.loc[t, "Position"] = 0
                    signals.loc[t, "Exit Price"] = current_price
                    position = 0
                    entry_price = np.nan
                else:
                    signals.loc[t, "Position"] = -1
                    signals.loc[t, "Entry Price"] = (
                        entry_price  # keep the original entry price
                    )
                    signals.loc[t, "Exit Price"] = (
                        current_price  # update exit price continuously
                    )

        signals["Entry Price"] = signals["Entry Price"].ffill()
        signals["Exit Price"] = signals["Exit Price"].ffill()

        return signals

    def simulate_strategy(self, signals):
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
