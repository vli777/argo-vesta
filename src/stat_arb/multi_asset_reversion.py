from typing import Optional
import numpy as np
import pandas as pd
import optuna
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import torch
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller

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
        now with OU dynamics for signal generation.

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

        # Validate and preprocess price data
        if (prices_df <= 0).any().any():
            raise ValueError("Price data must be strictly positive.")
        start_date = max(prices_df.apply(lambda col: col.first_valid_index()))
        end_date = min(prices_df.apply(lambda col: col.last_valid_index()))
        if start_date is None or end_date is None or start_date > end_date:
            raise ValueError("No overlapping period found among tickers.")
        prices_df = prices_df.loc[start_date:end_date].dropna(axis=0, how="any")

        # Convert to log prices and compute returns (log returns) as differences.
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

        # Center the spread so that its average is near zero.
        self.spread_series = self.spread_series - self.spread_series.mean()

        # Estimate OU parameters on the basket spread.
        self.ou_kappa, self.ou_mu, self.ou_sigma = self.estimate_ou_parameters()

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
                    cointegration_vector /= np.sum(
                        np.abs(cointegration_vector)
                    )  # Normalize
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
                    cointegration_vector /= np.sum(
                        np.abs(cointegration_vector)
                    )  # Normalize
                    for asset, weight in zip(cluster_assets, cointegration_vector):
                        hedge_ratios[asset] = weight
                    basket_spread += cluster_prices.dot(cointegration_vector)
                except Exception as e:
                    logger.info(f"Cluster {cluster_id} Johansen test error: {e}")
        basket_spread = basket_spread - basket_spread.mean()
        return hedge_ratios, basket_spread

    def estimate_ou_parameters(self):
        """
        Estimate OU parameters on the basket spread (assumed mean-reverting).
        We perform a simple OLS regression:
            Δx_t = α + β * x_(t-1) + ε_t
        and compute:
            kappa = -β,   μ = α / kappa,   σ = std(ε_t) scaled by a factor.
        """
        x = self.spread_series.dropna()
        delta_x = x.diff().dropna()
        x_lag = x.shift(1).dropna()
        # Align delta_x with x_lag
        delta_x = delta_x.loc[x_lag.index]
        beta, alpha = np.polyfit(x_lag, delta_x, 1)
        kappa = -beta  # assume dt = 1
        mu = alpha / kappa if kappa != 0 else 0
        residuals = delta_x - (alpha + beta * x_lag)
        # Scale residual std to account for OU process dynamics
        sigma = np.std(residuals) * np.sqrt(2 * kappa / (1 - np.exp(-2 * kappa)))
        return kappa, mu, sigma

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
        self, stop_loss: Optional[float] = None, take_profit: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate trading signals based on OU dynamics.
        The deviation is computed as:
            deviation = spread_series - ou_mu
        We enter a long position if the deviation is below stop_loss and exit (or reverse)
        when the deviation moves above take_profit.

        Optional ADF and trend filters are applied.
        """
        if stop_loss is None or take_profit is None:
            stop_loss, take_profit = self.calculate_optimal_bounds()

        # Compute OU deviation (using the estimated long-run mean)
        ou_deviation = self.spread_series - self.ou_mu

        # Apply stationarity filter if enabled
        if self.use_adf_filter:
            is_stationary = self.compute_rolling_adf(
                self.spread_series, self.adf_window, self.adf_pvalue
            )
        else:
            is_stationary = pd.Series(True, index=self.spread_series.index)

        # Trend filter based on moving averages of the OU deviation
        if self.use_trend_filter:
            short_ma = ou_deviation.rolling(
                self.short_ma_window, min_periods=self.short_ma_window
            ).mean()
            long_ma = ou_deviation.rolling(
                self.long_ma_window, min_periods=self.long_ma_window
            ).mean()
        else:
            short_ma = pd.Series(0.0, index=ou_deviation.index)
            long_ma = pd.Series(0.0, index=ou_deviation.index)

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
        # Use a small epsilon as an exit buffer (could be a fraction of ou_sigma)
        epsilon = 0.1 * self.ou_sigma

        for t in signals.index:
            # Skip if insufficient data or non-stationary period.
            if np.isnan(ou_deviation.loc[t]) or not is_stationary.loc[t]:
                signals.loc[t, "Position"] = 0
                continue

            dev = ou_deviation.loc[t]

            # Apply trend filter: if short_ma > long_ma, avoid short; if short_ma < long_ma, avoid long.
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
                    signals.loc[t, "Position"] = 1  # Enter long
                    signals.loc[t, "Entry Price"] = dev
                    signals.loc[t, "Exit Price"] = dev
                elif (dev > take_profit) and can_short:
                    position = -1
                    entry_value = dev
                    signals.loc[t, "Position"] = -1  # Enter short
                    signals.loc[t, "Entry Price"] = dev
                    signals.loc[t, "Exit Price"] = dev
                else:
                    signals.loc[t, "Position"] = 0
            elif position == 1:
                # For a long position, exit if deviation rises above -epsilon (near the mean).
                if dev >= -epsilon:
                    signals.loc[t, "Position"] = 0
                    signals.loc[t, "Exit Price"] = dev
                    position = 0
                    entry_value = np.nan
                else:
                    signals.loc[t, "Position"] = 1
                    signals.loc[t, "Entry Price"] = entry_value
                    signals.loc[t, "Exit Price"] = dev
            elif position == -1:
                # For a short position, exit if deviation falls below epsilon.
                if dev <= epsilon:
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
        We compute returns as the change in the basket spread multiplied by the lagged signal.
        """
        # Use the difference in spread_series as the "return"
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
