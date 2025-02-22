import sys
import numpy as np
import optuna
import pandas as pd

from utils.performance_metrics import sharpe_ratio


class PortfolioAllocator:
    def __init__(self, risk_target: float = 0.15, leverage_cap: float = 1.0):
        """
        Initializes the PortfolioAllocator with risk management parameters.

        Args:
            risk_target (float): Target portfolio volatility.
            leverage_cap (float): Maximum allowable leverage.
        """
        self.risk_target = risk_target
        self.leverage_cap = leverage_cap

    def compute_allocations(
        self,
        individual_returns: dict,
        multi_asset_returns: pd.Series,
        hedge_ratios: dict,
        individual_signals: dict,
    ) -> pd.Series:
        """
        Computes the final portfolio allocation by combining individual strategy signals
        with basket signals. The basket allocation is constructed by distributing exposure
        via hedge ratios and refining it with a basket signal derived from multi-asset returns.
        The individual signals are used to adjust the baseline optimized weights continuously:
        the multiplier is determined by how far the current deviation is from the threshold relative to σ.

        Args:
            individual_returns (dict): Dictionary of per-ticker strategy returns.
            multi_asset_returns (pd.Series): Multi-asset reversion returns (used as the basket signal).
            hedge_ratios (dict): Hedge ratios from the basket signal.
            individual_signals (dict): For each ticker, a dict with keys: "signal", "current_deviation", "stop_loss", "take_profit", "sigma".

        Returns:
            pd.Series: Final portfolio weights.
        """
        # --- Compute individual (baseline) allocation ---
        df_indiv = pd.DataFrame(individual_returns).fillna(0)
        risk_parity_indiv = self.compute_risk_parity(df_indiv)
        kelly_indiv = self.compute_kelly_sizing(df_indiv)
        optimal_indiv = self.optimize_kelly_risk_parity(
            kelly_indiv, risk_parity_indiv, df_indiv
        )

        # --- Compute basket allocation using hedge ratios ---
        w_basket = pd.Series(hedge_ratios)
        if w_basket.sum() != 0:
            w_basket = w_basket / w_basket.abs().sum()
        else:
            w_basket = pd.Series(1, index=optimal_indiv.index)
            w_basket = w_basket / w_basket.sum()

        # --- Incorporate basket signal from multi-asset returns ---
        basket_signal = multi_asset_returns.iloc[-1]
        basket_scale = np.tanh(basket_signal)
        refined_basket = basket_scale * w_basket

        # --- Compute continuous adjustment multipliers based on individual signals ---
        def compute_multiplier(sig_info):
            # For BUY: if current_deviation < stop_loss, multiplier = 1 + (stop_loss - d)/sigma.
            # For SELL: if current_deviation > take_profit, multiplier = 1 - (d - take_profit)/sigma.
            # For NO_SIGNAL, multiplier = 1.
            d = sig_info["current_deviation"]
            sigma = sig_info["sigma"]
            if sig_info["signal"] == "BUY":
                multiplier = 1 + (sig_info["stop_loss"] - d) / sigma
            elif sig_info["signal"] == "SELL":
                multiplier = 1 - (d - sig_info["take_profit"]) / sigma
            else:
                multiplier = 1.0
            # Optionally clip multiplier to a range:
            multiplier = max(0.5, min(1.5, multiplier))
            return multiplier

        signal_factors = {
            ticker: compute_multiplier(info)
            for ticker, info in individual_signals.items()
        }
        signal_factor_series = pd.Series(signal_factors)

        # --- Adjust baseline allocations using these multipliers ---
        adjusted_indiv = optimal_indiv * signal_factor_series
        adjusted_basket = refined_basket * signal_factor_series

        # --- Optimize combination parameter gamma to blend the two allocations ---
        def objective(trial):
            gamma = trial.suggest_float("gamma", 0.0, 1.0, step=0.01)
            combined = gamma * adjusted_indiv + (1 - gamma) * adjusted_basket
            if combined.sum() == 0:
                return -np.inf
            combined = combined / combined.abs().sum()
            port_returns = (df_indiv * combined).sum(axis=1)
            if port_returns.std() == 0:
                return -np.inf
            sr = sharpe_ratio(port_returns)
            return sr if not np.isnan(sr) else -np.inf

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=50, n_jobs=-1)
        best_gamma = study.best_params.get("gamma", 0.5)

        final_weights = best_gamma * adjusted_indiv + (1 - best_gamma) * adjusted_basket
        if final_weights.sum() != 0:
            final_weights = final_weights / final_weights.abs().sum()
        else:
            final_weights = optimal_indiv  # fallback

        final_allocations = self.apply_adaptive_leverage(final_weights, df_indiv)
        return final_allocations

    def compute_risk_parity(self, returns_df: pd.DataFrame) -> pd.Series:
        """
        Computes risk parity weights based on historical volatilities.

        Args:
            returns_df (pd.DataFrame): Returns of all strategies.

        Returns:
            pd.Series: Risk parity weights.
        """
        vol = returns_df.std().replace(0, 1e-6)
        risk_parity_weights = 1 / vol
        return risk_parity_weights / risk_parity_weights.sum()

    def compute_kelly_sizing(self, returns_df: pd.DataFrame) -> pd.Series:
        """
        Computes Kelly-optimal bet sizing for each strategy.

        Args:
            returns_df (pd.DataFrame): Returns of all strategies.

        Returns:
            pd.Series: Kelly fractions.
        """
        mean_returns = returns_df.mean()
        variance = returns_df.var()
        # Replace zero variance with NaN to avoid division by zero
        variance = variance.replace(0, np.nan)
        kelly_fractions = mean_returns / variance
        # Replace any NaNs (which may occur if mean is also zero) with zero
        kelly_fractions = kelly_fractions.fillna(0)

        total = kelly_fractions.sum()
        if total == 0:
            # If all values are zero, return zeros to avoid division by zero
            return pd.Series(0, index=returns_df.columns)

        return kelly_fractions / total

    def optimize_kelly_risk_parity(
        self,
        kelly_weights: pd.Series,
        risk_parity_weights: pd.Series,
        returns_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Uses Optuna to jointly optimize Kelly scaling and risk parity allocation.
        Returns:
            pd.Series: Optimized final allocation weights.
        """

        def objective(trial):
            kelly_scaling = trial.suggest_float("kelly_scaling", 0.1, 1.0)
            risk_parity_scaling = trial.suggest_float("risk_parity_scaling", 0.1, 1.0)
            combined = (
                kelly_scaling * kelly_weights
                + risk_parity_scaling * risk_parity_weights
            )
            if combined.sum() == 0:
                return -np.inf
            combined /= combined.abs().sum()
            port_returns = (returns_df * combined).sum(axis=1)
            if port_returns.std() == 0:
                return -np.inf
            sr = sharpe_ratio(port_returns)
            if np.isnan(sr):
                return -np.inf
            return sr

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=50, n_jobs=-1)
        best_params = study.best_params
        final_weights = (
            best_params["kelly_scaling"] * kelly_weights
            + best_params["risk_parity_scaling"] * risk_parity_weights
        )
        scaling_total = (
            best_params["kelly_scaling"] + best_params["risk_parity_scaling"]
        )
        if scaling_total == 0:
            return risk_parity_weights
        final_weights /= scaling_total
        return final_weights

    def apply_adaptive_leverage(
        self, weights: pd.Series, returns_df: pd.DataFrame
    ) -> pd.Series:
        """
        Applies dynamic leverage based on market volatility.

        Returns:
            pd.Series: Final leverage-adjusted weights.
        """
        # Compute rolling volatility (ensure there are enough data points)
        realized_volatility = (
            returns_df.rolling(window=30, min_periods=5).std().mean(axis=1).iloc[-1]
        )

        # Prevent division by zero
        if realized_volatility == 0 or np.isnan(realized_volatility):
            realized_volatility = 1e-6  # Small nonzero value to avoid infinite leverage

        leverage = min(self.risk_target / realized_volatility, self.leverage_cap)

        return weights * leverage
