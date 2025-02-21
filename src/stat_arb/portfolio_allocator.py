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
    ) -> pd.Series:
        """
        Computes the final portfolio allocation using Kelly + Risk Parity.

        Args:
            individual_returns (dict): Dictionary of per-ticker strategy returns.
            multi_asset_returns (pd.Series): Multi-asset reversion strategy returns.
            hedge_ratios (dict): Hedge ratios used in the cointegrated strategy.

        Returns:
            pd.Series: Final portfolio weights.
        """
        # Convert dictionary to DataFrame
        returns_df = pd.DataFrame(individual_returns)
        returns_df["multi_asset"] = multi_asset_returns
        returns_df = returns_df.fillna(0)

        # Compute risk parity weights
        risk_parity_weights = self.compute_risk_parity(returns_df)
        # Compute Kelly scaling
        kelly_weights = self.compute_kelly_sizing(returns_df)
        # Optimize Kelly & Risk Parity jointly using Optuna (pass returns_df)
        optimal_weights = self.optimize_kelly_risk_parity(
            kelly_weights, risk_parity_weights, returns_df
        )

        # Apply adaptive leverage
        final_allocations = self.apply_adaptive_leverage(optimal_weights, returns_df)

        # Distribute multi_asset into individual tickers based on hedge ratios
        final_allocations = self.distribute_multi_asset_weight(
            final_allocations, hedge_ratios
        )

        return final_allocations

    def distribute_multi_asset_weight(
        self, final_allocations: pd.Series, hedge_ratios: dict
    ) -> pd.Series:
        """
        Distributes the 'multi_asset' weight into individual tickers based on hedge ratios.

        Args:
            final_allocations (pd.Series): Portfolio allocations, including 'multi_asset'.
            hedge_ratios (dict): Hedge ratios used in the cointegrated strategy.

        Returns:
            pd.Series: Updated allocations where 'multi_asset' is distributed among tickers.
        """
        if "multi_asset" not in final_allocations:
            return final_allocations  # No changes needed if multi_asset isn't present

        # Extract multi-asset weight
        multi_asset_weight = final_allocations.pop("multi_asset")

        if multi_asset_weight == 0:
            return final_allocations  # No redistribution needed if weight is zero

        hedge_ratio_series = pd.Series(hedge_ratios)
        hedge_ratio_series /= hedge_ratio_series.abs().sum()  # Normalize hedge ratios

        # Distribute multi-asset weight among tickers based on hedge ratios
        distributed_weights = multi_asset_weight * hedge_ratio_series

        # Add to the existing individual allocations
        final_allocations = final_allocations.add(distributed_weights, fill_value=0)

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

            # Combine the weights with the scaling factors
            combined = (
                kelly_scaling * kelly_weights
                + risk_parity_scaling * risk_parity_weights
            )

            total = combined.sum()
            if total == 0:
                return -np.inf  # Prevent division by zero

            combined /= total  # Normalize so the weights sum to 1

            # Simulate portfolio returns
            portfolio_returns = (combined * returns_df).sum(axis=1)

            # If the portfolio's volatility is zero, return a penalty
            if portfolio_returns.std() == 0:
                return -np.inf

            sr = sharpe_ratio(portfolio_returns)

            if np.isnan(sr):
                return -np.inf

            return sr

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=50)

        best_scaling = study.best_params
        # Combine the weights using the best scaling parameters
        final_weights = (
            best_scaling["kelly_scaling"] * kelly_weights
            + best_scaling["risk_parity_scaling"] * risk_parity_weights
        )
        scaling_total = (
            best_scaling["kelly_scaling"] + best_scaling["risk_parity_scaling"]
        )
        if scaling_total == 0:
            return risk_parity_weights  # fallback
        final_weights /= scaling_total  # Normalize final weights to sum to 1

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
