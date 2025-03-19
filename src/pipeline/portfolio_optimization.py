from pathlib import Path
import sys
from typing import Any, Callable, Dict, List
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import Config
from models.nco import nested_clustered_optimization
from models.optimize_portfolio import estimated_portfolio_volatility
from utils import logger
from utils.caching_utils import (
    load_model_results_from_cache,
    make_cache_key,
    save_model_results_to_cache,
)
from utils.portfolio_utils import (
    convert_weights_to_series,
    limit_portfolio_size,
    normalize_weights,
    optimal_portfolio_size,
)

# Dispatch table mapping model names to their corresponding optimization functions
OPTIMIZATION_DISPATCH: Dict[str, Callable[..., Any]] = {
    "nested_clustering": nested_clustered_optimization,
    # Future models can be added here
}


def run_optimization(
    model: str, cov: pd.DataFrame, mu: pd.Series, args: Dict[str, Any]
) -> pd.Series:
    """
    Dispatch to the appropriate optimization function based on `model` and provided arguments.
    Returns weights or optimization results.
    """
    try:
        optimization_func = OPTIMIZATION_DISPATCH[model]
    except KeyError:
        raise ValueError(f"Unsupported optimization method: {model}")

    # Pass configuration parameters directly to the optimization function
    return optimization_func(cov=cov, mu=mu, **args)


def run_optimization_and_save(
    df: pd.DataFrame,
    config: Config,
    start_date: str,
    end_date: str,
    symbols: List[str],
    stack: Dict,
    years: str,
    plot: bool,
):
    final_weights = None

    for model in config.models[years]:
        cache_key = make_cache_key(
            model=model,
            years=years,
            objective=config.optimization_objective,
            symbols=symbols,
        )

        # Check cache
        # cached = load_model_results_from_cache(cache_key)
        # if cached is not None:
        #     print(f"Using cached results for {model} with {years} years.")
        #     normalized_weights = normalize_weights(cached, config.min_weight)
        #     final_weights = normalized_weights
        #     stack[model + str(years)] = normalized_weights.to_dict()
        #     save_model_results_to_cache(cache_key, final_weights.to_dict())
        # else:

        # Ensure shorter-history stocks are retained
        asset_returns = np.log(df[symbols]).diff().dropna(how="all")

        # Ensure covariance matrix is computed only on valid assets
        valid_assets = asset_returns.dropna(
            thresh=int(len(asset_returns) * 0.5), axis=1
        ).columns
        asset_returns = asset_returns[valid_assets]

        # Compute covariance with aligned data
        try:
            lw = LedoitWolf()
            cov_daily = lw.fit(asset_returns).covariance_
            cov_daily = pd.DataFrame(
                cov_daily, index=valid_assets, columns=valid_assets
            )
        except ValueError as e:
            logger.error(f"Covariance computation failed: {e}")
            return pd.Series(dtype=float)

        trading_days_per_year = 252
        mu_daily = asset_returns.mean()
        mu_annual = mu_daily * trading_days_per_year
        cov_annual = cov_daily * trading_days_per_year

        # Ensure `mu` is aligned with covariance matrix
        mu_annual = mu_annual.loc[valid_assets]
        mu_annual = mu_annual.reindex(valid_assets)

        min_weight = config.min_weight
        max_weight = config.max_weight
        optimization_objective = config.optimization_objective
        cluster_method = config.clustering_type
        allow_short = config.allow_short
        max_gross_exposure = config.max_gross_exposure
        risk_free_rate = config.risk_free_rate
        risk_free_rate_log_daily = np.log(1 + risk_free_rate) / trading_days_per_year
        use_annealing = (
            config.use_global_optimization
            and config.global_optimization_type == "annealing"
        )
        use_diffusion = (
            config.use_global_optimization
            and config.global_optimization_type == "diffusion"
        )

        model_args = {
            "returns": asset_returns,
            "min_weight": min_weight,
            "max_weight": max_weight,
            "objective": optimization_objective,
            "allow_short": allow_short,
            "max_gross_exposure": max_gross_exposure,
            "risk_free_rate": risk_free_rate_log_daily,
            "use_annealing": use_annealing,
            "use_diffusion": use_diffusion,
            "plot": plot,
            "cluster_method": cluster_method,
        }

        try:
            weights = run_optimization(
                model=model,
                cov=cov_annual,
                mu=mu_annual,
                args=model_args,
            )
            # Convert weights from Pandas Series to NumPy array
            weights_array = weights.to_numpy().reshape(
                -1, 1
            )  # Ensure it's a column vector
            # Ensure weights align with cov_annual's tickers
            weights = weights.reindex(cov_annual.index, fill_value=0)
            # Convert to NumPy and compute portfolio volatility
            weights_array = weights.to_numpy()
            current_vol: float = estimated_portfolio_volatility(
                weights_array, cov_annual.to_numpy()
            )
            try:
                portfolio_max_size = (
                    config.portfolio_max_size
                    if config.portfolio_max_size is not None
                    else optimal_portfolio_size(returns=asset_returns, threshold=0.95)
                )
            except Exception:  # Handles errors in `optimal_portfolio_size`
                portfolio_max_size = len(weights)

            logger.info(f"portfolio max size: {portfolio_max_size}")
            weights = convert_weights_to_series(weights, index=mu_annual.index)
            normalized_weights = normalize_weights(weights, config.min_weight)
            final_weights = limit_portfolio_size(
                normalized_weights, portfolio_max_size, target_sum=1.0
            )
            # 3) Save new result to cache
            save_model_results_to_cache(cache_key, final_weights.to_dict())

            # 4) Update your stack
            stack[model + str(years)] = final_weights.to_dict()

        except Exception as e:
            logger.error(
                f"Error processing weights for {model} {optimization_objective} {years} years: {e}"
            )
            final_weights = pd.Series(dtype=float)

        # Ensure final_weights is valid
        if final_weights is None or final_weights.empty:
            logger.warning(
                f"No valid weights generated for {model} {optimization_objective} ({years} years)."
            )
            final_weights = pd.Series(dtype=float)  # Default empty Series
