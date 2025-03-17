from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import Config
from boxplot import generate_boxplot_data
from utils.performance_metrics import (
    calculate_portfolio_alpha,
    calculate_portfolio_performance,
    conditional_var,
    kappa_ratio,
    max_drawdown,
    omega_ratio,
    portfolio_volatility,
    risk_return_contributions,
    sharpe_ratio,
    time_under_water,
)
from utils import logger
from utils.portfolio_utils import (
    trim_weights,
)


def output(
    data: pd.DataFrame,
    allocation_weights: Dict[str, float],
    start_date: Any,
    end_date: Any,
    inputs: Optional[str] = None,
    optimization_model: Optional[str] = None,
    time_period: float = 1.0,
    config: Config = None,
):
    """
    Produces console output (stats) and returns daily/cumulative returns of
    the portfolio + each ticker.
    """

    # Ensure allocation_weights is a dictionary
    clean_weights = (
        allocation_weights
        if isinstance(allocation_weights, dict)
        else dict(allocation_weights)
    )

    # Validate that data contains all required symbols
    missing_symbols = [
        symbol for symbol in clean_weights.keys() if symbol not in data.columns
    ]
    if missing_symbols:
        logger.warning(
            f"The following symbols are missing in the data and will be filled with zeros: {missing_symbols}"
        )
        missing_df = pd.DataFrame(0, index=data.index, columns=missing_symbols)
        data = pd.concat([data, missing_df], axis=1)

    # Trim if we have more assets than allowed
    portfolio_max_size = (
        len(clean_weights)
        if config.portfolio_max_size is None
        else min(config.portfolio_max_size, len(clean_weights))
    )
    # portfolio_max_size = estimate_optimal_num_assets(
    #     vol_limit=config.portfolio_max_vol,
    #     portfolio_max_size=config.portfolio_max_size,
    # ) or len(clean_weights)

    if len(clean_weights) > portfolio_max_size:
        clean_weights = trim_weights(clean_weights, portfolio_max_size)

    # If nothing left, can't proceed
    if len(clean_weights) == 0:
        logger.info("No assets left after trimming or filtering. Exiting.")
        sys.exit()

    # --- Calculate portfolio performance
    (
        returns,
        portfolio_returns,
        portfolio_cumulative_returns,
        (all_daily_returns, all_cumulative_returns),
    ) = calculate_portfolio_performance(data[list(clean_weights.keys())], clean_weights)
    # from scipy.stats import skew
    # print("Portfolio return skew:", skew(portfolio_returns))

    sharpe = sharpe_ratio(portfolio_returns, risk_free_rate=config.risk_free_rate)
    # kappa = kappa_ratio(portfolio_returns, risk_free_rate=config.risk_free_rate)
    omega = omega_ratio(portfolio_returns, risk_free_rate=config.risk_free_rate)
    volatility = portfolio_volatility(portfolio_returns)
    cvar = conditional_var(portfolio_returns)
    max_dd = max_drawdown(portfolio_cumulative_returns)
    time_uw = time_under_water(portfolio_cumulative_returns)
    boxplot_stats = generate_boxplot_data(all_daily_returns)

    # --- Load market data and compute market returns over the same period
    market_file = Path(config.data_dir) / "SPY.parquet"
    alpha = None  # Default to None if SPY is missing

    try:
        market_data = pd.read_parquet(market_file)

        if "Adj Close" in market_data.columns:
            market_returns = (
                np.log(market_data["Adj Close"]).diff().ffill().dropna(how="all")
            )
        elif "Close" in market_data.columns:
            market_returns = (
                np.log(market_data["Close"]).diff().ffill().dropna(how="all")
            )
        else:
            raise ValueError(
                f"{market_file} is missing 'Adj Close' or 'Close' columns."
            )

        # Filter market_returns to match the analysis period
        market_returns = market_returns.loc[start_date:end_date]

        if not market_returns.empty:
            alpha = calculate_portfolio_alpha(
                portfolio_returns=portfolio_returns,
                market_returns=market_returns,
                weights=clean_weights,
                risk_free_rate=config.risk_free_rate,
            )
        else:
            print(f"Warning: Market data for {market_file} is empty after filtering.")

    except FileNotFoundError:
        print(
            f"Warning: Market data file {market_file} not found. Skipping Alpha calculation."
        )
    except Exception as e:
        print(
            f"Error loading market data from {market_file}: {e}. Skipping Alpha calculation."
        )

    cumulative_pct = round((portfolio_cumulative_returns.iloc[-1] - 1) * 100, 2)

    # Logging results (using print for cleaner output)
    if inputs is not None:
        print("\nWatchlist Inputs:", inputs)

    print(f"Time Period:\t{start_date} to {end_date} ({time_period} yrs)")
    print(f"Optimization Method:\t{optimization_model}")
    print(f"Sharpe Ratio:\t\t{sharpe:.2f}")
    # print(f"Kappa Ratio:\t\t{kappa:.2f}")
    print(f"Omega Ratio:\t\t{omega:.2f}")
    print(f"Portfolio Volatility:\t{volatility * 100:.2f}%")
    print(f"Conditional VaR:\t{cvar * 100:.2f}%")
    print(f"Max Drawdown:\t\t{max_dd * 100:.2f}%")
    print(f"Time Under Water:\t{time_uw} days")

    # Only print Alpha if it was successfully calculated
    if alpha is not None:
        print(f"Portfolio Alpha:\t{alpha * 100:.2f}%")

    print(f"Cumulative Return:\t{cumulative_pct:.2f}%")

    # Print portfolio allocation in tab-separated format
    sorted_weights = sorted(clean_weights.items(), key=lambda kv: kv[1], reverse=True)
    print("Symbol\tWeight\tLW\tQ1\tQ3\tUW")
    for asset, weight in sorted_weights:
        lw = boxplot_stats[asset]["lf"] * 100
        q1 = boxplot_stats[asset]["q1"] * 100
        q3 = boxplot_stats[asset]["q3"] * 100
        uw = boxplot_stats[asset]["uf"] * 100
        print(f"{asset}\t{weight * 100:.2f}%\t{lw:.2f}\t{q1:.2f}\t{q3:.2f}\t{uw:.2f}")

    return all_daily_returns, all_cumulative_returns, boxplot_stats


def output_results(
    df: pd.DataFrame,
    weights: Dict[str, float],
    model_name: str,
    start_date: Any,
    end_date: Any,
    years: float,
    config: Config,
) -> None:
    """
    Wrapper around `output` for portfolio optimization results.
    Handles logging and reporting without returning performance data.
    """
    output(
        data=df,
        allocation_weights=weights,
        inputs=model_name,
        start_date=start_date,
        end_date=end_date,
        optimization_model=model_name,
        time_period=years,
        config=config,
    )


def compute_performance_results(
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    allocation_weights: Dict[str, float],
    sorted_symbols: list,
    combined_input_files: str,
    combined_models: str,
    sorted_time_periods: list,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Compute daily returns, cumulative returns, boxplot stats, and risk/return contributions.

    Returns:
        daily_returns, cumulative_returns, boxplot_stats,
        return_contributions_pct, risk_contributions_pct, valid_symbols
    """
    # Compute daily & cumulative returns
    daily_returns, cumulative_returns, boxplot_stats = output(
        data=data,
        start_date=start_date,
        end_date=end_date,
        allocation_weights=allocation_weights,
        inputs=combined_input_files,
        optimization_model=f"{combined_models} ENSEMBLE: {config.optimization_objective}",
        time_period=sorted_time_periods[0],
        config=config,
    )

    # Filter valid symbols: those present in cumulative_returns
    valid_symbols = [s for s in sorted_symbols if s in cumulative_returns.columns]

    # Re-sort valid_symbols according to the original order in sorted_symbols
    # (Since sorted_symbols is already sorted, this ensures consistency.)
    valid_symbols = sorted(valid_symbols, key=lambda s: sorted_symbols.index(s))

    # Build weights array in the same order
    np_weights = np.array([allocation_weights[s] for s in valid_symbols])

    # Extract final cumulative returns per stock (in the same order)
    contribution_cumulative_returns = cumulative_returns.loc[
        cumulative_returns.index[-1], valid_symbols
    ].values

    # Compute risk and return contributions
    return_contributions_pct, risk_contributions_pct = risk_return_contributions(
        weights=np_weights,
        daily_returns=daily_returns[valid_symbols],
        cumulative_returns=contribution_cumulative_returns,
    )

    return (
        daily_returns,
        cumulative_returns,
        boxplot_stats,
        return_contributions_pct,
        risk_contributions_pct,
        valid_symbols,
    )


def build_final_result_dict(
    start_date: str,
    end_date: str,
    models: str,
    symbols: list,
    normalized_avg: dict,
    daily_returns: pd.DataFrame,
    cumulative_returns: pd.DataFrame,
    boxplot_stats,
    return_contributions: np.ndarray,
    risk_contributions: np.ndarray,
) -> dict:
    """
    Constructs the final results dictionary.
    """
    return {
        "start_date": start_date,
        "end_date": end_date,
        "models": models,
        "symbols": symbols,
        "normalized_avg": normalized_avg,
        "daily_returns": daily_returns,
        "cumulative_returns": cumulative_returns,
        "boxplot_stats": boxplot_stats,
        "return_contributions": return_contributions,
        "risk_contributions": risk_contributions,
    }
