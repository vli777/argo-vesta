import sys
from typing import List, Optional
from config import Config
from core import run_pipeline

from plotly_graphs import plot_graphs
from reversion.reversion_plots import (
    plot_reversion_params,
)
from utils.caching_utils import load_parameters_from_pickle
from utils.logger import logger
from utils.portfolio_utils import estimate_optimal_num_assets


def iterative_pipeline_runner(
    config: Config,
    initial_symbols: Optional[List[str]] = None,
    max_epochs: Optional[int] = 1,
    min_weight: Optional[float] = None,
    portfolio_max_size: Optional[int] = None,
    run_local: bool = False,
):
    # Update config with provided arguments if they are valid
    if min_weight is not None:
        if isinstance(min_weight, float):
            config.min_weight = min_weight
        else:
            raise TypeError("min_weight must be a float")

    if portfolio_max_size is not None:
        if isinstance(portfolio_max_size, int):
            config.portfolio_max_size = portfolio_max_size
        else:
            raise TypeError("portfolio_max_size must be an integer")
    else:
        config.portfolio_max_size = estimate_optimal_num_assets(
            vol_limit=config.portfolio_max_vol,
            portfolio_max_size=config.portfolio_max_size,
        )

    # initial_symbols and run_local are handled separately as they may not be part of config
    symbols = initial_symbols
    previous_top_symbols = set()
    final_result = None
    plot_done = False
    reversion_plotted = False

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}")

        # Enable filters only in the first epoch
        if epoch > 0:
            config.use_anomaly_filter = False
            config.use_decorrelation = False

        # Run the pipeline
        result = run_pipeline(
            config=config,
            symbols_override=symbols,
        )

        # Exclude the simulated portfolio symbol from the next epoch
        valid_symbols = [symbol for symbol in result["symbols"] if symbol != "SIM_PORT"]

        logger.info(f"\nTop symbols from epoch {epoch + 1}: {valid_symbols}")

        # Check for convergence
        if set(valid_symbols) == previous_top_symbols:
            print("Convergence reached. Stopping epochs.")

            # Trim to portfolio_max_size if needed
            final_symbols = (
                valid_symbols[: config.portfolio_max_size]
                if len(valid_symbols) > config.portfolio_max_size
                else valid_symbols
            )
            final_result = run_pipeline(
                config=config,
                symbols_override=final_symbols,
            )
            break

        if (
            not config.portfolio_max_size
            or len(valid_symbols) <= config.portfolio_max_size
        ):
            print(
                f"Stopping epochs as the number of portfolio holdings ({len(valid_symbols)}) is <= the configured portfolio max size of {config.portfolio_max_size}."
            )

            # Use the last valid result instead of running the pipeline again
            final_result = result
            break

        # Update symbols for the next epoch
        previous_top_symbols = set(valid_symbols)
        symbols = valid_symbols
        final_result = result

    # Plot reversion signals if configured
    if (
        config.use_reversion
        and config.reversion_type == "z"
        and config.plot_reversion
        and not reversion_plotted
    ):
        reversion_cache_file = (
            f"optuna_cache/reversion_cache_{config.optimization_objective}.pkl"
        )
        reversion_cache = load_parameters_from_pickle(reversion_cache_file)

        reversion_params = reversion_cache["params"]
        if isinstance(reversion_params, dict):
            plot_reversion_params(data_dict=reversion_params)

        reversion_plotted = True

    # Ensure plotting is only done in the final result
    if run_local and not plot_done:
        plot_graphs(
            daily_returns=final_result["daily_returns"],
            cumulative_returns=final_result["cumulative_returns"],
            return_contributions=final_result["return_contributions"],
            risk_contributions=final_result["risk_contributions"],
            config=config,
            symbols=final_result["symbols"],
            theme="light",
        )
        plot_done = True

    return final_result


if __name__ == "__main__":
    config_file = "config.yaml"
    config = Config.from_yaml(config_file)

    final_result = iterative_pipeline_runner(
        config=config,
        initial_symbols=None,  # Or provide initial symbols as needed
        max_epochs=1,
        run_local=True,
    )
