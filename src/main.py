import sys
from typing import List, Optional
from config import Config
from pipeline.core import run_pipeline

from plotly_graphs import plot_graphs
from reversion.reversion_plots import (
    plot_reversion_params,
)
from utils.caching_utils import load_parameters_from_pickle
from utils.logger import logger
from utils.portfolio_utils import estimate_optimal_num_assets


def pipeline_runner(
    config: Config,
    initial_symbols: Optional[List[str]] = None,
    max_epochs: int = 1,
    run_local: bool = False,
    **overrides,
):
    """
    Run the pipeline using the provided config and optional overrides.
    Any keyword arguments provided will update config.options.
    """
    # Update config.options with any provided overrides
    for key, value in overrides.items():
        config.options[key] = value

    # Optional: Validate some overrides
    if "min_weight" in config.options and not isinstance(
        config.options["min_weight"], float
    ):
        raise TypeError("min_weight must be a float")

    if "portfolio_max_size" in config.options:
        if config.options["portfolio_max_size"] is not None and not isinstance(
            config.options["portfolio_max_size"], int
        ):
            raise TypeError("portfolio_max_size must be an integer")
    else:
        # If not provided, estimate the optimal portfolio size
        config.options["portfolio_max_size"] = estimate_optimal_num_assets(
            vol_limit=config.options.get("portfolio_max_vol"),
            portfolio_max_size=config.options.get("portfolio_max_size"),
        )

    symbols = initial_symbols
    previous_top_symbols = set()
    final_result = None
    plot_done = False
    reversion_plotted = False

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}")

        # Enable filters only in the first epoch
        if epoch > 0:
            config.options["use_anomaly_filter"] = False
            config.options["use_decorrelation"] = False

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

            max_size = config.options.get("portfolio_max_size")
            final_symbols = (
                valid_symbols[:max_size]
                if len(valid_symbols) > max_size
                else valid_symbols
            )
            final_result = run_pipeline(
                config=config,
                symbols_override=final_symbols,
            )
            break

        if not config.options.get("portfolio_max_size") or len(
            valid_symbols
        ) <= config.options.get("portfolio_max_size"):
            print(
                f"Stopping epochs as the number of portfolio holdings ({len(valid_symbols)}) "
                f"is <= the configured portfolio max size of {config.options.get('portfolio_max_size')}."
            )
            final_result = result
            break

        # Update for next epoch
        previous_top_symbols = set(valid_symbols)
        symbols = valid_symbols
        final_result = result

    # Plot reversion signals if configured
    if (
        config.options.get("use_reversion")
        and config.options.get("reversion_type") == "z"
        and config.options.get("plot_reversion")
        and not reversion_plotted
    ):
        reversion_cache_file = f"optuna_cache/reversion_cache_{config.options.get('optimization_objective')}.pkl"
        reversion_cache = load_parameters_from_pickle(reversion_cache_file)
        reversion_params = reversion_cache["params"]
        if isinstance(reversion_params, dict):
            plot_reversion_params(data_dict=reversion_params)
        reversion_plotted = True

    # Plot graphs only when running locally and if not already plotted
    if run_local and not plot_done:
        plot_graphs(
            daily_returns=final_result["daily_returns"],
            cumulative_returns=final_result["cumulative_returns"],
            return_contributions=final_result["return_contributions"],
            risk_contributions=final_result["risk_contributions"],
            plot_contribution=config.options["plot_contribution"],
            plot_cumulative_returns=config.options["plot_cumulative_returns"],
            plot_daily_returns=config.options["plot_daily_returns"],
            symbols=final_result["symbols"],
            theme="light",
        )
        plot_done = True

    return final_result


if __name__ == "__main__":
    config_file = "config.yaml"
    config = Config.from_yaml(config_file)

    final_result = pipeline_runner(
        config=config,
        initial_symbols=None,  # Or provide initial symbols as needed
        max_epochs=1,
        run_local=True,
    )
