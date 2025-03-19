# src/utils/portfolio_utils.py

from typing import Any, Dict, Optional, Tuple, Union
from collections import defaultdict
import numpy as np
import pandas as pd
from kneed import KneeLocator
import plotly.express as px

from .logger import logger


def convert_to_dict(weights: Any, asset_names: list) -> Dict[str, float]:
    """
    Convert optimizer weights to a dictionary.
    Handles numpy.ndarray, pandas.DataFrame, and dict types.

    Args:
        weights (Any): The weights to convert. Can be ndarray, DataFrame, or dict.
        asset_names (List[str]): List of asset names corresponding to the weights.

    Returns:
        Dict[str, float]: A dictionary mapping asset names to their corresponding weights.

    Raises:
        ValueError: If the length of weights does not match the number of asset names.
        TypeError: If weights is not of type ndarray, DataFrame, or dict.
    """
    if isinstance(weights, np.ndarray):
        logger.debug("Converting weights from numpy.ndarray to dict.")
        if weights.ndim > 1:
            weights = weights.flatten()
            logger.debug(f"Flattened weights: {weights}")

        if len(weights) != len(asset_names):
            logger.error(
                "The number of weights does not match the number of asset names."
            )
            raise ValueError(
                "The number of weights does not match the number of asset names."
            )

        converted = {asset: weight for asset, weight in zip(asset_names, weights)}
        logger.debug(f"Converted weights: {converted}")
        return converted

    elif isinstance(weights, pd.DataFrame):
        logger.debug("Converting weights from pandas.DataFrame to dict.")
        if weights.shape[1] != 1:
            logger.error("DataFrame must have exactly one column for weights.")
            raise ValueError("DataFrame must have exactly one column for weights.")

        weights_series = weights.squeeze()
        if not weights_series.index.equals(pd.Index(asset_names)):
            logger.error(
                "Asset names in DataFrame index do not match the provided asset_names list."
            )
            raise ValueError(
                "Asset names in DataFrame index do not match the provided asset_names list."
            )

        converted = weights_series.to_dict()
        logger.debug(f"Converted weights: {converted}")
        return converted

    elif isinstance(weights, dict):
        logger.debug("Weights are already in dictionary format.")
        if set(weights.keys()) != set(asset_names):
            logger.warning(
                "The asset names in weights do not match the provided asset_names list."
            )
        return weights

    else:
        logger.error(
            f"Unsupported weight type: {type(weights)}. Must be ndarray, DataFrame, or dict."
        )
        raise TypeError("Unsupported weight type: must be ndarray, DataFrame, or dict.")


def convert_weights_to_series(weights, index=None):
    """Convert weights to a pd.Series using the provided index if available."""
    if isinstance(weights, pd.Series):
        return weights
    elif isinstance(weights, dict):
        return pd.Series(weights)
    elif isinstance(weights, np.ndarray):
        if index is not None and len(weights) == len(index):
            return pd.Series(weights, index=index)
        else:
            logger.error(
                f"Mismatch in weights length ({len(weights)}) and expected length ({len(index)})"
            )
            return pd.Series(dtype=float)
    else:
        logger.error(f"Unsupported weights type: {type(weights)}")
        return pd.Series(dtype=float)


def normalize_weights(weights, min_weight: Optional[float] = 0.0) -> pd.Series:
    """
    Normalize the weights by:
      - Filtering out assets whose absolute weight is below a threshold.
        If min_weight > 0, use that value; if min_weight is 0.0, remove any weight below 0.1% (0.001).
      - Scaling the remaining weights so that their sum is exactly 1.
      - Rounding to three decimals.
      - Adjusting one asset so that rounding errors are corrected.

    If all assets are filtered out, the original weights are used.

    Args:
        weights (dict or pd.Series): Input weights (may include negatives if shorts are allowed).
        min_weight (float): Minimum threshold for filtering weights. If 0.0, a threshold of 0.1% (0.001) is used.

    Returns:
        pd.Series: Normalized weights that sum to 1.
    """
    import pandas as pd

    # Convert dict to Series if necessary.
    if isinstance(weights, dict):
        weights = pd.Series(weights)

    # Use the provided min_weight if > 0; otherwise, use 0.1% (0.001) as the threshold.
    threshold = min_weight if min_weight > 0 else 0.001

    # Filter out assets whose absolute weight is below the threshold.
    filtered = weights[weights.abs() >= threshold]
    if filtered.empty:
        # If filtering removes everything, revert to the original weights.
        filtered = weights.copy()

    total_weight = filtered.sum()
    if total_weight == 0:
        raise ValueError("Total weight is zero after filtering. Cannot normalize.")

    # Scale so that the net sum is 1.
    scaled = filtered / total_weight
    # Round to three decimals.
    rounded = scaled.round(3)
    # Adjust the first element to ensure the sum is exactly 1.
    diff = 1.0 - rounded.sum()
    if len(rounded) > 0:
        first = rounded.index[0]
        rounded.loc[first] += diff

    # Finally, remove any asset that rounds to exactly zero and renormalize.
    final = rounded[rounded != 0]
    final = final / final.sum()
    return final


def stacked_output(
    stack_dict: Dict[str, Dict[str, float]],
    period_weights: Optional[Union[Dict[str, float], np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Flatten a dictionary of dictionaries and compute the weighted or arithmetic average of the values.
    Missing keys in any sub-dictionary are treated as having a value of 0.

    Args:
        stack_dict (Dict[str, Dict[str, float]]): A dictionary where each key maps to another
            dictionary of asset weights.
        period_weights (Optional[Union[Dict[str, float], np.ndarray]]): If provided, assigns weights to each
            time period. Can be a dict (with keys matching stack_dict) or a numpy array (with the same order
            as stack_dict keys).

    Returns:
        Dict[str, float]: A single dictionary with averaged weights.
    """
    logger.debug(f"Input stack_dict: {stack_dict}")

    if not stack_dict:
        logger.error("Input stack_dict is empty.")
        raise ValueError("Input stack_dict is empty.")

    # Get a consistent ordering of periods
    period_keys = list(stack_dict.keys())

    # Collect all unique asset names across all portfolios
    all_assets = set()
    for portfolio in stack_dict.values():
        if isinstance(portfolio, dict):
            all_assets.update(portfolio.keys())
        else:
            logger.warning(f"Invalid portfolio format: {portfolio}. Expected a dict.")

    if not all_assets:
        logger.warning("No valid assets found in stack_dict.")
        return {}

    all_assets = sorted(all_assets)  # Ensure consistent ordering

    if period_weights is None:
        # Arithmetic average: sum the weights and divide by the number of portfolios.
        total_weights = defaultdict(float)
        num_portfolios = len(stack_dict)
        for portfolio in stack_dict.values():
            if isinstance(portfolio, dict):
                for asset in all_assets:
                    total_weights[asset] += portfolio.get(asset, 0.0)
        average_weights = {
            asset: round(total_weight / num_portfolios, 3)
            for asset, total_weight in total_weights.items()
        }
        logger.debug(f"Arithmetic average weights: {average_weights}")
        return average_weights
    else:
        # Build the period weight vector.
        if isinstance(period_weights, dict):
            weight_vector = np.array([period_weights.get(k, 0.0) for k in period_keys])
        elif isinstance(period_weights, np.ndarray):
            if period_weights.shape[0] != len(period_keys):
                raise ValueError(
                    "Length of period_weights array does not match number of periods in stack_weights"
                )
            weight_vector = period_weights
        else:
            raise ValueError("period_weights must be a dict or numpy array")

        total_weight = weight_vector.sum()
        if total_weight == 0:
            raise ValueError("Sum of period weights is zero.")
        # Normalize period weights to sum to 1.
        weight_vector = weight_vector / total_weight

        # Build a matrix with rows corresponding to periods and columns to assets.
        A = np.zeros((len(period_keys), len(all_assets)))
        asset_index = {asset: i for i, asset in enumerate(all_assets)}

        for i, period_key in enumerate(period_keys):
            portfolio = stack_dict[period_key]
            for asset, wt in portfolio.items():
                A[i, asset_index[asset]] = wt

        avg_array = np.round(weight_vector.dot(A), 3)
        average_weights = {asset: avg_array[j] for asset, j in asset_index.items()}
        logger.debug(f"Weighted average weights: {average_weights}")
        return average_weights


def holdings_match(
    cached_model_dict: Dict[str, Any], input_file_symbols: list, test_mode: bool = False
) -> bool:
    """
    Check if all selected symbols match between the cached model dictionary and the input file symbols.

    Args:
        cached_model_dict (Dict[str, Any]): Dictionary containing cached model symbols.
        input_file_symbols (List[str]): List of symbols from the input file.
        test_mode (bool, optional): If True, prints mismatched symbols. Defaults to False.

    Returns:
        bool: True if all symbols match, False otherwise.
    """
    cached_symbols = set(cached_model_dict.keys())
    input_symbols = set(input_file_symbols)

    missing_in_input = cached_symbols - input_symbols
    missing_in_cache = input_symbols - cached_symbols

    if missing_in_input:
        if test_mode:
            logger.warning(
                f"Symbols {missing_in_input} are in cached_model_dict but not in input_file_symbols."
            )
        else:
            logger.warning(
                f"Symbols {missing_in_input} are missing in input_file_symbols."
            )
        return False

    if missing_in_cache:
        if test_mode:
            logger.warning(
                f"Symbols {missing_in_cache} are in input_file_symbols but not in cached_model_dict."
            )
        else:
            logger.warning(
                f"Symbols {missing_in_cache} are missing in cached_model_dict."
            )
        return False

    logger.debug("All symbols match between cached_model_dict and input_file_symbols.")
    return True


def trim_weights(weights: Dict[str, float], max_size: int) -> Dict[str, float]:
    """
    Returns a new dict with only the largest `max_size` weights.
    No renormalization is done, assuming you already normalized.
    """
    if len(weights) <= max_size:
        return weights

    # Sort ascending by weight
    sorted_weights = sorted(weights.items(), key=lambda kv: kv[1])
    # Keep the largest `max_size` items
    trimmed = dict(sorted_weights[-max_size:])
    return trimmed


def limit_portfolio_size(
    weights: pd.Series, max_holdings: int, target_sum: float
) -> pd.Series:
    """
    Limit the portfolio to top N holdings by absolute weight and normalize the weights to a target sum.
    Handles both long and short positions correctly.

    Args:
        weights (pd.Series): Series of asset weights (can include negatives for short positions).
        max_holdings (int): Maximum number of assets to retain.
        target_sum (float): The desired total sum for the normalized weights.

    Returns:
        pd.Series: The portfolio weights limited to top holdings and normalized to the target sum.
    """
    # Select top N holdings by absolute weight
    top_holdings = weights.abs().nlargest(max_holdings).index
    limited_weights = weights.loc[top_holdings]

    # Normalize to the specified target sum while preserving the sign
    abs_sum = limited_weights.abs().sum()
    if abs_sum != 0:
        limited_weights = limited_weights / abs_sum * target_sum

    return limited_weights


def marginal_diversification_plot(selected_assets, diversification_values):
    df = pd.DataFrame(
        {
            "Asset": selected_assets,
            "Marginal Diversification Gain": diversification_values,
            "Cumulative Diversification": np.cumsum(diversification_values),
        }
    )

    fig = px.bar(
        df,
        x="Asset",
        y="Marginal Diversification Gain",
        title="Marginal Diversification Value per Asset",
        labels={"Marginal Diversification Gain": "Marginal Diversification Value"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()


def optimal_portfolio_size(returns):
    covariance = returns.cov()
    assets = returns.columns.tolist()

    selected_assets = []
    diversification_values = []
    remaining_assets = set(assets)
    current_cov = None

    # Iteratively add assets based on marginal diversification value
    for _ in range(len(assets)):
        best_asset = None
        best_div_gain = -np.inf

        for asset in remaining_assets:
            trial_assets = selected_assets + [asset]
            trial_cov = covariance.loc[trial_assets, trial_assets]
            eigenvalues = np.linalg.eigvalsh(trial_cov)
            normalized_eig = eigenvalues / np.sum(eigenvalues)
            trial_hhi = 1.0 / np.sum(normalized_eig**2)

            # Marginal Diversification Gain
            if current_cov is None:
                div_gain = trial_hhi
            else:
                current_eigen = np.linalg.eigvalsh(current_cov)
                current_norm_eig = current_eigen / np.sum(current_eigen)
                current_hhi = 1.0 / np.sum(current_norm_eig**2)
                div_gain = trial_hhi - current_hhi

            if div_gain > best_div_gain:
                best_div_gain = div_gain
                best_asset = asset
                best_trial_cov = trial_cov

        selected_assets.append(best_asset)
        diversification_values.append(best_div_gain)
        remaining_assets.remove(best_asset)
        current_cov = best_trial_cov

    # Identify optimal size via elbow method (no manual threshold required)
    cumulative_div = np.cumsum(diversification_values)
    x = range(1, len(cumulative_div) + 1)
    kn = KneeLocator(x, cumulative_div, curve="concave", direction="increasing")
    optimal_n = kn.knee if kn.knee else len(assets)

    # Log for reference
    logger.info(f"Optimal portfolio size based on MDV elbow method: {optimal_n}")

    return optimal_n
