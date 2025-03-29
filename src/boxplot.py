from typing import Any, Dict
import numpy as np
import pandas as pd


def generate_boxplot_data(returns_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute boxplot statistics (Q1, median, Q3, whiskers, and outliers)
    for each column in a DataFrame.

    Args:
        returns_df (pd.DataFrame): DataFrame where each column represents a stock's daily returns.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary where each stock symbol maps to its boxplot statistics.
    """
    boxplot_stats = {}

    for col in returns_df.columns:
        data = returns_df[col].dropna().values  # Drop NaNs for accurate calculations
        if len(data) == 0:
            continue  # Skip empty columns

        q1 = np.percentile(data, 25)
        median = np.percentile(data, 50)
        q3 = np.percentile(data, 75)
        # Compute outliers using IQR method
        iqr = q3 - q1

        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        # Whiskers are the most extreme non-outlier values
        lower_whisker = (
            np.min(data[data >= lower_fence]) if np.any(data >= lower_fence) else q1
        )
        upper_whisker = (
            np.max(data[data <= upper_fence]) if np.any(data <= upper_fence) else q3
        )

        outliers = data[(data < lower_fence) | (data > upper_fence)].tolist()

        boxplot_stats[col] = {
            "q1": q1,
            "median": median,
            "q3": q3,
            "lf": lower_fence,
            "uf": upper_fence,
            "min": lower_whisker,
            "max": upper_whisker,
            "outliers": outliers,
        }

    return boxplot_stats
