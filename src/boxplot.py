from typing import Any, Dict
import numpy as np
import pandas as pd


def generate_boxplot_data(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute boxplot statistics (Q1, median, Q3, whiskers, and outliers)
    for each column in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame where each column represents a stock's daily returns.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary where each stock symbol maps to its boxplot statistics.
    """
    boxplot_stats = {}

    for col in df.columns:
        data = df[col].dropna().values  # Drop NaNs for accurate calculations
        if len(data) == 0:
            continue  # Skip empty columns

        q1 = np.percentile(data, 25)
        median = np.percentile(data, 50)
        q3 = np.percentile(data, 75)
        lower_whisker = np.min(data)
        upper_whisker = np.max(data)

        # Compute outliers using IQR method
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = [x for x in data if x < lower_fence or x > upper_fence]

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
