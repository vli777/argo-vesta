from typing import Any, Dict
import numpy as np
import pandas as pd


def generate_boxplot_data(returns_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute boxplot statistics for each column in a DataFrame,
    skipping leading 0s that were likely bfilled before a stock started.

    Args:
        returns_df (pd.DataFrame): DataFrame where each column is a stock's daily returns.

    Returns:
        Dict[str, Dict[str, Any]]: Boxplot stats per stock symbol.
    """
    boxplot_stats = {}

    for col in returns_df.columns:
        col_data = returns_df[col]

        # Remove leading 0s — only those that come before the first non-zero value
        first_valid_index = col_data.ne(0).idxmax()  # First non-zero
        cleaned_data = col_data.loc[first_valid_index:].dropna()

        if cleaned_data.empty:
            continue

        values = cleaned_data.values
        q1 = np.percentile(values, 25)
        median = np.percentile(values, 50)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lf = q1 - 1.5 * iqr
        uf = q3 + 1.5 * iqr
        whisker_low = values[values >= lf].min()
        whisker_high = values[values <= uf].max()
        outliers = values[(values < lf) | (values > uf)].tolist()

        boxplot_stats[col] = {
            "q1": q1,
            "median": median,
            "q3": q3,
            "lf": lf,
            "uf": uf,
            "min": whisker_low,
            "max": whisker_high,
            "outliers": outliers,
        }

    return boxplot_stats
