from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import Config
from vis.cumulative_returns_plot import cumulative_returns_plot
from vis.daily_returns_plot import daily_returns_plot
from vis.risk_return_plot import risk_return_contributions_plot
from vis.utils import generate_color_map
from utils.logger import logger


def plot_graphs(
    daily_returns: pd.DataFrame,
    cumulative_returns: pd.DataFrame,
    return_contributions: np.ndarray,
    risk_contributions: np.ndarray,
    symbols: List[str],
    theme: str = "light",
    palette: str = "default",
    plot_daily_returns: bool = False,
    plot_cumulative_returns: bool = False,
    plot_contribution: bool = False,
) -> None:
    """
    Creates Plotly graphs for daily returns and cumulative returns using
    Plotly's built-in color palettes.
    daily_returns : pd.DataFrame
        DataFrame containing daily returns for each symbol.
    cumulative_returns : pd.DataFrame
        DataFrame containing cumulative returns for each symbol.
    return_contributions : np.ndarray
        np array containing returns contribution to the portfolio for each symbol.
    risk_contributions : np.ndarray
        np array containing risk contribution to the portfolio for each symbol.
    plot_daily_returns (bool)
    plot_cumulative_returns (bool)
    plot_contribution (bool)
    symbols : List[str]
        List of symbols corresponding to columns in `cumulative_returns`.
    theme : str
        The theme for the plot. Options: "light", "dark", "nyan".
    palette : str, optional
        Color palette for the plot.
    """
    # Define theme-based background colors
    theme_colors = {
        "light": "#f4f4f4",
        "dark": "#1e1e1e",
        "nyan": "#202346",
    }

    # Set colors based on theme
    paper_bgcolor = theme_colors.get(theme, "#f4f4f4")
    plot_bgcolor = theme_colors.get(theme, "#f4f4f4")

    color_map, sorted_symbols = generate_color_map(
        symbols, cumulative_returns, palette=palette
    )

    if plot_daily_returns:
        daily_returns_plot(
            daily_returns=daily_returns,
            color_map=color_map,
            paper_bgcolor=paper_bgcolor,
            plot_bgcolor=plot_bgcolor,
        )

    if plot_cumulative_returns:
        cumulative_returns_plot(
            cumulative_returns=cumulative_returns,
            color_map=color_map,
            paper_bgcolor=paper_bgcolor,
            plot_bgcolor=plot_bgcolor,
        )

    if plot_contribution:
        risk_return_contributions_plot(
            symbols=symbols,
            return_contributions=return_contributions,
            risk_contributions=risk_contributions,
            paper_bgcolor=paper_bgcolor,
            plot_bgcolor=plot_bgcolor,
            color_map=color_map,
        )
