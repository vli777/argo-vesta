from pathlib import Path
import json
import colorsys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import itertools

from config import Config
from utils.logger import logger


def load_colors_from_json(file_name: str, directory: str = "styles") -> dict:
    """
    Loads color palettes from a JSON file located in a specified directory.

    Parameters
    ----------
    file_name : str
        Name of the JSON file containing color definitions.
    directory : str, optional
        Directory where the JSON file is located, by default "styles".

    Returns
    -------
    dict
        A dictionary where keys are palette names and values are lists of hex color strings.
    """
    # Resolve the file path relative to the script location
    script_dir = Path(__file__).parent
    file_path = script_dir / directory / file_name

    # Load JSON
    with file_path.open("r") as file:
        return json.load(file)


def lighten_color(hex_color: str, factor: float = 0.3) -> str:
    """
    Lightens a given hex color by blending it with white.

    Parameters
    ----------
    hex_color : str
        The original hex color (e.g., '#FF0000').
    factor : float, optional
        Amount to lighten the color, by default 0.3.

    Returns
    -------
    str
        A new, lightened color in hex format.
    """
    # Strip '#' if present
    hex_color = hex_color.lstrip("#")

    # Convert hex to RGB (0-1 range)
    r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Convert to HLS, adjust lightness, and back to RGB
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(1, l + factor)
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    # Convert back to hex
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def generate_color_map(
    symbols: List[str],
    cumulative_returns: pd.DataFrame,
    palette: str = "default",
    color_dict: Dict[str, List[str]] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Generates a symbol-to-color mapping using either a custom palette
    or a default gradient. Symbols are sorted by their final cumulative
    return value (ascending).

    - Ensures symbols are only selected if they exist in cumulative_returns
    - Handles missing values correctly to avoid sorting issues
    """
    valid_symbols = list(set(symbols).intersection(cumulative_returns.columns))

    if not valid_symbols:
        raise ValueError("No valid symbols provided for plotting.")

    # Handle missing data by filling NaN with 0 before sorting
    sorted_symbols = sorted(
        valid_symbols, key=lambda x: cumulative_returns[x].fillna(0).iloc[-1]
    )
    num_symbols = len(sorted_symbols)

    colors = []
    if color_dict and palette in color_dict:
        base_colors = color_dict[palette]
        for i in range(num_symbols):
            base_color = base_colors[i % len(base_colors)]
            lightened_color = lighten_color(
                base_color, factor=0.2 * (i // len(base_colors))
            )
            colors.append(lightened_color)
    else:
        hues = (
            np.linspace(0, 270, num_symbols, endpoint=False) if num_symbols > 1 else [0]
        )
        colors = [f"hsl({int(h)}, 70%, 50%)" for h in hues]

    color_map = {symbol: color for symbol, color in zip(sorted_symbols, colors)}

    # Ensure SIM_PORT is always assigned a color
    if "SIM_PORT" in cumulative_returns.columns:
        color_map["SIM_PORT"] = "hsl(50, 100%, 50%)"

    return color_map, sorted_symbols


def update_plot_layout(
    fig: go.Figure,
    title: str = "",
    paper_bgcolor: str = "#f1f1f1",
    plot_bgcolor: str = "#0476D0",
    hovermode: str = "x unified",
) -> None:
    """
    Updates the layout for a Plotly figure with common styling options.

    Parameters
    ----------
    fig : go.Figure
        The figure to update.
    title : str, optional
        Plot title, by default "".
    paper_bgcolor : str, optional
        Background color of the paper, by default "#f1f1f1".
    plot_bgcolor : str, optional
        Background color of the plotting area, by default "#0476D0".
    hovermode : str, optional
        Hover behavior, by default "x unified".
    """
    fig.update_layout(
        title=title,
        hovermode=hovermode,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        hoverdistance=10,
        margin=dict(l=40, r=40, t=40, b=40),
        hoverlabel=dict(font=dict(size=16), namelength=-1),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            tickmode="auto",
            tickformat="%b %Y",
            ticks="outside",
            type="date",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
        ),
    )


def get_hovertemplate_with_diff() -> str:
    """
    Generates a hovertemplate that shows the symbol name, y-value as a percentage,
    and the difference from SIM_PORT in colored text.

    Returns
    -------
    str
        A Plotly hovertemplate string.
    """
    return (
        "%{meta}: %{y:.2%}"
        ' (<span style="color:%{customdata[1]}">%{customdata[0]:+.2%}</span>)'
        "<extra></extra>"
    )


def get_hovertemplate() -> str:
    """
    Generates a simpler hovertemplate that shows only the symbol name
    and y-value as a percentage.

    Returns
    -------
    str
        A Plotly hovertemplate string.
    """
    return "%{meta}: %{y:.2%}<extra></extra>"


def plot_daily_returns(
    daily_returns: pd.DataFrame,
    color_map: Dict[str, str],
    config: Config,
    paper_bgcolor: str,
    plot_bgcolor: str,
) -> None:
    """
    Plots daily returns as box plots for each symbol, handling missing data correctly.

    - Ensures all stocks are plotted, even if missing in `daily_returns`
    - Assigns a default color only for missing stocks
    """
    if not config.plot_daily_returns:
        return

    fig = go.Figure()

    # Reindex to unify time range across stocks
    all_dates = daily_returns.index
    daily_returns = daily_returns.reindex(index=all_dates, fill_value=np.nan)

    for col in daily_returns.columns:
        fig.add_trace(
            go.Box(
                y=daily_returns[col], marker_color=color_map.get(col, "gray"), name=col
            )
        )

    fig.update_layout(
        title="Daily Returns",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
        yaxis=dict(showgrid=False, zeroline=False),
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
    )

    fig.show()


def plot_cumulative_returns(
    cumulative_returns: pd.DataFrame,
    color_map: Dict[str, str],
    config: Config,
    paper_bgcolor: str,
    plot_bgcolor: str,
) -> None:
    """
    Plots cumulative returns as time-series lines, handling missing data correctly.

    - Ensures all stocks are aligned on the same timeline (unified index)
    - Handles missing `SIM_PORT` properly
    """
    if not config.plot_cumulative_returns:
        return

    fig = go.Figure()

    # Ensure a common time range for all tickers
    all_dates = cumulative_returns.index
    cumulative_returns = cumulative_returns.reindex(index=all_dates, fill_value=np.nan)

    sim_port_data = cumulative_returns.get("SIM_PORT")

    for col in cumulative_returns.columns:
        col_data = cumulative_returns[col]

        if col != "SIM_PORT" and sim_port_data is not None:
            diff = col_data - sim_port_data
            diff_color = ["green" if d >= 0 else "red" for d in diff]
            customdata = np.array([diff.values, diff_color], dtype=object).T

            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=col_data,
                    mode="lines",
                    meta=col,
                    name=col,
                    line=dict(width=3, color=color_map.get(col, "gray")),
                    hovertemplate=get_hovertemplate_with_diff(),
                    customdata=customdata,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=col_data,
                    mode="lines",
                    meta=col,
                    name=col,
                    line=dict(width=6, color=color_map.get(col, "gold")),
                    hovertemplate=get_hovertemplate(),
                )
            )

    update_plot_layout(
        fig,
        title="Cumulative Returns",
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
    )
    fig.show()


def plot_graphs(
    daily_returns: pd.DataFrame,
    cumulative_returns: pd.DataFrame,
    config: Config,
    symbols: List[str],
    paper_bgcolor: str = "#f4f4f4",
    plot_bgcolor: str = "#202346",
    palette: str = "default",
) -> None:
    """
    Creates Plotly graphs for daily returns and cumulative returns.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        DataFrame containing daily returns for each symbol.
    cumulative_returns : pd.DataFrame
        DataFrame containing cumulative returns for each symbol.
    config : Config
        Configuration object with boolean attributes:
            - plot_daily_returns (bool)
            - plot_cumulative_returns (bool)
            - sort_by_weights (bool)
    symbols : List[str]
        List of symbols corresponding to columns in `cumulative_returns`.
    paper_bgcolor : str, optional
        Background color for the entire figure (paper)
    plot_bgcolor : str, optional
        Background color for the plotting area
    """
    # Load color dictionary and generate color map
    color_dict = load_colors_from_json("color_themes.json")
    color_map, sorted_symbols = generate_color_map(
        symbols, cumulative_returns, palette=palette, color_dict=color_dict
    )

    # Plot daily returns
    if config.plot_daily_returns:
        plot_daily_returns(
            daily_returns, color_map, config, paper_bgcolor, plot_bgcolor
        )

    # Plot cumulative returns
    if config.plot_cumulative_returns:
        plot_cumulative_returns(
            cumulative_returns, color_map, config, paper_bgcolor, plot_bgcolor
        )


def plot_risk_return_contributions(
    symbols: List[str], return_contributions: np.ndarray, risk_contributions: np.ndarray
) -> None:
    """
    Plots the return and risk contributions as pie charts in the top row,
    and a Sharpe Ratio bar chart in the bottom row (sorted in ascending order).
    """
    # Ensure all input arrays have the same length
    min_length = min(len(symbols), len(return_contributions), len(risk_contributions))

    if len(symbols) != min_length:
        logger.info(f"Trimming symbols list from {len(symbols)} to {min_length}")
        symbols = symbols[:min_length]

    if len(return_contributions) != min_length:
        logger.warning(
            f"Trimming return contributions from {len(return_contributions)} to {min_length}"
        )
        return_contributions = return_contributions[:min_length]

    if len(risk_contributions) != min_length:
        logger.warning(
            f"Trimming risk contributions from {len(risk_contributions)} to {min_length}"
        )
        risk_contributions = risk_contributions[:min_length]

    # Compute Sharpe Ratio (Avoid division by zero)
    sharpe_ratios = np.divide(
        return_contributions,
        risk_contributions,
        out=np.zeros_like(return_contributions),
        where=risk_contributions != 0,
    )

    # Create DataFrame
    df_contributions = pd.DataFrame(
        {
            "Asset": symbols,
            "Return Contribution (%)": return_contributions.round(2),
            "Risk Contribution (%)": risk_contributions.round(2),
            "Sharpe Ratio": sharpe_ratios.round(2),
        }
    )

    # Sort by Sharpe Ratio in ascending order
    df_contributions = df_contributions.sort_values(by="Sharpe Ratio", ascending=True)

    # Get sorted asset order
    sorted_assets = df_contributions["Asset"].tolist()

    # Get available colors from Jet palette
    available_colors = px.colors.sequential.Plasma + px.colors.sequential.Viridis

    # Cycle colors if more assets exist than colors available
    custom_colors = list(
        itertools.islice(itertools.cycle(available_colors), len(sorted_assets))
    )

    # Create color mapping based on sorted assets
    color_map = {asset: color for asset, color in zip(sorted_assets, custom_colors)}
    sorted_colors = [color_map[asset] for asset in sorted_assets]

    # Create figure with two pie charts (top) and a sorted bar chart (bottom)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Portfolio Return Contribution",
            "Portfolio Risk Contribution",
            "Sharpe Ratio per Asset (higher is better)",
        ),
        specs=[
            [{"type": "domain"}, {"type": "domain"}],  # Two pie charts
            [{"type": "xy", "colspan": 2}, None],  # Bar chart spanning both cols
        ],
        row_heights=[0.6, 0.6],
        vertical_spacing=0.2,
    )

    # Add return contribution pie chart
    fig.add_trace(
        go.Pie(
            labels=df_contributions["Asset"],
            values=df_contributions["Return Contribution (%)"],
            name="Return Contribution",
            hovertemplate="<b>%{label}</b><br>Return Contribution: %{value:.2f}%",
            textinfo="percent",
            hole=0.3,
            marker=dict(colors=sorted_colors),
        ),
        row=1,
        col=1,
    )

    # Add risk contribution pie chart
    fig.add_trace(
        go.Pie(
            labels=df_contributions["Asset"],
            values=df_contributions["Risk Contribution (%)"],
            name="Risk Contribution",
            hovertemplate="<b>%{label}</b><br>Risk Contribution: %{value:.2f}%",
            textinfo="percent",
            hole=0.3,
            marker=dict(colors=sorted_colors),
        ),
        row=1,
        col=2,
    )

    # Add Sharpe Ratio bar chart
    fig.add_trace(
        go.Bar(
            x=df_contributions["Asset"],
            y=df_contributions["Sharpe Ratio"],
            name="Sharpe Ratio",
            marker=dict(color=sorted_colors),
            text=df_contributions["Sharpe Ratio"].map(
                lambda x: f"{x:.2f}"
            ),  # Format labels
            textposition="outside",
        ),
        row=2,
        col=1,
    )

    # Adjust layout for proper spacing & formatting
    fig.update_layout(
        title_text="Portfolio Return, Risk, and Sharpe Ratio Contribution",
        showlegend=False,
        paper_bgcolor="#ffffff",  # Set background to white
        plot_bgcolor="rgba(0,0,0,0)",  # Fully transparent plot area
        margin=dict(t=50, b=50, l=50, r=50),
        height=800,
    )

    # Fix x-axis and remove unnecessary elements from the bar chart
    fig.update_xaxes(title_text="", showgrid=False, zeroline=False, row=2, col=1)
    fig.update_yaxes(
        title_text="",
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        row=2,
        col=1,
    )

    # Show figure
    fig.show()
