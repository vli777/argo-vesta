import itertools
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import Config
from vis.plot_utils import get_text_color
from utils.logger import logger


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
