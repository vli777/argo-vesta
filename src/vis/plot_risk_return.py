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
    symbols: list,
    return_contributions: np.ndarray,
    risk_contributions: np.ndarray,
    paper_bgcolor: str,
    plot_bgcolor: str,
    color_map: dict,
) -> None:
    """
    Plots a figure with subplots:
      - Top row: two pie charts spaced evenly:
          • Returns Contribution pie chart.
          • Risk Contribution pie chart.
      - Bottom row: a scatter plot of risk (x) vs. return contribution (y),
        where the y-axis is displayed on a decimal scale.

    The hover text in the scatter plot shows the symbol (also in the trace name),
    the original return and risk contributions (in percentage format), and the computed Sharpe ratio.
    """
    text_color = get_text_color(plot_bgcolor)
    subtext_color = "#a3a3a3"
    min_length = min(len(symbols), len(return_contributions), len(risk_contributions))
    symbols = symbols[:min_length]
    return_contributions = return_contributions[:min_length]
    risk_contributions = risk_contributions[:min_length]

    # Preserve original contributions for hover labels and Sharpe ratio calculation.
    risk_orig = risk_contributions.copy()
    return_orig = return_contributions.copy()

    # Compute Sharpe ratios using original values (protect against division by zero).
    risk_for_sharpe = np.where(risk_orig == 0, 1e-3, risk_orig)
    sharpe_ratios = np.divide(
        return_orig,
        risk_for_sharpe,
        out=np.zeros_like(return_orig),
        where=risk_for_sharpe != 0,
    )

    y_scatter = return_orig
    x_scatter = risk_orig

    # Create subplots:
    # - Top row: two pie charts (domain type) for returns and risk contributions.
    # - Bottom row: one scatter plot spanning both columns.
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "domain"}, {"type": "domain"}],
            [{"colspan": 2, "type": "xy"}, None],
        ],
        subplot_titles=(
            "Returns Contribution",
            "Risk Contribution",
            "Risk vs. Return Contribution",
        ),
    )

    # Pie chart for Returns Contribution (top left).
    fig.add_trace(
        go.Pie(
            labels=symbols,
            values=return_orig,
            hoverinfo="label+percent",
            textinfo="label+percent",
            hovertemplate="%{label}<br>Returns contribution: %{percent}",
            showlegend=False,  # No legend for pie chart
            name="",
            textfont=dict(size=10, color=text_color),
        ),
        row=1,
        col=1,
    )

    # Pie chart for Risk Contribution (top right).
    fig.add_trace(
        go.Pie(
            labels=symbols,
            values=risk_orig,
            hoverinfo="label+percent",
            textinfo="label+percent",
            hovertemplate="%{label}<br>Risk contribution: %{percent}",
            showlegend=False,  # No legend for pie chart
            name="",
            textfont=dict(size=10, color=text_color),
        ),
        row=1,
        col=2,
    )

    # Scatter plot (bottom row): one trace per symbol.
    for i, symbol in enumerate(symbols):
        fig.add_trace(
            go.Scatter(
                x=[x_scatter[i]],
                y=[y_scatter[i]],
                mode="markers+text",
                text=[symbol],
                textposition="top center",
                marker=dict(
                    color=color_map.get(symbol, "gray"), size=10, line=dict(width=0)
                ),
                name=symbol,
                textfont=dict(color=text_color),
                hovertemplate=(
                    f"<b>{symbol}</b><br>"
                    f"Return Contribution: {return_orig[i]:.2f}%<br>"
                    f"Risk Contribution: {risk_orig[i]:.2f}%<br>"
                    f"Sharpe Ratio: {sharpe_ratios[i]:.2f}"
                ),
            ),
            row=2,
            col=1,
        )

    # Determine axis limits for the scatter plot directly with decimal values.
    x_min, x_max = min(x_scatter), max(x_scatter)
    y_min, y_max = min(y_scatter), max(y_scatter)

    # Calculate ranges and padding in decimal form.
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_padding = 0.1 * x_range if x_range != 0 else 0.01
    y_padding = 0.1 * y_range if y_range != 0 else 0.01

    # Adjusted axis limits with padding.
    x_low_adj = x_min - x_padding
    x_high_adj = x_max + x_padding
    y_low_adj = y_min - y_padding
    y_high_adj = y_max + y_padding

    # Update scatter plot axes formatting.
    fig.update_xaxes(
        title_text="Risk Contribution",
        tickformat=".2f",  # Show two decimal places for precision
        tickfont=dict(color=text_color),
        range=[x_low_adj, x_high_adj],
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Returns Contribution",
        tickformat=".2f",  # Show two decimal places for precision
        tickfont=dict(color=text_color),
        range=[y_low_adj, y_high_adj],
        row=2,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text="Portfolio Contributions",
            font=dict(
                family="Roboto, sans-serif",
                size=32,
                weight="bold",
                color=text_color,
            ),
            x=0.02,
            y=0.98,
            xanchor="left",
        ),
        annotations=[
            dict(
                text="Returns Contribution",
                showarrow=False,
                font=dict(color=subtext_color),
            ),
            dict(
                text="Risk Contribution",
                showarrow=False,
                font=dict(color=subtext_color),
            ),
            dict(
                text="Risk vs. Return Contribution",
                showarrow=False,
                font=dict(color=subtext_color),
            ),
        ],
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            tickfont=dict(color=subtext_color),
            title_font=dict(color=subtext_color),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=subtext_color),
            title_font=dict(color=subtext_color),
        ),
        legend=dict(font=dict(color=text_color)),  # Ensure legend color matches theme
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        showlegend=True,
    )

    fig.show()
