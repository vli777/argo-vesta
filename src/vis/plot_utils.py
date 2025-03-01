import colorsys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


from config import Config
from utils.logger import logger


def lighten_color(hex_color: str, factor: float = 0.3) -> str:
    """
    Lightens a given hex color by blending it with white.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(1, l + factor)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def get_base_colors(palette: str) -> List[str]:
    """
    Retrieves a list of base colors from Plotly's built-in qualitative palettes.
    If palette=="default", combine several sequences to cover more colors.
    """
    if palette == "default":
        # Combining several built-in sequences
        return (
            px.colors.qualitative.Plotly
            + px.colors.qualitative.D3
            + px.colors.qualitative.G10
        )
    else:
        try:
            return getattr(px.colors.qualitative, palette)
        except AttributeError:
            # Fallback to Plotly palette if the given one is not found
            return px.colors.qualitative.Plotly


def generate_color_map(
    symbols: List[str],
    cumulative_returns: pd.DataFrame,
    palette: str = "default",
) -> Tuple[Dict[str, str], List[str]]:
    """
    Generates a symbol-to-color mapping using Plotly's built-in color palettes.
    If the number of symbols exceeds the number of base colors, it applies a
    lightening factor on subsequent cycles. Symbols are sorted by their final cumulative
    return value (ascending).
    """
    valid_symbols = list(set(symbols).intersection(cumulative_returns.columns))
    if not valid_symbols:
        raise ValueError("No valid symbols provided for plotting.")

    # Sort symbols by their final cumulative return (after filling missing values)
    sorted_symbols = sorted(
        valid_symbols, key=lambda x: cumulative_returns[x].fillna(0).iloc[-1]
    )
    num_symbols = len(sorted_symbols)
    base_colors = get_base_colors(palette)
    n_base = len(base_colors)
    colors = []

    for i in range(num_symbols):
        base_color = base_colors[i % n_base]
        # For each complete cycle, apply a further lightening
        factor = 0.2 * (i // n_base)
        color = lighten_color(base_color, factor=factor) if factor > 0 else base_color
        colors.append(color)

    color_map = {symbol: color for symbol, color in zip(sorted_symbols, colors)}

    # Ensure SIM_PORT is always assigned a specific color
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


def get_text_color(bgcolor: str) -> str:
    """
    Determines a text color based on the background color.
    Returns white if the background is dark, otherwise black.
    """
    hex_color = bgcolor.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return "white" if brightness < 128 else "black"
