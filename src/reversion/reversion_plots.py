from typing import Dict
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import logger


def plot_reversion_params(data_dict: dict):
    # Convert dictionary to DataFrame
    df = (
        pd.DataFrame.from_dict(data_dict, orient="index")
        .reset_index()
        .rename(columns={"index": "symbol"})
    )

    # Create subplots for positive and negative Z thresholds
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Positive Z Thresholds", "Negative Z Thresholds"),
        horizontal_spacing=0.1,
    )

    # Positive Z threshold 3D scatter plot
    fig.add_trace(
        go.Scatter3d(
            x=df["window_daily"],
            y=df["window_weekly"],
            z=df["z_threshold_daily_positive"],
            mode="markers",
            marker=dict(
                size=5, color=df["z_threshold_daily_positive"], colorscale="Viridis"
            ),
            text=df[
                [
                    "symbol",
                    "window_daily",
                    "window_weekly",
                    "z_threshold_daily_positive",
                    "z_threshold_weekly_positive",
                ]
            ].apply(
                lambda row: f"Symbol: {row['symbol']}<br>Window Daily: {row['window_daily']}<br>Window Weekly: {row['window_weekly']}<br>Z Daily: {row['z_threshold_daily_positive']}<br>Z Weekly: {row['z_threshold_weekly_positive']}",
                axis=1,
            ),
            hoverinfo="text",
        ),
        row=1,
        col=1,
    )

    # Negative Z threshold 3D scatter plot
    fig.add_trace(
        go.Scatter3d(
            x=df["window_daily"],
            y=df["window_weekly"],
            z=df["z_threshold_daily_negative"],
            mode="markers",
            marker=dict(
                size=5, color=df["z_threshold_daily_negative"], colorscale="Cividis"
            ),
            text=df[
                [
                    "symbol",
                    "window_daily",
                    "window_weekly",
                    "z_threshold_daily_negative",
                    "z_threshold_weekly_negative",
                ]
            ].apply(
                lambda row: f"Symbol: {row['symbol']}<br>Window Daily: {row['window_daily']}<br>Window Weekly: {row['window_weekly']}<br>Z Daily: {row['z_threshold_daily_negative']}<br>Z Weekly: {row['z_threshold_weekly_negative']}",
                axis=1,
            ),
            hoverinfo="text",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="3D Scatter Plot of Clusters: Positive and Negative Z Thresholds",
        height=800,
        width=1600,
        margin=dict(l=100, r=100, b=150, t=100),  # Increased bottom margin for padding
    )

    fig.show()


def plot_reversion_signals(data: Dict[str, float]):
    """
    Plots mean reversion signals using Plotly, centering on 0:
      - A signal of 1 => plotted as 0 (no change).
      - Values > 1 => plotted as > 0 (bullish / oversold).
      - Values < 1 => plotted as < 0 (bearish / overbought).

    Positive values are rendered from yellow to green, negative from orange to red.
    """
    if not data:
        print("No reversion signals available for plotting.")
        return

    # Convert dict -> DataFrame with columns ["Asset", "Value"]
    df = pd.DataFrame(list(data.items()), columns=["Asset", "Value"])

    # Filter only numeric signals
    df = df[df["Value"].apply(lambda x: isinstance(x, (int, float)))]
    if df.empty:
        print("No numeric values for reversion signals. Skipping plot.")
        return

    # Subtract 1 so that 1.0 => 0.0
    df["Value"] = df["Value"] - 1

    # Remove tickers whose signal is exactly 0 after subtracting 1
    df = df[df["Value"] != 0]

    if df.empty:
        print("All signals are at baseline (0) or non-existent. Skipping plot.")
        return

    # Sort by Value for better visual
    df = df.sort_values(by="Value")

    # Determine min and max for axis range
    min_val = df["Value"].min()
    max_val = df["Value"].max()
    max_abs = max(abs(min_val), abs(max_val))

    # Function to color negative vs positive
    def get_bar_color(value: float):
        if value < 0:
            # Negative => from orange (value near 0) to deep red (value near min_val)
            norm = abs(value) / abs(min_val) if min_val != 0 else 0
            r = 255 - int((255 - 150) * norm)  # from 255 to ~150
            g = 150 - int(150 * norm)  # from 150 to 0
            b = 0
        else:
            # Positive => from yellow (value near 0) to green (value near max_val)
            norm = value / max_val if max_val != 0 else 0
            r = 255 - int(255 * norm)  # from 255 to 0
            g = 255 - int((255 - 200) * norm)  # from 255 to ~200
            b = 100 - int(100 * norm)  # from 100 to 0
        return (r, g, b)

    # Create color lists for each bar
    bar_colors = []
    text_colors = []
    for val in df["Value"]:
        r, g, b = get_bar_color(val)
        bar_colors.append(f"rgb({r},{g},{b})")
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        text_colors.append("black" if brightness > 140 else "white")

    df["BarColor"] = bar_colors
    df["TextColor"] = text_colors

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df["Asset"],
            x=df["Value"],
            orientation="h",
            marker=dict(color=df["BarColor"]),
            text=df["Asset"],
            textposition="inside",
            textfont=dict(color=df["TextColor"], size=16),
            hoverinfo="x+y+text",
            width=0.8,
        )
    )

    # Configure axes
    fig.update_yaxes(showticklabels=True, showgrid=False, zeroline=True, showline=True)
    fig.update_xaxes(
        range=[-max_abs, max_abs],
        zeroline=True,
        zerolinecolor="grey",
        zerolinewidth=2,
        showgrid=False,
        showline=True,
        showticklabels=True,
    )

    # Layout settings
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(family="Arial", size=14),
        title="Mean Reversion Signals (0 = Baseline)",
    )

    # Overbought/Oversold annotations
    fig.add_annotation(
        x=0.0,
        y=0.5,
        xref="paper",
        yref="paper",
        text="<b>Overbought</b>",
        showarrow=False,
        font=dict(size=16),
        align="center",
        textangle=-90,
        xanchor="left",
        yanchor="middle",
    )
    fig.add_annotation(
        x=1.0,
        y=0.5,
        xref="paper",
        yref="paper",
        text="<b>Oversold</b>",
        showarrow=False,
        font=dict(size=16),
        align="center",
        textangle=90,
        xanchor="right",
        yanchor="middle",
    )

    fig.show()
