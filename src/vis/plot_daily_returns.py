from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


from config import Config
from vis.plot_utils import get_text_color
from utils.logger import logger


def plot_daily_returns(
    daily_returns: pd.DataFrame,
    color_map: Dict[str, str],
    config: Config,
    paper_bgcolor: str,
    plot_bgcolor: str,
) -> None:
    if not config.plot_daily_returns:
        return

    # Determine text color based on the plot background color.
    text_color = get_text_color(plot_bgcolor)
    subtext_color = "#a3a3a3"
    fig = go.Figure()
    all_dates = daily_returns.index
    daily_returns = daily_returns.reindex(index=all_dates, fill_value=np.nan)

    for col in daily_returns.columns:
        asset_data = daily_returns[col].dropna()
        if asset_data.empty:
            continue

        # Calculate statistics.
        q1 = asset_data.quantile(0.25)
        q3 = asset_data.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        min_val = asset_data.min()
        max_val = asset_data.max()

        # Compute an offset (5% of the range) to position the annotation above the highest value.
        offset = (max_val - min_val) * 0.05 if (max_val - min_val) != 0 else 0.01

        # Add the box trace for the asset.
        fig.add_trace(
            go.Box(
                y=daily_returns[col],
                marker_color=color_map.get(col, "gray"),
                name=col,
            )
        )

        # Create annotation text with all statistics formatted as percentages.
        annotation_text = (
            f"Min: {min_val:.2%}<br>"
            f"LF: {lower_fence:.2%}<br>"
            f"Q1: {q1:.2%}<br>"
            f"Q3: {q3:.2%}<br>"
            f"UF: {upper_fence:.2%}<br>"
            f"Max: {max_val:.2%}"
        )

        # Add the annotation above the asset's highest value.
        fig.add_annotation(
            x=col,
            y=max_val + offset,
            text=annotation_text,
            showarrow=False,
            font=dict(color=text_color, size=10),
            xanchor="center",
            yanchor="bottom",
            align="center",
        )

    fig.update_layout(
        title=dict(
            text="Daily Returns",
            font=dict(
                family="Roboto, sans-serif",
                size=32,
                weight="bold",
                color=text_color,
            ),  # Title color
            x=0.02,  # Left padding
            y=0.98,
            xanchor="left",
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            tickfont=dict(color=subtext_color),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=subtext_color),
        ),
        legend=dict(font=dict(color=text_color)),  # Ensure legend color matches theme
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
    )

    fig.show()
