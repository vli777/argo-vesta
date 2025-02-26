from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.logger import logger


def plot_anomaly_overview(
    anomalous_assets: list[str], cache: dict, returns_df: pd.DataFrame
) -> None:
    """
    Plots anomalies for assets flagged as anomalous.
    For each asset (that exists in returns_df.columns and in anomalous_assets):
      - The asset’s return series is sliced to start at its first nonzero return.
      - The x-axis is limited to the date range with data and shows only the start and end year.
      - The y-axis is padded (or, if the range is very small, forced to a log scale [0.001, 1]).
      - The returns are plotted as a blue line, and anomaly points (from cache's anomaly_flags)
        are overlaid as red 'x' markers.
      - Each subplot is ensured a minimum height (240px per row) and the layout is scrollable
        if it exceeds the browser window.

    Args:
        anomalous_assets (list): List of assets detected as anomalous.
        cache (dict): Dictionary of asset-specific parameters (including thresholds,
                      anomaly flags, and estimates). May contain keys not present in returns_df.
        returns_df (pd.DataFrame): DataFrame with asset return series.
    """
    # Only consider assets present in returns_df.columns and anomalous_assets.
    assets = list(set(returns_df.columns) & set(anomalous_assets))
    if not assets:
        logger.info("No assets available for plotting anomalies.")
        return

    # Layout: here we use 3 columns; adjust as needed.
    cols = 3
    rows = (len(assets) + cols - 1) // cols

    # Create subplots with one subplot per asset (each subplot gets a title)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=assets)

    for i, asset in enumerate(assets):
        # Get the series for this asset and drop missing values.
        series = returns_df[asset].dropna()
        if series.empty:
            logger.warning(f"No data for asset {asset}. Skipping.")
            continue

        # Slice the series so that it starts at the first nonzero return.
        nonzero = series[series != 0]
        if nonzero.empty:
            logger.warning(f"All returns are zero for asset {asset}. Skipping.")
            continue
        first_valid = nonzero.index[0]
        series = series[series.index >= first_valid]

        # Compute x-axis range and determine tick labels (just start and end year).
        x_min = series.index.min()
        x_max = series.index.max()
        start_year = x_min.year
        end_year = x_max.year

        # Compute y-axis range. If the range is very small, force a [0.001, 1] range with log scale.
        min_val, max_val = series.min(), series.max()
        if max_val - min_val < 0.01:
            y_range = [0.001, 1]
            y_type = "log"
        else:
            padding = (max_val - min_val) * 0.1
            y_range = [min_val - padding, max_val + padding]
            y_type = "linear"

        # Get the asset's anomaly_flags from cache (default: all False).
        asset_info = cache.get(asset, {})
        anomaly_flags = asset_info.get(
            "anomaly_flags", pd.Series(False, index=series.index)
        )
        if not isinstance(anomaly_flags, pd.Series):
            anomaly_flags = pd.Series(False, index=series.index)
        # Slice anomaly_flags to the same date range as series.
        anomaly_flags = anomaly_flags.reindex(series.index, fill_value=False)

        # Identify anomaly points.
        anomalies = series[anomaly_flags == True]

        # Determine the subplot location.
        row_idx = i // cols + 1
        col_idx = i % cols + 1

        # Add the returns line.
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                line=dict(color="blue"),
                name=asset,
            ),
            row=row_idx,
            col=col_idx,
        )
        # Overlay the anomaly markers on the same subplot.
        if not anomalies.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomalies.index,
                    y=anomalies.values,
                    mode="markers",
                    marker=dict(color="red", size=6, symbol="x"),
                    name="Anomaly",
                    showlegend=False,
                ),
                row=row_idx,
                col=col_idx,
            )
        # Update x-axis: limit to the data range and set ticks to just the start and end year.
        fig.update_xaxes(
            range=[x_min, x_max],
            tickmode="array",
            tickvals=[x_min, x_max],
            ticktext=[str(start_year), str(end_year)],
            row=row_idx,
            col=col_idx,
        )
        # Update y-axis: set computed range and scale; remove the y-axis title.
        fig.update_yaxes(
            range=y_range, type=y_type, title_text="", row=row_idx, col=col_idx
        )

    # Update overall layout.
    fig.update_layout(
        autosize=False,
        width=1600,
        height=max(240 * rows, 800),  # at least 240px per row
        margin=dict(t=100),
        title_text="Anomaly Detection Across Assets",
    )

    # Show the figure (with scroll zoom enabled in the browser).
    fig.show(renderer="browser")
