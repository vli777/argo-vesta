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
