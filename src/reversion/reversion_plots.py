from typing import Dict
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.logger import logger


# def plot_reversion_params(data_dict):
#     """
#     Generates two interactive Plotly figures:
#       - Left: Distribution of Daily Windows with Spline Fit
#       - Right: Distribution of Weekly Windows with Spline Fit

#     Expects data_dict to contain the following keys:
#       - "cluster" (int or str)
#       - "window_daily" (int)
#       - "z_threshold_daily_positive" (float)
#       - "z_threshold_daily_negative" (float)
#       - "window_weekly" (int)
#       - "z_threshold_weekly_positive" (float)
#       - "z_threshold_weekly_negative" (float)
#     """
#     if not data_dict:
#         print("No data available for plotting reversion parameters.")
#         return

#     # Convert dictionary to DataFrame and ensure cluster is treated as a string
#     df = pd.DataFrame.from_dict(data_dict, orient="index").reset_index()
#     df = df.rename(columns={"index": "cluster"})
#     df["cluster"] = df["cluster"].astype(str)

#     # Create the figure with two subplots
#     fig = go.Figure()

#     # Daily window distribution
#     daily_bar_data = df["window_daily"].value_counts().reset_index()
#     daily_bar_data.columns = ["window", "count"]
#     daily_bar_data = daily_bar_data.sort_values("window")

#     # Hover text showing tickers in each cluster for daily windows
#     daily_hover_text = (
#         df.groupby("window_daily")["cluster"]
#         .apply(lambda x: ", ".join(x))
#         .reindex(daily_bar_data["window"])
#         .fillna("")
#     )

#     # Bar chart for daily windows
#     daily_bars = go.Bar(
#         x=daily_bar_data["window"],
#         y=daily_bar_data["count"],
#         text=daily_hover_text,
#         hoverinfo="text",
#         marker_color="rgba(55, 83, 109, 0.7)",
#         name="Daily Window Distribution",
#         xaxis="x1",
#         yaxis="y1",
#     )

#     # Spline interpolated curve for daily windows
#     daily_spline_x = np.linspace(
#         daily_bar_data["window"].min(), daily_bar_data["window"].max(), 200
#     )
#     daily_spline_y = np.interp(
#         daily_spline_x, daily_bar_data["window"], daily_bar_data["count"]
#     )

#     daily_curve = go.Scatter(
#         x=daily_spline_x,
#         y=daily_spline_y,
#         mode="lines",
#         line=dict(shape="spline", color="rgba(26, 118, 255, 0.7)", width=3),
#         name="Daily Spline Fit",
#         xaxis="x1",
#         yaxis="y1",
#     )

#     # Weekly window distribution
#     weekly_bar_data = df["window_weekly"].value_counts().reset_index()
#     weekly_bar_data.columns = ["window", "count"]
#     weekly_bar_data = weekly_bar_data.sort_values("window")

#     # Hover text showing tickers in each cluster for weekly windows
#     weekly_hover_text = (
#         df.groupby("window_weekly")["cluster"]
#         .apply(lambda x: ", ".join(x))
#         .reindex(weekly_bar_data["window"])
#         .fillna("")
#     )

#     # Bar chart for weekly windows
#     weekly_bars = go.Bar(
#         x=weekly_bar_data["window"],
#         y=weekly_bar_data["count"],
#         text=weekly_hover_text,
#         hoverinfo="text",
#         marker_color="rgba(255, 99, 71, 0.7)",
#         name="Weekly Window Distribution",
#         xaxis="x2",
#         yaxis="y2",
#     )

#     # Spline interpolated curve for weekly windows
#     weekly_spline_x = np.linspace(
#         weekly_bar_data["window"].min(), weekly_bar_data["window"].max(), 200
#     )
#     weekly_spline_y = np.interp(
#         weekly_spline_x, weekly_bar_data["window"], weekly_bar_data["count"]
#     )

#     weekly_curve = go.Scatter(
#         x=weekly_spline_x,
#         y=weekly_spline_y,
#         mode="lines",
#         line=dict(shape="spline", color="rgba(255, 165, 0, 0.7)", width=3),
#         name="Weekly Spline Fit",
#         xaxis="x2",
#         yaxis="y2",
#     )

#     # Add all traces to the figure
#     fig.add_traces([daily_bars, daily_curve, weekly_bars, weekly_curve])

#     # Update layout for subplots
#     fig.update_layout(
#         title="Distribution of Daily and Weekly Windows with Spline Fit",
#         grid=dict(rows=1, columns=2, pattern="independent"),
#         xaxis=dict(title="Daily Window", domain=[0.0, 0.45]),
#         yaxis=dict(title="Frequency"),
#         xaxis2=dict(title="Weekly Window", domain=[0.55, 1.0]),
#         yaxis2=dict(title="Frequency", anchor="x2"),
#         template="plotly_white",
#         showlegend=True,
#     )

#     fig.show()


def plot_reversion_params(data_dict):
    """
    Converts a dictionary of mean reversion parameters into a DataFrame
    and generates an interactive Plotly figure with two subplots:
      - Left: Daily Window vs. Z-Threshold (Positive and Negative)
      - Right: Weekly Window vs. Z-Threshold (Positive and Negative)

    Expects data_dict to contain the following keys (at minimum):
      - "cluster"
      - "window_daily"
      - "z_threshold_daily_positive"
      - "z_threshold_daily_negative"
      - "window_weekly"
      - "z_threshold_weekly_positive"
      - "z_threshold_weekly_negative"
    """
    if not data_dict:
        print("No data available for plotting reversion parameters.")
        return

    # Convert dictionary to DataFrame and use assets as clusters
    df = (
        pd.DataFrame.from_dict(data_dict, orient="index")
        .reset_index()
        .rename(columns={"index": "cluster"})  # Treat asset as cluster
    )

    required_columns = {
        "window_daily",
        "z_threshold_daily_positive",
        "z_threshold_daily_negative",
        "window_weekly",
        "z_threshold_weekly_positive",
        "z_threshold_weekly_negative",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Missing columns in data: {missing_columns}")
        return

    # Convert all cluster values (assets) to strings and then map to numeric codes.
    df["cluster_str"] = df["cluster"].astype(str)
    unique_clusters = {
        label: idx for idx, label in enumerate(sorted(df["cluster_str"].unique()))
    }
    df["cluster_numeric"] = df["cluster_str"].map(unique_clusters)

    # Apply jitter to prevent overlap
    jitter_scale = 0.2
    df["window_daily_jitter"] = df["window_daily"] + np.random.normal(
        0, jitter_scale, df.shape[0]
    )
    df["z_threshold_daily_positive_jitter"] = df[
        "z_threshold_daily_positive"
    ] + np.random.normal(0, jitter_scale, df.shape[0])
    df["z_threshold_daily_negative_jitter"] = df[
        "z_threshold_daily_negative"
    ] + np.random.normal(0, jitter_scale, df.shape[0])
    df["window_weekly_jitter"] = df["window_weekly"] + np.random.normal(
        0, jitter_scale, df.shape[0]
    )
    df["z_threshold_weekly_positive_jitter"] = df[
        "z_threshold_weekly_positive"
    ] + np.random.normal(0, jitter_scale, df.shape[0])
    df["z_threshold_weekly_negative_jitter"] = df[
        "z_threshold_weekly_negative"
    ] + np.random.normal(0, jitter_scale, df.shape[0])

    # Create the figure with two subplots
    fig = go.Figure()

    # Left subplot: Daily parameters
    fig.add_trace(
        go.Scatter(
            x=df["window_daily_jitter"],
            y=df["z_threshold_daily_positive_jitter"],
            mode="markers",
            marker=dict(
                color=df["cluster_numeric"],
                colorscale="viridis",
                size=8,
                opacity=0.7,
                symbol="circle",
            ),
            text=df.apply(
                lambda row: f"Asset: {row['cluster']}<br>Window (Daily): {row['window_daily']}<br>Positive Z-Threshold: {row['z_threshold_daily_positive']}",
                axis=1,
            ),
            hoverinfo="text",
            name="Daily Positive",
            xaxis="x1",
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["window_daily_jitter"],
            y=df["z_threshold_daily_negative_jitter"],
            mode="markers",
            marker=dict(
                color=df["cluster_numeric"],
                colorscale="viridis",
                size=8,
                opacity=0.7,
                symbol="square",
            ),
            text=df.apply(
                lambda row: f"Asset: {row['cluster']}<br>Window (Daily): {row['window_daily']}<br>Negative Z-Threshold: {row['z_threshold_daily_negative']}",
                axis=1,
            ),
            hoverinfo="text",
            name="Daily Negative",
            xaxis="x1",
            yaxis="y1",
        )
    )

    # Right subplot: Weekly parameters
    fig.add_trace(
        go.Scatter(
            x=df["window_weekly_jitter"],
            y=df["z_threshold_weekly_positive_jitter"],
            mode="markers",
            marker=dict(
                color=df["cluster_numeric"],
                colorscale="viridis",
                size=8,
                opacity=0.7,
                symbol="circle",
            ),
            text=df.apply(
                lambda row: f"Ticker: {row['cluster']}<br>Window (Weekly): {row['window_weekly']}<br>Positive Z-Threshold: {row['z_threshold_weekly_positive']}",
                axis=1,
            ),
            hoverinfo="text",
            name="Weekly Positive",
            xaxis="x2",
            yaxis="y2",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["window_weekly_jitter"],
            y=df["z_threshold_weekly_negative_jitter"],
            mode="markers",
            marker=dict(
                color=df["cluster_numeric"],
                colorscale="viridis",
                size=8,
                opacity=0.7,
                symbol="square",
            ),
            text=df.apply(
                lambda row: f"Ticker: {row['cluster']}<br>Window (Weekly): {row['window_weekly']}<br>Negative Z-Threshold: {row['z_threshold_weekly_negative']}",
                axis=1,
            ),
            hoverinfo="text",
            name="Weekly Negative",
            xaxis="x2",
            yaxis="y2",
        )
    )

    # Update layout
    fig.update_layout(
        title="Mean Reversion Clusters: Daily vs. Weekly Parameters",
        grid=dict(rows=1, columns=2, pattern="independent"),
        xaxis=dict(title="Window (Daily)", domain=[0.0, 0.45]),
        yaxis=dict(title="Z-Threshold (Daily)"),
        xaxis2=dict(title="Window (Weekly)", domain=[0.55, 1.0]),
        yaxis2=dict(title="Z-Threshold (Weekly)", anchor="x2"),
        template="plotly_white",
        legend_title="Parameter Type",
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
