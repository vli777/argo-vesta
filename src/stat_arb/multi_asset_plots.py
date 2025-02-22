import plotly.graph_objects as go
import pandas as pd


def plot_multi_asset_signals(
    spread_series, signals, title="Multi-Asset Mean Reversion Trading Signals"
):
    """
    Plots the spread series and overlays buy/sell signals using Plotly.

    Args:
        spread_series (pd.Series): The spread time series.
        signals (pd.DataFrame): DataFrame with trading signals (Position column).
        title (str): Plot title.

    Returns:
        None (displays an interactive Plotly plot).
    """
    fig = go.Figure()

    # Add spread series (mean reversion indicator)
    fig.add_trace(
        go.Scatter(
            x=spread_series.index,
            y=spread_series.values,
            mode="lines",
            name="Spread (Z-Score)",
            line=dict(color="blue"),
        )
    )

    # Extract buy/sell signals
    buy_signals = signals[signals["Position"] == 1]
    sell_signals = signals[signals["Position"] == -1]

    # Add Buy signals
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=spread_series.loc[buy_signals.index],
            mode="markers",
            name="Buy Signal",
            marker=dict(symbol="triangle-up", color="green", size=10),
        )
    )

    # Add Sell signals
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=spread_series.loc[sell_signals.index],
            mode="markers",
            name="Sell Signal",
            marker=dict(symbol="triangle-down", color="red", size=10),
        )
    )

    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Spread (Z-Score)",
        template="plotly_white",
    )

    fig.show()


def plot_baseline_vs_reversion_returns(
    baseline_returns,
    adjusted_returns,
    title="Baseline vs. Reversion Adjusted Cumulative Returns",
):
    """
    Plots cumulative returns for the baseline allocation and the reversion adjusted allocation.

    Args:
        baseline_returns (pd.Series): Cumulative returns from the baseline (optimized) allocation.
        adjusted_returns (pd.Series): Cumulative returns from the reversion adjusted allocation.
        title (str): Title of the plot.

    Returns:
        None (displays an interactive Plotly plot).
    """
    fig = go.Figure()

    # Baseline cumulative returns (dashed gray line)
    fig.add_trace(
        go.Scatter(
            x=baseline_returns.index,
            y=baseline_returns.values,
            mode="lines",
            name="Baseline Allocation",
            line=dict(color="gray", width=2, dash="dash"),
        )
    )

    # Reversion adjusted cumulative returns (solid blue line)
    fig.add_trace(
        go.Scatter(
            x=adjusted_returns.index,
            y=adjusted_returns.values,
            mode="lines",
            name="Reversion Adjusted Allocation",
            line=dict(color="blue", width=2),
        )
    )

    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Cumulative Return",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0, y=1.1, orientation="h"),
    )

    fig.show()
