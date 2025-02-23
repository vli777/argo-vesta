import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_multi_ou_signals(mar, signals, stop_loss, take_profit):
    """
    Create a Plotly figure plotting the scaled OU deviation, the threshold levels,
    the 20-day rolling z-score of the spread, and the buy/sell signals.

    Parameters:
        mar: MultiAssetReversion instance containing spread_series, ou_mu, and scale.
        signals: DataFrame with datetime index and a "Position" column.
        stop_loss: The long-entry threshold (a negative number, in scaled units).
        take_profit: The short-entry threshold (a positive number, in scaled units).

    Returns:
        Plotly Figure.
    """
    # Define color palette.
    palette = px.colors.qualitative.Plotly
    ou_color = palette[0]  # Color for the scaled OU deviation line.
    sell_color = palette[1]
    buy_color = palette[2]
    zscore_color = palette[3]  # Color for the z-score line.

    # Compute the scaled OU deviation: x_scaled = scale * (spread_series - ou_mu)
    x_scaled = mar.scale * (mar.spread_series - mar.ou_mu)

    # Compute the 20-day rolling z-score of the spread.
    rolling_mean = mar.spread_series.rolling(window=20, min_periods=20).mean()
    rolling_std = mar.spread_series.rolling(window=20, min_periods=20).std()
    zscore = (mar.spread_series - rolling_mean) / rolling_std

    # Create the base figure.
    fig = go.Figure()

    # Scaled OU deviation line.
    fig.add_trace(
        go.Scatter(
            x=x_scaled.index,
            y=x_scaled.values,
            mode="lines",
            name="Scaled OU Deviation",
            line=dict(color=ou_color),
        )
    )

    # Z-score line.
    fig.add_trace(
        go.Scatter(
            x=zscore.index,
            y=zscore.values,
            mode="lines",
            name="20-Day Rolling Z-Score",
            line=dict(color=zscore_color, dash="dot"),
        )
    )

    # Add horizontal lines for stop_loss and take_profit thresholds.
    fig.add_trace(
        go.Scatter(
            x=[x_scaled.index[0], x_scaled.index[-1]],
            y=[stop_loss, stop_loss],
            mode="lines",
            name="Stop Loss",
            line=dict(color=buy_color, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x_scaled.index[0], x_scaled.index[-1]],
            y=[take_profit, take_profit],
            mode="lines",
            name="Take Profit",
            line=dict(color=sell_color, dash="dash"),
        )
    )

    # Extract buy and sell signals.
    buy_signals = signals[signals["Position"] == 1]
    sell_signals = signals[signals["Position"] == -1]

    # Plot buy signals using up triangles.
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=(
                mar.scale * (mar.spread_series.loc[buy_signals.index] - mar.ou_mu)
            ).values,
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color=buy_color),
            name="Buy Signal",
        )
    )
    # Plot sell signals using down triangles.
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=(
                mar.scale * (mar.spread_series.loc[sell_signals.index] - mar.ou_mu)
            ).values,
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color=sell_color),
            name="Sell Signal",
        )
    )

    fig.update_layout(
        title="Scaled OU Dynamics, Z-Score & Trading Signals",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
    )

    return fig
