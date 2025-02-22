import pandas as pd
import plotly.graph_objects as go


def plot_multi_asset_signals(
    multi_asset_signals: pd.DataFrame,
    price_series: pd.Series,
    title="Multi-Asset Mean Reversion Trading Signals",
):
    """
    Plots the aggregate price series and overlays buy/sell signals using Plotly.

    Args:
        multi_asset_signals (pd.DataFrame): Multi-asset trading signals DataFrame with columns:
            - "Position" (1 for buy, -1 for sell, 0 for hold/neutral)
            - "Ticker" (Comma-separated list of tickers in the position)
            - "Entry Price" and "Exit Price" for trade visualization
        price_series (pd.Series): Aggregated price time series for the asset basket.
        title (str): Plot title.

    Returns:
        None (displays an interactive Plotly plot).
    """
    fig = go.Figure()

    # Add the aggregated price series
    fig.add_trace(
        go.Scatter(
            x=price_series.index,
            y=price_series.values,
            mode="lines",
            name="Basket Price",
            line=dict(color="blue"),
        )
    )

    # Extract buy/sell signals
    buy_signals = multi_asset_signals[multi_asset_signals["Position"] == 1]
    sell_signals = multi_asset_signals[multi_asset_signals["Position"] == -1]

    # Add Buy signals
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=price_series.loc[buy_signals.index],
            mode="markers",
            name="Buy Signal",
            marker=dict(symbol="triangle-up", color="green", size=10),
            text=buy_signals["Ticker"],  # Show involved tickers on hover
            hovertemplate="Buy Signal<br>%{text}<br>Price: %{y:.2f}",
        )
    )

    # Add Sell signals
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=price_series.loc[sell_signals.index],
            mode="markers",
            name="Sell Signal",
            marker=dict(symbol="triangle-down", color="red", size=10),
            text=sell_signals["Ticker"],  # Show involved tickers on hover
            hovertemplate="Sell Signal<br>%{text}<br>Price: %{y:.2f}",
        )
    )

    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
    )

    fig.show()
