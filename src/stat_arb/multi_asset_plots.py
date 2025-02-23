import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_multi_asset_signals(
    multi_asset_signals: pd.DataFrame,
    spread_series: pd.Series,
    title="Multi-Asset Mean Reversion Trading Signals",
):
    """
    Plots the spread series and overlays buy/sell signals using Plotly.

    Args:
        multi_asset_signals (pd.DataFrame): Multi-asset trading signals DataFrame with columns:
            - "Position" (1 for buy, -1 for sell, 0 for hold/neutral)
            - "Ticker" (Comma-separated list of tickers in the position)
            - "Entry Price" and "Exit Price" for trade visualization
        spread_series (pd.Series): Spread time series for the asset basket.
        price_series (pd.Series): Price time series for the asset basket.
        title (str): Plot title.

    Returns:
        None (displays an interactive Plotly plot).
    """
    fig = go.Figure()

    # Define color palette using Plotly's default palette
    palette = px.colors.qualitative.Plotly
    zscore_color = palette[0]  # Primary color for z-score spread
    sell_color = palette[1]
    buy_color = palette[2]

    # Add the z-score spread series as the primary y-axis
    fig.add_trace(
        go.Scatter(
            x=spread_series.index,
            y=spread_series.values,
            mode="lines",
            name="Z-Score Spread",
            line=dict(color=zscore_color, width=2),
            showlegend=True,
        )
    )

    # Extract buy/sell signals
    buy_signals = multi_asset_signals[multi_asset_signals["Position"] == 1]
    sell_signals = multi_asset_signals[multi_asset_signals["Position"] == -1]

    # Add Buy signals
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=spread_series.loc[buy_signals.index],
            mode="markers",
            name="Buy Signal",
            marker=dict(symbol="triangle-up", color=buy_color, size=10),
            text=buy_signals["Ticker"],
            hovertemplate="Buy Signal<br>%{text}<br>Z-Score Spread: %{y:.2f}",
            showlegend=True,
        )
    )

    # Add Sell signals
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=spread_series.loc[sell_signals.index],
            mode="markers",
            name="Sell Signal",
            marker=dict(symbol="triangle-down", color=sell_color, size=10),
            text=sell_signals["Ticker"],
            hovertemplate="Sell Signal<br>%{text}<br>Z-Score Spread: %{y:.2f}",
            showlegend=True,
        )
    )

    # Customize layout for dual y-axes
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Z-Score Spread",
        template="plotly_white",
        showlegend=True,
        hovermode="x unified",
        yaxis=dict(
            title="Z-Score Spread",
            showgrid=True,
            zeroline=True,
            tickformat=".2f",
        ),
        yaxis2=dict(
            title="",
            overlaying="y",
            side="right",
            showgrid=False,
            showticklabels=False,  # Hide price axis labels
        ),
    )

    fig.show()
