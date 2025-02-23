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


def plot_multi_ou_signals(mar, signals, stop_loss, take_profit):
    """
    Create a Plotly figure plotting the OU deviation, the threshold levels,
    and the buy/sell signals.

    Parameters:
        mar: MultiAssetReversion instance containing spread_series and ou_mu.
        signals: DataFrame with datetime index and a "Position" column.
        stop_loss: The long-entry threshold (a negative number).
        take_profit: The short-entry threshold (a positive number).

    Returns:
        Plotly Figure.
    """
    # Define color palette
    palette = px.colors.qualitative.Plotly
    zscore_color = palette[0]  # Primary color for the OU deviation line
    sell_color = palette[1]
    buy_color = palette[2]

    # Compute the OU deviation as: deviation = spread_series - ou_mu.
    ou_deviation = mar.spread_series - mar.ou_mu

    # Create the base figure with the OU deviation line.
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ou_deviation.index,
        y=ou_deviation.values,
        mode='lines',
        name='OU Deviation',
        line=dict(color=zscore_color)
    ))

    # Add horizontal lines for stop_loss and take_profit thresholds.
    fig.add_trace(go.Scatter(
        x=[ou_deviation.index[0], ou_deviation.index[-1]],
        y=[stop_loss, stop_loss],
        mode='lines',
        name='Stop Loss',
        line=dict(color=buy_color, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=[ou_deviation.index[0], ou_deviation.index[-1]],
        y=[take_profit, take_profit],
        mode='lines',
        name='Take Profit',
        line=dict(color=sell_color, dash='dash')
    ))

    # Extract buy and sell signals.
    buy_signals = signals[signals["Position"] == 1]
    sell_signals = signals[signals["Position"] == -1]

    # Plot buy signals using up triangles.
    fig.add_trace(go.Scatter(
        x=buy_signals.index,
        y=(mar.spread_series.loc[buy_signals.index] - mar.ou_mu).values,
        mode='markers',
        marker=dict(symbol='triangle-up', size=10, color=buy_color),
        name='Buy Signal'
    ))
    # Plot sell signals using down triangles.
    fig.add_trace(go.Scatter(
        x=sell_signals.index,
        y=(mar.spread_series.loc[sell_signals.index] - mar.ou_mu).values,
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color=sell_color),
        name='Sell Signal'
    ))

    fig.update_layout(
        title="OU Dynamics: Spread Deviation & Trading Signals",
        xaxis_title="Time",
        yaxis_title="Deviation from OU Mean",
        template="plotly_white"
    )
    
    return fig
