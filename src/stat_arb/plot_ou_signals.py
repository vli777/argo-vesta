import plotly.graph_objs as go
from plotly.subplots import make_subplots
import math


def plot_all_ticker_signals(
    price_data: dict,
    signal_data: dict,
    max_cols: int = 5,
    title: str = "Mean Reversion Trading Signals for All Tickers",
):
    """
    Plot all tickers with buy/sell signals on the same or multiple subplots.

    Args:
        price_data (dict): Dictionary of price Series per ticker, e.g., {"AAPL": price_series}.
        signal_data (dict): Dictionary of signals DataFrame per ticker, e.g., {"AAPL": signal_df}.
        max_cols (int, optional): Maximum number of columns in the subplot grid.
        title (str, optional): Title of the plot.
    """
    num_tickers = len(price_data)

    # Calculate grid size with a limit on maximum columns
    cols = min(max_cols, num_tickers)
    rows = math.ceil(num_tickers / cols)

    # Automatically adjust horizontal and vertical spacing based on grid size
    horizontal_spacing = min(0.05, 1 / (cols * 1.2))
    vertical_spacing = min(0.05, 1 / (rows * 1.2))

    # Create a subplot grid
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(price_data.keys()),
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )

    # Iterate over tickers and plot each one in its own subplot
    for i, (ticker, price_series) in enumerate(price_data.items()):
        row = (i // cols) + 1
        col = (i % cols) + 1

        signals = signal_data.get(ticker)
        if signals is None:
            continue

        # Convert signal positions to numeric: 1 for BUY, -1 for SELL, 0 for NO_SIGNAL
        buy_signals = signals[signals["Position"] == 1]
        sell_signals = signals[signals["Position"] == -1]

        # Align the price series with the signal dates
        aligned_price_series = price_series.reindex(signals.index, method="ffill")

        # Plot price line
        fig.add_trace(
            go.Scatter(
                x=aligned_price_series.index,
                y=aligned_price_series.values,
                mode="lines",
                name=f"{ticker} Price",
                line=dict(width=2, color="blue"),
                showlegend=(i == 0),  # Only show in the first plot to avoid repetition
            ),
            row=row,
            col=col,
        )

        # Plot buy signals
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=aligned_price_series.loc[buy_signals.index],
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(symbol="triangle-up", color="green", size=8),
                    hovertemplate=f"{ticker} Buy<br>Price: %{{y:.2f}}<br>Date: %{{x}}",
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

        # Plot sell signals
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=aligned_price_series.loc[sell_signals.index],
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(symbol="triangle-down", color="red", size=8),
                    hovertemplate=f"{ticker} Sell<br>Price: %{{y:.2f}}<br>Date: %{{x}}",
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

    # Update layout
    fig.update_layout(
        title=title,
        height=160 * rows,
        width=420 * cols,
        showlegend=True,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.5)"),
    )

    fig.show()
