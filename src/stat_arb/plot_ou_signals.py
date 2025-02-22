import math
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_all_ticker_signals(
    price_data: dict,
    signal_data: dict,
    max_cols: int = 4,
    title: str = "Mean Reversion Trading Signals with Dual Axes and Rolling Z-Score",
):
    """
    Plot all tickers with buy/sell signals and dual y-axes:
      - Primary axis: Price series.
      - Secondary axis: 20-day rolling z-score of log prices.
    Also plots solid boundary lines for ±1σ and ±2σ with labels.

    Args:
        price_data (dict): Dictionary of price Series per ticker.
        signal_data (dict): Dictionary of signals DataFrame per ticker.
        max_cols (int, optional): Maximum number of columns in the subplot grid.
        title (str, optional): Plot title.
    """
    # Define updated color palette
    palette = {
        "price": "#00ADB5",  # teal
        "zscore": "#EEEEEE",  # light gray
        "1sd_line": "#C9D6DF",  # ±1σ med gray
        "2sd_line": "#222831",  # ±2σ dark gray
    }

    num_tickers = len(price_data)
    cols = min(max_cols, num_tickers)
    rows = math.ceil(num_tickers / cols)

    horizontal_spacing = 0.05
    vertical_spacing = 0.002

    # Create subplot grid with dual y-axes.
    specs = [[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(price_data.keys()),
        specs=specs,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )

    for i, (ticker, price_series) in enumerate(price_data.items()):
        row = (i // cols) + 1
        col = (i % cols) + 1

        signals = signal_data.get(ticker)
        if signals is None:
            continue

        buy_signals = signals[signals["Position"] == 1]
        sell_signals = signals[signals["Position"] == -1]

        # Align the price series with the signals index.
        aligned_price_series = price_series.reindex(signals.index, method="ffill")

        # Compute rolling z-score from log prices using a 20-day window.
        log_price_series = np.log(aligned_price_series)
        rolling_mean = log_price_series.rolling(window=20, min_periods=1).mean()
        rolling_std = (
            log_price_series.rolling(window=20, min_periods=1).std().replace(0, 1e-6)
        )
        z_series = (log_price_series - rolling_mean) / rolling_std

        # --- Plot Buy Signals (on top)
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=aligned_price_series.loc[buy_signals.index],
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(symbol="triangle-up", color="#00cc33", size=10),
                    hovertemplate="%{y}",
                    showlegend=(i == 0),
                    zorder=99,
                ),
                row=row,
                col=col,
                secondary_y=False,
            )

        # --- Plot Sell Signals (on top)
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=aligned_price_series.loc[sell_signals.index],
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(symbol="triangle-down", color="red", size=10),
                    hovertemplate="%{y}",
                    showlegend=(i == 0),
                    zorder=99,
                ),
                row=row,
                col=col,
                secondary_y=False,                
            )

        # --- Primary Axis: Plot Price Series (below signals)
        fig.add_trace(
            go.Scatter(
                x=aligned_price_series.index,
                y=aligned_price_series.values,
                mode="lines",
                line=dict(width=2, color=palette["price"], shape="spline"),
                name=f"{ticker}",
                showlegend=(i == 0),
            ),
            row=row,
            col=col,
            secondary_y=False,
        )

        # --- Secondary Axis: Plot Rolling Z-Score (light gray solid line below price)
        fig.add_trace(
            go.Scatter(
                x=z_series.index,
                y=z_series.values,
                mode="lines",
                line=dict(width=1.5, color=palette["zscore"], shape="spline"),
                name="Z-Score",
                showlegend=(i == 0),
                opacity=0.6,
            ),
            row=row,
            col=col,
            secondary_y=True,
        )

        # --- Plot ±1σ and ±2σ as solid lines (bottom layer)
        for sigma, color, label in [(1, "1sd_line", "±1σ"), (2, "2sd_line", "±2σ")]:
            fig.add_trace(
                go.Scatter(
                    x=z_series.index,
                    y=[sigma] * len(z_series),
                    mode="lines",
                    line=dict(color=palette[color], width=1.5, dash="solid"),
                    name=label,
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
                secondary_y=True,
            )
            fig.add_trace(
                go.Scatter(
                    x=z_series.index,
                    y=[-sigma] * len(z_series),
                    mode="lines",
                    line=dict(color=palette[color], width=1.5, dash="solid"),
                    name=label,
                    showlegend=False,
                ),
                row=row,
                col=col,
                secondary_y=True,
            )

    fig.update_layout(
        title=title,
        height=500 * rows,
        width=500 * cols,
        showlegend=True,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.5)"),
    )
    fig.update_yaxes(automargin=True, showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False, zeroline=False)

    fig.show()
