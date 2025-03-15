from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


from config import Config
from vis.utils import get_text_color, update_plot_layout
from utils.logger import logger


def cumulative_returns_plot(
    cumulative_returns: pd.DataFrame,
    color_map: Dict[str, str],
    paper_bgcolor: str,
    plot_bgcolor: str,
) -> None:
    # Determine default text color and annotation circle color based on the plot background.
    text_color = get_text_color(plot_bgcolor)
    subtext_color = "#a3a3a3"
    # Hover labels: translucent background remains as before.
    hover_bgcolor = (
        "rgba(0,0,0,0.7)" if text_color == "#f4f4f4" else "rgba(255,255,255,0.7)"
    )
    # Annotations: no background (transparent) and no border.
    annotation_bgcolor = "rgba(0,0,0,0)"
    annotation_text_color = text_color

    fig = go.Figure()
    all_dates = cumulative_returns.index
    cumulative_returns = cumulative_returns.reindex(index=all_dates, fill_value=np.nan)

    # Use Plotly's built-in "Plotly" palette.
    palette = px.colors.qualitative.Plotly
    sell_color = palette[1]  # red for negative change
    buy_color = palette[2]  # green for positive change

    # -----------------------------
    # Add traces with hover tooltip (no exact date shown)
    # -----------------------------
    for col in cumulative_returns.columns:
        col_data = cumulative_returns[col]
        delta = col_data.diff().fillna(0)
        delta_color = [
            buy_color if d > 0 else (sell_color if d < 0 else text_color) for d in delta
        ]
        customdata = np.column_stack([delta.values, delta_color])
        line_width = 6 if col == "SIM_PORT" else 1
        default_color = "gold" if col == "SIM_PORT" else "gray"
        trace_color = color_map.get(col, default_color)
        opacity = 1.0 if col == "SIM_PORT" else 0.5

        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=col_data,
                mode="lines",
                meta=col,
                name=col,
                line=dict(width=line_width, color=trace_color),
                customdata=customdata,
                hovertemplate=(
                    "%{meta}: %{y:.2%} "
                    "<span style='color:%{customdata[1]};'>%{customdata[0]:+.2%}</span>"
                    "<extra></extra>"
                ),
                hoverlabel=dict(
                    font=dict(size=12, color=text_color), bgcolor=hover_bgcolor
                ),
                opacity=opacity,
            )
        )

    update_plot_layout(
        fig,
        title="Cumulative Returns",
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
    )

    # -----------------------------
    # Add annotations for SIM_PORT only.
    # -----------------------------
    marker_x = []
    marker_y = []
    if "SIM_PORT" in cumulative_returns.columns:
        sim_data = cumulative_returns["SIM_PORT"]
        if pd.api.types.is_datetime64_any_dtype(sim_data.index):
            # Resample using 3-month frequency at month-end ("3ME") to get one tick per 3 months.
            resampled = sim_data.resample("3ME").last()
            major_ticks = list(resampled.index)
            # For each resampled date, find the nearest actual tick.
            actual_ticks = []
            for d in major_ticks:
                pos = sim_data.index.get_indexer([d], method="nearest")
                if pos[0] != -1:
                    actual_ticks.append(sim_data.index[pos[0]])
            major_ticks = sorted(list(set(actual_ticks)))
        else:
            major_ticks = sim_data.index

        # Skip the first tick so that change can be computed.
        if len(major_ticks) > 1:
            major_ticks = major_ticks[1:]
        else:
            major_ticks = []

        # Offsets: base arrow length and jitter; also include circle marker radius for padding.
        arrow_length = 50  # base arrow length in pixels
        jitter_amount = 50  # extra jitter for alternating annotations
        circle_radius = 16  # circle marker radius in pixels

        prev_cr = 0
        for i, tick in enumerate(major_ticks):
            try:
                cr = sim_data.loc[tick]
            except KeyError:
                continue
            if pd.isna(cr):
                continue

            # Format tick as "Mon YYYY" (end-of-month label).
            tick_str = (
                tick.strftime("%b %Y") if hasattr(tick, "strftime") else str(tick)
            )
            change = cr - prev_cr
            if change > 0:
                triangle_str = "▲"
                triangle_color = buy_color
            elif change < 0:
                triangle_str = "▼"
                triangle_color = sell_color
            else:
                triangle_str = ""
                triangle_color = text_color
            change_line = f"<br><span style='color:{triangle_color};'>{triangle_str} {change:+.2%}</span>"
            prev_cr = cr

            # Total offset: base arrow length plus circle_radius (for padding) plus optional jitter.
            total_offset = (
                arrow_length + circle_radius + (jitter_amount if i % 2 else 0)
            )

            # Build annotation text: first line is Month Year, second line CR, third line the change.
            annotation_text = f"{tick_str}<br>CR {cr:.2%}{change_line}"
            fig.add_annotation(
                x=tick,
                y=cr,
                ax=0,
                ay=-total_offset,  # arrow now starts padded by circle_radius
                xanchor="center",
                yanchor="bottom",
                text=annotation_text,
                showarrow=True,
                arrowhead=0,
                arrowcolor=annotation_text_color,
                arrowwidth=1,
                font=dict(size=10, color=annotation_text_color),
                align="center",
                bgcolor=annotation_bgcolor,
                borderwidth=0,
                borderpad=0,
            )
            marker_x.append(tick)
            marker_y.append(cr)

    # Add a scatter trace for the empty circle markers at the annotation points.
    if marker_x and marker_y:
        fig.add_trace(
            go.Scatter(
                x=marker_x,
                y=marker_y,
                mode="markers",
                name="",
                hoverinfo="skip",  # do not show hover for marker circles
                marker=dict(
                    symbol="circle",  # use a filled circle with transparent fill
                    size=circle_radius,
                    color="rgba(0,0,0,0)",  # transparent fill ensures only the outline is visible
                    line=dict(
                        width=2,
                        color=annotation_text_color,  # outline matches annotation text color
                    ),
                ),
                showlegend=False,
            )
        )

    # Set initial zoom to focus on SIM_PORT y-axis but include the full x-axis.
    if "SIM_PORT" in cumulative_returns.columns:
        sim_data = cumulative_returns["SIM_PORT"]
        y_min, y_max = sim_data.min() * 5, sim_data.max() * 5
        # Add a margin (10% of the range) to y-axis limits.
        y_margin = (y_max - y_min) * 0.1 if (y_max - y_min) != 0 else 0.1
        fig.update_yaxes(
            range=[y_min - y_margin, y_max + y_margin],
            showticklabels=False,
            color=text_color,
        )
        fig.update_xaxes(
            range=[all_dates.min(), all_dates.max()], color=text_color
        )  # Ensure x-axis labels match theme

    # Update layout for the title font, aligned left
    fig.update_layout(
        title=dict(
            text="Cumulative Returns",
            font=dict(
                family="Roboto, sans-serif",
                size=32,
                weight="bold",
                color=text_color,
            ),  # Title color
            x=0.02,  # Left padding
            y=0.98,
            xanchor="left",
        ),
        xaxis=dict(
            tickfont=dict(color=subtext_color),
        ),
        yaxis=dict(
            tickfont=dict(color=subtext_color),
        ),
        legend=dict(font=dict(color=text_color)),  # Ensure legend color matches theme
        hoverlabel=dict(font=dict(size=12, color=text_color), bgcolor=hover_bgcolor),
    )

    fig.show()
