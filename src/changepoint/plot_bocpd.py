import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vis.utils import hex_to_rgba


def plot_bocpd_result(
    R,
    feature_series=None,
    series_title="Feature Series",
    series_label="Feature Value",
    title="BOCPD Run-Length Probabilities and Feature Series",
    dates=None,
    run_length_range=None,
    regime_boundaries=None,
    regime_labels=None,
):
    """
    Plot the BOCPD run-length probability matrix along with the input feature series.
    The figure has two subplots:
      - Top: The feature series over time.
      - Bottom: The run-length probability heatmap using the "sunset" color scale.
    Regime segments labeled "Bullish" or "Bearish" are overlaid as vertical rectangles.
    Neutral regimes are not labeled.

    Note: The diagonal line in the heatmap is expected because in BOCPD,
          if no change point is detected the run length increases by one at every time step,
          forming a diagonal; a reset (change point) breaks this pattern.

    Parameters:
      R: 2D numpy array (run-length probability matrix) with shape (T+1, T+1).
         Row 0 is the prior; rows 1...T correspond to each observation.
      feature_series: Optional pandas Series for the input feature (e.g., rolling mean returns).
      series_title: Title for the series subplot figure.
      series_label: Y-axis label for the series values.
      title: Title for the figure.
      dates: Optional sequence (e.g., pd.Index) to use for the x-axis.
      run_length_range: Optional list/array for the y-axis (run lengths). Defaults to range(R.shape[1]).
      regime_boundaries: Optional list of integer indices defining segment boundaries.
      regime_labels: Optional list of labels (strings) for each regime segment.

    Returns:
      fig: A Plotly Figure object.
    """
    # Determine the x-axis values for both subplots.
    if dates is not None:
        x_vals = list(dates)
    elif feature_series is not None:
        x_vals = list(feature_series.index)
    else:
        x_vals = list(range(R.shape[0]))

    # Prepare the heatmap data: drop the first row and column.
    R_plot = R[1:, 1:]
    if run_length_range is None:
        run_length_range = list(range(1, R.shape[1]))

    # Create a subplot figure with 2 rows.
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.3, 0.7],
        vertical_spacing=0.05,
        subplot_titles=(series_title, "Run-Length Probability Heatmap"),
    )

    # Top subplot: add the feature series if provided.
    if feature_series is not None:
        # Use x_vals computed above.
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=feature_series.values, mode="lines", name="Feature Series"
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text=series_label, row=1, col=1)

    # Bottom subplot: add the heatmap.
    heatmap = go.Heatmap(
        z=R_plot,
        x=x_vals,
        y=run_length_range,
        colorscale=px.colors.sequential.Sunset,
        colorbar=dict(title="Probability"),
    )
    fig.add_trace(heatmap, row=2, col=1)
    fig.update_yaxes(title_text="Run Length", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    # Update overall layout for a light background.
    fig.update_layout(
        template="plotly_white",
        title=title,
    )

    # Use Plotly's qualitative palette for regime colors.
    palette = px.colors.qualitative.Plotly
    bearish_color = palette[1]  # typically red
    bullish_color = palette[2]  # typically green
    regime_colors = {
        "Bullish": hex_to_rgba(bullish_color),
        "Bearish": hex_to_rgba(bearish_color),
    }

    # Overlay regime segmentation as vertical rectangles (only for Bullish and Bearish).
    if regime_boundaries is not None and regime_labels is not None:
        for i in range(len(regime_boundaries) - 1):
            start_idx = regime_boundaries[i]
            end_idx = regime_boundaries[i + 1]
            if start_idx < len(x_vals) and (end_idx - 1) < len(x_vals):
                x0 = x_vals[start_idx]
                x1 = x_vals[end_idx - 1]
            else:
                continue
            label = regime_labels[i]
            if label in ["Bullish", "Bearish"]:
                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=regime_colors[label],
                    opacity=0.7,
                    line_width=0,
                    layer="below",
                )
            # For neutral regimes, skip adding any vertical rectangle.

    return fig
