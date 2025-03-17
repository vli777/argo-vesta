import plotly.express as px


def plot_bocpd_result(
    R,
    title="BOCPD Run-Length Probabilities",
    dates=None,
    run_length_range=None,
    regime_boundaries=None,
    regime_labels=None,
):
    """
    Plot the BOCPD run-length probability matrix using Plotly and overlay regime segments.

    Parameters:
      R: 2D numpy array (run-length probability matrix) with shape (T+1, T+1).
         Row 0 is the prior; rows 1...T correspond to each observation.
      title: Title of the plot.
      dates: Optional sequence (e.g., pd.Index) to use for the x-axis.
             If provided, the function will plot R[1:,1:] so that the first observation aligns with the first date.
      run_length_range: Optional list/array for the y-axis (run lengths). Defaults to range(R.shape[1]).
      regime_boundaries: Optional list of integer indices (0 to T) defining segment boundaries.
      regime_labels: Optional list of labels (strings) for each regime segment.
      
    Returns:
      fig: A Plotly Figure object.
    """
    T = R.shape[0] - 1  # T observations
    if dates is not None:
        # Use provided dates for the x-axis; drop the first row and first column of R
        time_range = list(dates)
        R_plot = R[1:, 1:]
        if run_length_range is None:
            run_length_range = list(range(1, R.shape[1]))
    else:
        time_range = list(range(R.shape[0]))
        R_plot = R
        if run_length_range is None:
            run_length_range = list(range(R.shape[1]))

    fig = px.imshow(
        R_plot,
        labels=dict(x="Time", y="Run Length", color="Probability"),
        x=time_range,
        y=run_length_range,
        aspect="auto",
        title=title,
    )
    fig.update_xaxes(side="bottom")
    fig.update_layout(
        coloraxis_colorbar=dict(title="Probability"),
        xaxis_title="Time",
        yaxis_title="Run Length",
    )

    # If regime segmentation info is provided, overlay the regime labels as vertical bands.
    if regime_boundaries is not None and regime_labels is not None:
        # Define colors for different regimes.
        regime_colors = {
            "Bullish": "rgba(0,255,0,0.2)",
            "Bearish": "rgba(255,0,0,0.2)",
            "Neutral": "rgba(128,128,128,0.2)",
        }
        # Loop over each regime segment. Note: regime_boundaries is expected to have length n+1,
        # and regime_labels to have length n.
        for i in range(len(regime_boundaries) - 1):
            start_idx = regime_boundaries[i]
            end_idx = regime_boundaries[i + 1]
            # Ensure indices are within the time_range length.
            if start_idx < len(time_range) and end_idx - 1 < len(time_range):
                x0 = time_range[start_idx]
                x1 = time_range[end_idx - 1]
            else:
                continue
            label = regime_labels[i]
            fillcolor = regime_colors.get(label, "rgba(128,128,128,0.2)")
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=fillcolor,
                opacity=0.3,
                line_width=0,
                annotation_text=label,
                annotation_position="top left",
            )
    return fig
