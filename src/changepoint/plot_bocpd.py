import plotly.express as px


def plot_bocpd_result(
    R, title="BOCPD Run-Length Probabilities", dates=None, run_length_range=None
):
    """
    Plot the BOCPD run-length probability matrix using Plotly.

    Parameters:
      R: 2D numpy array (run-length probability matrix) with shape (T+1, T+1).
         Row 0 is the prior; rows 1...T correspond to each observation.
      title: Title of the plot.
      dates: Optional sequence (e.g., pd.Index) to use for the x-axis.
             If provided, the function will plot R[1:,:] so that the first observation aligns with the first date.
      run_length_range: Optional list/array for the y-axis (run lengths). Defaults to range(T+1).

    Returns:
      fig: A Plotly Figure object.
    """
    T = R.shape[0] - 1  # T observations
    if dates is not None:
        # Use provided dates for the x-axis; drop the first row of R (the prior) so that the dimensions match.
        time_range = list(dates)
        R_plot = R[1:, :]
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
    return fig
