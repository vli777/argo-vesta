import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def plot_global_optimization(
    search_history,
    final_solution,
    objective_function,
    grid_resolution=50,
    title="Global Optimization Contour",
    flip_objective=False,
):
    """
    Create a 3D contour/surface plot of the global optimization manifold for the whole portfolio.

    Parameters:
        search_history: np.array, shape (n_iterations, n_assets)
            Recorded weight vectors from the global optimization search.
        final_solution: np.array, shape (n_assets,)
            The final optimum weight vector obtained from the global optimizer.
        objective_function: callable
            A function that takes a weight vector (1D array of length n_assets) and returns a scalar objective value.
        grid_resolution: int, optional
            Resolution of the grid in the reduced (2D) space.
        title: str, optional
            Title for the plot.
        flip_objective: bool, optional
            If True, flip the objective (multiply by -1) so that the optimum appears as a maximum.

    Returns:
        fig: Plotly Figure object showing the 3D contour (surface) plot with the optimum marked.
    """
    # Define the effective objective function based on flip_objective flag.
    effective_obj = (
        (lambda w: -objective_function(w)) if flip_objective else objective_function
    )

    # Use PCA to reduce the high-dimensional search history to 2 dimensions.
    pca = PCA(n_components=2)
    projected_history = pca.fit_transform(search_history)
    projected_final = pca.transform(final_solution.reshape(1, -1))[0]

    # Define grid boundaries based on the projected search history.
    x_min, x_max = projected_history[:, 0].min(), projected_history[:, 0].max()
    y_min, y_max = projected_history[:, 1].min(), projected_history[:, 1].max()

    # Create a grid in the 2D PCA space.
    grid_x, grid_y = np.mgrid[
        x_min : x_max : grid_resolution * 1j, y_min : y_max : grid_resolution * 1j
    ]
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Inverse-transform grid points back to the original space.
    original_points = pca.inverse_transform(grid_points)

    # Evaluate the effective objective function on the grid.
    Z = np.array([effective_obj(pt) for pt in original_points])
    Z = Z.reshape(grid_x.shape)

    # Create the 3D surface plot.
    fig = go.Figure(
        data=[
            go.Surface(
                x=grid_x,
                y=grid_y,
                z=Z,
                colorscale="Viridis",
                opacity=0.85,
                showscale=True,
                colorbar=dict(title="Objective"),
            )
        ]
    )

    # Add a marker for the final optimum.
    final_obj_value = effective_obj(final_solution)
    fig.add_trace(
        go.Scatter3d(
            x=[projected_final[0]],
            y=[projected_final[1]],
            z=[final_obj_value],
            mode="markers",
            marker=dict(color="red", size=8, symbol="x"),
            name="Final Optimum",
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Weights PC1",
            yaxis_title="Weights PC2",
            zaxis_title="Objective Value",
        ),
    )

    return fig
