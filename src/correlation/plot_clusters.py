import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
import umap


def visualize_clusters_umap(
    returns_df: pd.DataFrame,
    cluster_labels,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    title: str = "UMAP Visualization of Asset Clusters",
):
    """
    Visualize asset clusters using UMAP and Plotly, displaying the ticker above each point.

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and assets as columns.
        cluster_labels (array-like): Cluster labels for each asset (in the same order as returns_df.columns).
        n_neighbors (int): UMAP parameter that controls the local neighborhood size.
        min_dist (float): UMAP parameter that controls how tightly UMAP packs points together.
        metric (str): The distance metric to use.
        title (str): Plot title.
    """
    # Transpose the DataFrame so that each asset is represented as a feature vector.
    asset_data = returns_df.T
    n_samples = asset_data.shape[0]

    # Ensure n_neighbors is always valid
    n_neighbors = min(n_neighbors, max(n_samples - 1, 1))

    # Create the UMAP reducer and transform the data.
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42
    )
    umap_results = reducer.fit_transform(asset_data)

    # Create a DataFrame for plotting.
    umap_df = pd.DataFrame(umap_results, columns=["UMAP1", "UMAP2"])
    umap_df["Ticker"] = asset_data.index
    umap_df["Cluster"] = cluster_labels

    # Create an interactive scatter plot with tickers shown above points.
    fig = px.scatter(
        umap_df,
        x="UMAP1",
        y="UMAP2",
        color="Cluster",
        text="Ticker",
        hover_data=["Ticker"],
        title=title,
    )
    # Set the text position and font size to help reduce overlap.
    fig.update_traces(textposition="top center", textfont=dict(size=10))
    fig.show()


def visualize_clusters_tsne(
    returns_df: pd.DataFrame,
    cluster_labels,
    perplexity: float = 30,
    max_iter: int = 1000,
    title: str = "t-SNE Visualization of Asset Clusters",
):
    """
    Visualize asset clusters using t-SNE and Plotly, displaying the ticker above each point.

    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and assets as columns.
        cluster_labels (array-like): Cluster labels for each asset (in the same order as returns_df.columns).
        perplexity (int): t-SNE perplexity parameter.
        max_iter (int): Number of iterations for t-SNE.
        title (str): Plot title.
    """
    # Transpose the DataFrame so that each asset is represented as a feature vector.
    asset_data = returns_df.T

    # Optionally, you can standardize the data here if needed.
    tsne = TSNE(
        perplexity=min(perplexity, len(asset_data) - 1),
        max_iter=max_iter,
        random_state=42,
    )
    tsne_results = tsne.fit_transform(asset_data)

    # Create a DataFrame for plotting.
    tsne_df = pd.DataFrame(tsne_results, columns=["TSNE1", "TSNE2"])
    tsne_df["Ticker"] = asset_data.index
    tsne_df["Cluster"] = cluster_labels

    # Create an interactive scatter plot with tickers shown above points.
    fig = px.scatter(
        tsne_df,
        x="TSNE1",
        y="TSNE2",
        color="Cluster",
        text="Ticker",
        hover_data=["Ticker"],
        title=title,
    )
    # Set the text position and font size.
    fig.update_traces(textposition="top center", textfont=dict(size=10))
    fig.show()
