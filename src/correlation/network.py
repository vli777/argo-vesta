import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community as nx_comm
import math

from correlation.correlation_utils import compute_correlation_matrix


def mst_community_detection(returns_df: pd.DataFrame) -> np.ndarray:
    """
    Cluster assets by constructing a Minimum Spanning Tree (MST) from the distance matrix
    (derived from the correlation matrix of returns) and then detecting communities using
    the greedy modularity algorithm.
    
    Args:
        returns_df (pd.DataFrame): DataFrame with dates as index and asset returns as columns.
    
    Returns:
        np.ndarray: Array of integer cluster labels (one per asset, in the order of returns_df.columns).
    """
    # Step 1: Compute the correlation matrix from returns.
    corr = compute_correlation_matrix(returns_df)
    
    # Step 2: Convert the correlation matrix into a distance matrix.
    dist = 1 - corr
    
    n = dist.shape[0]
    tickers = returns_df.columns.tolist()
    
    # Step 3: Build a complete weighted graph from the distance matrix.
    # networkx.from_numpy_array creates nodes [0, 1, ..., n-1]; we relabel them using tickers.
    G_complete = nx.from_numpy_array(dist)
    mapping = {i: tickers[i] for i in range(n)}
    G_complete = nx.relabel_nodes(G_complete, mapping)
    
    # Step 4: Compute the Minimum Spanning Tree (MST) of the complete graph.
    MST = nx.minimum_spanning_tree(G_complete, weight='weight')
    
    # Step 5: Run community detection on the MST.
    # Using the greedy modularity communities algorithm.
    communities = list(nx_comm.greedy_modularity_communities(MST, weight='weight'))
    
    # Create a dictionary mapping each ticker to its community label.
    labels = {}
    for i, comm in enumerate(communities):
        for node in comm:
            labels[node] = i
            
    # Step 6: Build an array of cluster labels in the order of returns_df.columns.
    cluster_labels = np.array([labels[ticker] for ticker in tickers])
    return cluster_labels

# Example usage:
if __name__ == "__main__":
    # Generate dummy returns data for 50 assets over 252 trading days.
    np.random.seed(42)
    dummy_returns = np.random.randn(252, 50)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    tickers = [f"Asset_{i}" for i in range(50)]
    returns_df = pd.DataFrame(dummy_returns, index=dates, columns=tickers)
    
    # Perform MST + community detection clustering.
    labels = mst_community_detection(returns_df)
    print("Cluster labels:", labels)
    
    # Optionally, you can inspect the number of clusters.
    n_clusters = len(np.unique(labels))
    print(f"Number of clusters detected: {n_clusters}")
