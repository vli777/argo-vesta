import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import coint


def construct_graph(
    log_prices: pd.DataFrame, returns: pd.DataFrame, p_value_threshold: float = 0.05
) -> Data:
    """
    Constructs a graph where nodes represent assets and edges represent cointegration
    relationships based on pairwise cointegration tests.

    Args:
        log_prices (pd.DataFrame): DataFrame of log-prices with dates as index and assets as columns.
        returns (pd.DataFrame): DataFrame of asset returns with dates as index and assets as columns.
        p_value_threshold (float): Threshold to decide if two assets are cointegrated.

    Returns:
        Data: A PyTorch Geometric Data object containing node features (asset return series),
              edge_index (connectivity), and edge_attr (test statistics as weights).
    """
    num_assets = log_prices.shape[1]
    edges = []
    edge_weights = []

    # Iterate over all unique asset pairs (O(n^2); consider parallelizing for large n)
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            score, p_value, _ = coint(log_prices.iloc[:, i], log_prices.iloc[:, j])
            if p_value < p_value_threshold:
                # Add both directions to create an undirected graph
                edges.append([i, j])
                edges.append([j, i])
                edge_weights.append(score)
                edge_weights.append(score)

    if not edges:
        raise ValueError(
            "No cointegration relationships found with the given p-value threshold."
        )

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)

    # Use the returns time series as node features.
    # Each asset's feature vector is its full return history.
    node_features = torch.tensor(returns.T.values, dtype=torch.float)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        """
        Graph Convolutional Encoder for a Graph Autoencoder.

        Args:
            in_channels (int): Dimension of input node features.
            hidden_channels (int): Dimension of the latent embedding space.
        """
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels * 2)
        self.conv2 = GCNConv(hidden_channels * 2, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def train_gae(
    data: Data,
    hidden_channels: int = 32,
    num_epochs: int = 200,
    learning_rate: float = 0.01,
    patience: int = 20,
    min_delta: float = 1e-8,
    device: str = None,
) -> GAE:
    """
    Trains a Graph Autoencoder (GAE) with early stopping if the loss converges.

    Args:
        data (Data): PyTorch Geometric Data object containing the graph.
        hidden_channels (int): Dimension of the latent embeddings.
        num_epochs (int): Maximum number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Number of epochs with no improvement to wait before stopping.
        min_delta (float): Minimum change in loss to qualify as an improvement.
        device (str): Computation device ('cpu' or 'cuda'). Defaults to 'cuda' if available.

    Returns:
        GAE: The trained Graph Autoencoder model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = GCNEncoder(data.num_features, hidden_channels)
    model = GAE(encoder).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early Stopping Parameters
    best_loss = float("inf")
    epochs_no_improve = 0

    model.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")

        # Early Stopping Check
        if loss.item() < best_loss - min_delta:
            best_loss = loss.item()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} with loss {best_loss:.4f}")
            break

    return model


def cluster_embeddings(
    model: GAE, data: Data, num_clusters: int = 10, device: str = None
) -> dict:
    """
    Extracts latent node embeddings from the trained GAE model and clusters them using KMeans.
    Each cluster can be interpreted as a basket of cointegrated assets.

    Args:
        model (GAE): Trained Graph Autoencoder.
        data (Data): Graph data used for training.
        num_clusters (int): Number of clusters (baskets) for KMeans.
        device (str): Computation device.

    Returns:
        dict: Mapping of asset names (or indices) to cluster labels.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index).cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # Generate asset labels; modify if you have custom asset names.
    asset_names = [f"Asset_{i}" for i in range(data.num_nodes)]
    return {asset_names[i]: int(clusters[i]) for i in range(data.num_nodes)}
