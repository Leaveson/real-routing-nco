import torch
import torch.nn as nn


class RVRPInitEmbedding(nn.Module):
    """Initial embedding for the Vehicle Routing Problems (VRP).

    Embeds node features into a shared embedding space:
    - locs: x, y coordinates (for depot and customers)
    - demand: customer demands
    - distance: pairwise distances between nodes
    """

    def __init__(
        self,
        embed_dim,
        linear_bias=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Initialize embeddings based on configurations
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)
        self.init_embed = nn.Linear(3, embed_dim, linear_bias)

    def forward(self, td, phase):
        locs = td["locs"].float()
        demand = td["demand"].float()
        distance = td["distance_matrix"]
        return self._embed_without_distance(locs, demand, distance)

    def _embed_without_distance(self, locs, demand, distance):
        depot, cities = locs[:, :1, :], locs[:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        cities_feats = torch.cat([cities, demand[..., None]], dim=-1)

        node_embeddings = self.init_embed(cities_feats)
        out = torch.cat([depot_embedding, node_embeddings], dim=-2)

        return out, distance
