import torch
import torch.nn as nn


class RVRPTWInitEmbedding(nn.Module):
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
    
        self.init_embed_depot = nn.Linear(6, embed_dim, linear_bias)
        self.init_embed = nn.Linear(7, embed_dim, linear_bias)
    
    def forward(self, td, phase):
        locs = td["locs"].float()
        demand = td["demand_linehaul"]
        time_windows = td["time_windows"]
        service_time = td["service_time"]
        vrp_attr = torch.cat(
            [demand.unsqueeze(-1), time_windows, service_time.unsqueeze(-1)], dim=-1
        )
        depot, cities = locs[:, :1, :], locs[:, 1:, :]
        depot_embedding = self.init_embed_depot(torch.cat([depot, vrp_attr[:,0:1]], dim=-1))
        cities_feats = torch.cat([cities, vrp_attr[:,1:]], dim=-1)
        node_embeddings = self.init_embed(cities_feats)
        out = torch.cat([depot_embedding, node_embeddings], dim=-2)

        return out, td["distance"]