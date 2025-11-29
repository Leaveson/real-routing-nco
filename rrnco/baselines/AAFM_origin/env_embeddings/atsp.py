import torch
import torch.nn as nn


class ATSPInitEmbedding(nn.Module):
    """Initial embedding for the Asymmetric TSP (ATSP).
    Embeds the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - distance: distance between the nodes
    """

    def __init__(
        self,
        embed_dim,
        linear_bias=True,
    ):
        super(ATSPInitEmbedding, self).__init__()
        
        self.init_embed = nn.Linear(2, embed_dim, linear_bias)
        
    def forward(self, td, phase):
        locs, distance = td["locs"].float(), td["distance_matrix"]
        B, N, _ = locs.shape
        node_embeddings = self.init_embed(locs)  # [B, N, embed_dim]

        return node_embeddings.clone(), distance