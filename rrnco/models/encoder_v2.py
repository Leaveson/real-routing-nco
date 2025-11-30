from typing import Tuple, Union

import torch
import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive import AutoregressiveEncoder
from tensordict import TensorDict
from torch import Tensor

from rrnco.models.env_embeddings import env_init_embedding
from rrnco.models.nn.attn_freenet_v2 import AttnFreeNetV2

class TrafficSimulator(nn.Module):
    """
    Simulates future traffic conditions to generate duration snapshots.
    Currently uses a heuristic approach based on scaling factors.
    """
    def __init__(self, num_snapshots: int = 3):
        super().__init__()
        self.num_snapshots = num_snapshots
        # Heuristic scaling factors for [Start, Peak, End]
        # e.g., 1.0 (base), 1.5 (peak traffic), 1.2 (calming down)
        # These could be learned parameters in a more advanced version.
        self.scaling_factors = nn.Parameter(torch.tensor([1.0, 1.5, 1.2]), requires_grad=True)

    def forward(self, duration_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            duration_matrix: (B, N, N) - Base duration matrix at t=0
        Returns:
            duration_snapshots: (B, K, N, N)
        """
        B, N, _ = duration_matrix.shape
        snapshots = []
        for k in range(self.num_snapshots):
            # Apply scaling factor
            # We use sigmoid * 2 + 0.5 to keep factors in reasonable range [0.5, 2.5]
            # or just use the parameter directly if we trust initialization.
            # Let's use simple multiplication for now.
            factor = self.scaling_factors[k]
            snapshots.append(duration_matrix * factor)
        
        return torch.stack(snapshots, dim=1)


class GlobalTrafficEncoder(AutoregressiveEncoder):
    """
    Encoder that incorporates global traffic context via Multi-Snapshot Neural Adaptive Bias.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = None,
        env_name: str = "rcvrp",
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        sdpa_fn=None,
        moe_kwargs: dict = None,
        use_coords: bool = False,
        use_polar_feats: bool = False,
        num_snapshots: int = 3,
    ):
        super(GlobalTrafficEncoder, self).__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name

        self.init_embedding = (
            env_init_embedding(
                env_name, {"embed_dim": embed_dim, **init_embedding_kwargs}
            )
            if init_embedding is None
            else init_embedding
        )
        
        self.traffic_simulator = TrafficSimulator(num_snapshots=num_snapshots)

        self.net = (
            AttnFreeNetV2(
                embed_dim=embed_dim,
                feedforward_hidden=feedforward_hidden,
                num_layers=num_layers,
                normalization=normalization,
                num_snapshots=num_snapshots,
            )
            if net is None
            else net
        )

    def forward(
        self, td: TensorDict, phase: str, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.
        """

        # Transfer to embedding space
        row_emb, col_emb, distance = self.init_embedding(td, phase)

        # Generate duration snapshots
        if self.env_name == "smtvrp":
            base_duration = td["duration_matrix"].type(torch.float32) / 1440
            duration_snapshots = self.traffic_simulator(base_duration)
            
            row_emb, col_emb = self.net(
                row_emb,
                col_emb,
                distance,
                td["locs"].type(torch.float32) / 1000,
                duration_snapshots,
            )
        else:
            # Fallback for other envs (though this encoder is designed for SMTVRP)
            row_emb, col_emb = self.net(
                row_emb, col_emb, distance, td["locs"].type(torch.float32)
            )

        return row_emb, col_emb
