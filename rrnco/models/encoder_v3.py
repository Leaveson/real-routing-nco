import math
from typing import Tuple, Union

import torch
import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive import AutoregressiveEncoder
from tensordict import TensorDict
from torch import Tensor

from rrnco.models.env_embeddings import env_init_embedding
from rrnco.models.nn.attn_freenet_v2 import AttnFreeNetV2


class TrafficSimulatorV3(nn.Module):
    """
    Simulates future traffic conditions using a physics-informed approach based on the SMTVRP environment logic.
    It learns 'time anchors' to generate expected duration snapshots at critical times.
    """
    def __init__(self, num_snapshots: int = 3):
        super().__init__()
        self.num_snapshots = num_snapshots
        
        # Learnable time anchors.
        # Initialize them spread out through the day (0 to 1440 minutes).
        # e.g., 8:00 (480), 12:00 (720), 17:00 (1020)
        # We initialize them somewhat reasonably but let them float.
        self.time_anchors = nn.Parameter(
            torch.tensor([480.0, 720.0, 1020.0]), # Start with Morning Peak, Noon, Evening Peak
            requires_grad=True
        )

    def _normal_dist_torch(self, x, mean, std):
        return torch.exp(-((x - mean) ** 2) / (2 * std ** 2)) / (std * math.sqrt(2 * math.pi))

    def forward(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distance_matrix: (B, N, N) - Distance matrix (assumed to be unnormalized or we need to know scale)
                             In SMTVRP, distance is roughly coordinate distance * 100? 
                             Wait, in env.py: "dist_fac = 1 - torch.exp(-distance_matrix / 50.0)"
                             The input distance_matrix to encoder is usually normalized?
                             In `RRNetEncoder.forward`: `distance` is returned by `init_embedding`.
                             But we also have `td["distance_matrix"]`.
                             In `GlobalTrafficEncoder.forward` (v2), we passed `td["duration_matrix"]`.
                             Here we need `distance_matrix` to compute the delay.
                             We should use the raw distance matrix from TD to match env logic.
        Returns:
            duration_snapshots: (B, K, N, N)
        """
        B, N, _ = distance_matrix.shape
        snapshots = []
        
        # Ensure time anchors are within reasonable bounds (optional, but good for stability)
        # times = torch.clamp(self.time_anchors, 0, 1440) 
        # Actually, let's not clamp to allow learning, but maybe sigmoid if we want strict range.
        # For now, just use raw values.
        times = self.time_anchors

        for k in range(self.num_snapshots):
            t = times[k] # Scalar (or learnable parameter)
            
            # 1. Time Factor (Gaussian Peaks)
            # t is scalar, broadcast later
            morning_peak = self._normal_dist_torch(t, 480, 90)
            evening_peak = self._normal_dist_torch(t, 1020, 90)
            
            # rush_hour_effect for LogNormal params
            rush_hour_effect = morning_peak + evening_peak
            
            # time_fac for Base Delay
            time_fac = 0.5 + 2 * rush_hour_effect
            
            # 2. Distance Factor
            # dist_fac = 1 - exp(-distance / 50)
            # We assume distance_matrix is in the same scale as env (0-100 range usually for VRP 0-1 coords?)
            # SMTVRP coords are 0-100?
            # Generator: coords sampled from [0, 100].
            # So distance is in [0, 141].
            dist_fac = 1 - torch.exp(-distance_matrix / 50.0)
            
            # 3. Base Delay
            # base_delay = 0.25 * time_fac * dist_fac
            # time_fac is scalar (for this snapshot)
            base_delay = 0.25 * time_fac * dist_fac
            
            # 4. Expected Random Factor
            # mu = 0 + 0.1 * rush_hour_effect
            # sigma = 0.3 + 0.2 * rush_hour_effect
            mu = 0 + 0.1 * rush_hour_effect
            sigma = 0.3 + 0.2 * rush_hour_effect
            
            # E[LogNormal] = exp(mu + sigma^2 / 2)
            expected_rand_factor = torch.exp(mu + (sigma ** 2) / 2)
            
            expected_delay = base_delay * expected_rand_factor
            
            # 5. Expected Accident Delay
            # acc_rate = 0.05 * normal(t, 1260, 120)
            acc_rate = 0.05 * self._normal_dist_torch(t, 1260, 120)
            acc_rate = torch.clamp(acc_rate, min=0)
            
            # E[Poisson(lambda)] = lambda
            # accident_delay = num_accidents * 75.0
            expected_accident_delay = acc_rate * 75.0
            
            # Total Expected Duration
            expected_duration = distance_matrix + expected_delay + expected_accident_delay
            
            snapshots.append(expected_duration)
        
        return torch.stack(snapshots, dim=1) # (B, K, N, N)


class GlobalTrafficEncoderV3(AutoregressiveEncoder):
    """
    Encoder that incorporates global traffic context via Multi-Snapshot Neural Adaptive Bias,
    using a Physics-Informed Traffic Simulator (V3).
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
        super(GlobalTrafficEncoderV3, self).__init__()

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
        
        self.traffic_simulator = TrafficSimulatorV3(num_snapshots=num_snapshots)

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
            # We need the raw distance matrix for the simulator logic
            # td["distance_matrix"] is likely the one we want.
            # Note: We normalize the output passed to the net, but simulator uses raw values for physics.
            raw_distance = td["distance_matrix"].type(torch.float32)
            
            duration_snapshots = self.traffic_simulator(raw_distance)
            
            # Normalize snapshots for the network (divide by 1440 as is standard in this codebase for duration)
            duration_snapshots_norm = duration_snapshots / 1440.0
            
            row_emb, col_emb = self.net(
                row_emb,
                col_emb,
                distance,
                td["locs"].type(torch.float32) / 1000,
                duration_snapshots_norm,
            )
        else:
            # Fallback
            row_emb, col_emb = self.net(
                row_emb, col_emb, distance, td["locs"].type(torch.float32)
            )

        return row_emb, col_emb
