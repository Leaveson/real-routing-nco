from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.utils.pylogger import get_pylogger
from rrnco.models.nn.attn_freenet import AttnFree_Block, Attn_Free_Layer, AttnFreeNet, Normalization, TransformerFFN, AFTFull

log = get_pylogger(__name__)

class MultiSnapshotNAB(nn.Module):
    """
    Multi-Snapshot Neural Adaptive Bias.
    Fuses multiple duration snapshots (e.g., Start, Peak, End) to provide global temporal context.
    """

    def __init__(self, embed_dim: int, num_snapshots: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_snapshots = num_snapshots

        # Input: distance + angle + num_snapshots * duration
        # We assume distance and angle are always present.
        input_dim = embed_dim * (2 + num_snapshots) 

        self.dist_emb = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.angle_emb = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Shared embedding layer for duration snapshots
        self.dur_emb = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 2 + num_snapshots), # Weights for dist, angle, and each snapshot
        )
        self.gate_temperature = nn.Parameter(torch.tensor(5.0))
        self.out_lin = nn.Linear(embed_dim, 1)

    def forward(self, coords: torch.Tensor, cost_mat: torch.Tensor, duration_snapshots: torch.Tensor):
        """
        Args:
            coords: (B, N, 2)
            cost_mat: (B, N, N) - Distance matrix
            duration_snapshots: (B, K, N, N) - K duration snapshots
        """
        B, N, _ = cost_mat.shape
        
        # 1. Embed Distance
        dist_emb = self.dist_emb(cost_mat.unsqueeze(-1)) # (B, N, N, E)

        # 2. Embed Angle
        coords_expanded_1 = coords.unsqueeze(2)
        coords_expanded_2 = coords.unsqueeze(1)
        pairwise_diff = coords_expanded_1 - coords_expanded_2
        angles = torch.atan2(pairwise_diff[..., 1], pairwise_diff[..., 0])
        angle_emb = self.angle_emb(angles.unsqueeze(-1)) # (B, N, N, E)

        # 3. Embed Duration Snapshots
        # duration_snapshots: (B, K, N, N) -> (B, N, N, K)
        dur_snaps = duration_snapshots.permute(0, 2, 3, 1)
        dur_embs = []
        for k in range(self.num_snapshots):
            # Extract k-th snapshot: (B, N, N, 1)
            snap = dur_snaps[..., k : k+1]
            dur_embs.append(self.dur_emb(snap)) # (B, N, N, E)
        
        # 4. Gating
        # Concatenate all embeddings
        gate_in_list = [dist_emb, angle_emb] + dur_embs
        gate_in = torch.cat(gate_in_list, dim=-1) # (B, N, N, (2+K)*E)
        
        logits = self.gate(gate_in)
        g = F.softmax(logits / self.gate_temperature.exp(), dim=-1) # (B, N, N, 2+K)

        # 5. Weighted Sum
        fused_emb = g[..., [0]] * dist_emb + g[..., [1]] * angle_emb
        for k in range(self.num_snapshots):
            fused_emb += g[..., [2+k]] * dur_embs[k]

        # 6. Output Bias
        adapt_bias = self.out_lin(fused_emb).squeeze(-1) # (B, N, N)
        return adapt_bias


class AttnFree_Block_V2(AttnFree_Block):
    def __init__(
        self,
        embed_dim: int = 128,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "instance",
        parallel_gated_kwargs: Optional[dict] = None,
        num_snapshots: int = 3,
        **kwargs,
    ):
        # Initialize parent without nab_type logic (we override it)
        super(AttnFree_Block, self).__init__() 
        self.embed_dim = embed_dim
        self.alpha = nn.Parameter(torch.ones(1))
        self.attn_free = AFTFull(dim=embed_dim, hidden_dim=embed_dim)
        self.multi_head_combine = nn.Linear(embed_dim, embed_dim)

        # Use MultiSnapshotNAB
        self.neural_adaptive_bias = MultiSnapshotNAB(embed_dim, num_snapshots)

        self.feed_forward = TransformerFFN(
            embed_dim=embed_dim,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            parallel_gated_kwargs=parallel_gated_kwargs,
        )

        self.norm1 = Normalization(embed_dim=embed_dim, normalization=normalization)
        self.norm2 = Normalization(embed_dim=embed_dim, normalization=normalization)
        self.norm3 = Normalization(embed_dim=embed_dim, normalization=normalization)

    def forward(self, row_emb, col_emb, cost_mat, coords, duration_snapshots=None):
        row_emb = self.norm1(row_emb)
        col_emb = self.norm2(col_emb)

        # Use MultiSnapshotNAB
        adapt_bias = self.neural_adaptive_bias(coords, cost_mat, duration_snapshots) * self.alpha
        
        out_concat = self.attn_free(row_emb, y=col_emb, adapt_bias=adapt_bias)

        multi_head_out = self.multi_head_combine(out_concat)
        multi_head_out = self.norm3(multi_head_out)

        ffn_out = self.feed_forward(multi_head_out, row_emb)

        return ffn_out


class Attn_Free_Layer_V2(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "instance",
        parallel_gated_kwargs: Optional[dict] = None,
        num_snapshots: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.row_encoding_block = AttnFree_Block_V2(
            embed_dim=embed_dim,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            parallel_gated_kwargs=parallel_gated_kwargs,
            num_snapshots=num_snapshots,
            **kwargs,
        )
        self.col_encoding_block = AttnFree_Block_V2(
            embed_dim=embed_dim,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            parallel_gated_kwargs=parallel_gated_kwargs,
            num_snapshots=num_snapshots,
            **kwargs,
        )

    def forward(self, row_emb, col_emb, cost_mat, coords, duration_snapshots=None):
        row_emb_out = self.row_encoding_block(
            row_emb, col_emb, cost_mat, coords, duration_snapshots
        )
        
        if duration_snapshots is not None:
            # Transpose duration snapshots for column encoding: (B, K, N, N) -> (B, K, N, N) transpose last two dims
            trans_duration_snapshots = duration_snapshots.transpose(-2, -1)
        else:
            trans_duration_snapshots = None

        col_emb_out = self.col_encoding_block(
            col_emb, row_emb, cost_mat.transpose(1, 2), coords, trans_duration_snapshots
        )

        return row_emb_out, col_emb_out


class AttnFreeNetV2(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        feedforward_hidden: int = 512,
        num_layers: int = 3,
        normalization: Optional[str] = "instance",
        parallel_gated_kwargs: Optional[dict] = None,
        num_snapshots: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Attn_Free_Layer_V2(
                    embed_dim=embed_dim,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    parallel_gated_kwargs=parallel_gated_kwargs,
                    num_snapshots=num_snapshots,
                    **kwargs,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, row_emb, col_emb, cost_mat, coords, duration_snapshots=None):
        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, cost_mat, coords, duration_snapshots)

        return row_emb, col_emb
