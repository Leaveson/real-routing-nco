import math

from dataclasses import dataclass, fields
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from einops import rearrange
from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.autoregressive.decoder import AutoregressiveDecoder
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.models.nn.mlp import MLP
from rl4co.utils.ops import batchify, gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from .env_embeddings import env_context_embedding, env_dynamic_embedding

log = get_pylogger(__name__)


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    graph_context: Union[Tensor, float]
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor

    @property
    def fields(self):
        return tuple(getattr(self, x.name) for x in fields(self))

    def batchify(self, num_starts):
        new_embs = []
        for emb in self.fields:
            if isinstance(emb, Tensor) or isinstance(emb, TensorDict):
                new_embs.append(batchify(emb, num_starts))
            else:
                new_embs.append(emb)
        return PrecomputedCache(*new_embs)


class AAFMDecoder(AutoregressiveDecoder):
    """
    Auto-regressive decoder based on Kool et al. (2019): https://arxiv.org/abs/1803.08475.
    Given the environment state and the embeddings, compute the logits and sample actions autoregressively until
    all the environments in the batch have reached a terminal state.
    In this case we additionally have a `pre_decoder_hook` method that allows to precompute the embeddings before
    the decoder is called, which saves a lot of computation.


    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        env_name: Name of the environment used to initialize embeddings
        context_embedding: Context embedding module
        dynamic_embedding: Dynamic embedding module
        mask_inner: Whether to mask the inner loop
        out_bias_pointer_attn: Whether to use a bias in the pointer attention
        linear_bias: Whether to use a bias in the linear layer
        use_graph_context: Whether to use the graph context
        check_nan: Whether to check for nan values during decoding
        sdpa_fn: scaled_dot_product_attention function
        pointer: Module implementing the pointer logic (defaults to PointerAttention)
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
        self,
        embed_dim: int = 128,
        env_name: str = "rcvrp",
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = True,
        check_nan: bool = True,
        pointer: nn.Module = None,
        moe_kwargs: dict = None,
    ):
        super().__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.Wq_last = nn.Linear(embed_dim+1, embed_dim, bias=False)
        self.alpha1 = nn.Parameter(torch.tensor([1.0]))
        self.alpha2 = nn.Parameter(torch.tensor([1.0]))

        self.context_embedding = (
            env_context_embedding(self.env_name, {"embed_dim": embed_dim})
            if context_embedding is None
            else context_embedding
        )
        self.dynamic_embedding = (
            env_dynamic_embedding(self.env_name, {"embed_dim": embed_dim})
            if dynamic_embedding is None
            else dynamic_embedding
        )
        self.is_dynamic_embedding = (
            False if isinstance(self.dynamic_embedding, StaticEmbedding) else True
        )

        self.project_node_embeddings = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.pointer = AAFM_pointer(embed_dim)

    def _compute_q(self, cached: PrecomputedCache, td: TensorDict):
        node_embeds_cache = cached.node_embeddings
        graph_context_cache = cached.graph_context

        if td.dim() == 2 and isinstance(graph_context_cache, Tensor):
            graph_context_cache = graph_context_cache.unsqueeze(1)

        step_context = self.context_embedding(node_embeds_cache, td)
        glimpse_q = step_context + graph_context_cache
        # add seq_len dim if not present
        glimpse_q = glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q

        return glimpse_q

    def _compute_kvl(self, cached: PrecomputedCache, td: TensorDict):
        glimpse_k_stat, glimpse_v_stat, logit_k_stat = (
            cached.glimpse_key,
            cached.glimpse_val,
            cached.logit_key,
        )
        # Compute dynamic embeddings and add to static embeddings
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(td)
        glimpse_k = glimpse_k_stat + glimpse_k_dyn
        glimpse_v = glimpse_v_stat + glimpse_v_dyn
        logit_k = logit_k_stat + logit_k_dyn

        return glimpse_k, glimpse_v, logit_k

    def forward(
        self,
        td: TensorDict,
        cached: PrecomputedCache,
        num_starts: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the logits of the next actions given the current state

        Args:
            cache: Precomputed embeddings
            td: TensorDict with the current environment state
            num_starts: Number of starts for the multi-start decoding
        """

        has_dyn_emb_multi_start = self.is_dynamic_embedding and num_starts > 1

        # Handle efficient multi-start decoding
        if has_dyn_emb_multi_start:
            # if num_starts > 0 and we have some dynamic embeddings, we need to reshape them to [B*S, ...]
            # since keys and values are not shared across starts (i.e. the episodes modify these embeddings at each step)
            cached = cached.batchify(num_starts=num_starts)
        elif num_starts > 1:
            td = unbatchify(td, num_starts)

        glimpse_q = self._compute_q(cached, td)
        glimpse_k, glimpse_v, logit_k = self._compute_kvl(cached, td)
        log2_N = torch.log2(torch.tensor(td["distance_matrix"].shape[-1], dtype=td["distance_matrix"].dtype, device=td["distance_matrix"].device))

        # Compute logits
        mask = td["action_mask"]
        distance = gather_by_index(td["distance_matrix"], td["current_node"], dim=-2)
        adaptation_bias =  -1 * self.alpha1 * log2_N * distance

        logits = self.pointer(glimpse_q, glimpse_k, glimpse_v, adaptation_bias, mask)
        logits = logits.to(torch.float32)
        inductive_bias = -1 * self.alpha2 * log2_N * distance

        logits = torch.log(torch.exp(logits - inductive_bias) + 1e-6)
        # Now we need to reshape the logits and mask to [B*S,N,...] is num_starts > 1 without dynamic embeddings
        # note that rearranging order is important here

        if num_starts > 1 and not has_dyn_emb_multi_start:
            logits = rearrange(logits, "b s l -> (s b) l", s=num_starts)
            mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)

        return logits, mask

    def pre_decoder_hook(
        self, td, env, embeddings, num_starts: int = 0
    ) -> Tuple[TensorDict, RL4COEnvBase, PrecomputedCache]:
        """Precompute the embeddings cache before the decoder is called"""
        return td, env, self._precompute_cache(embeddings, num_starts=num_starts)

    def _precompute_cache(self, embeddings: Tuple[Tensor, Tensor], num_starts: int = 0):
        if isinstance(embeddings, tuple):
            row_emb, col_emb = embeddings
        else:
            row_emb = col_emb = embeddings

        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key,
        ) = self.project_node_embeddings(
            col_emb
        ).chunk(3, dim=-1)

        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=row_emb,
            graph_context=0,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key,
        )


class AAFM_pointer(nn.Module):
    """Calculate logits given query, key and value and logit key.
    This follows the pointer mechanism of Vinyals et al. (2015) (https://arxiv.org/abs/1506.03134).

    Note:
        With Flash Attention, masking is not supported

    Performs the following:
        1. Apply cross attention to get the heads
        2. Project heads to get glimpse
        3. Compute attention score between glimpse and logit key

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        mask_inner: whether to mask inner attention
        linear_bias: whether to use bias in linear projection
        check_nan: whether to check for NaNs in logits
        sdpa_fn: scaled dot product attention function (SDPA) implementation
    """

    def __init__(
        self,
        embed_dim: int,
        mask_inner: bool = True,
        out_bias: bool = False,
        check_nan: bool = True,
        sdpa_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super(AAFM_pointer, self).__init__()
        self.mask_inner = mask_inner
        self.hidden_dim = embed_dim
        self.project = nn.Linear(embed_dim, embed_dim, bias=out_bias)
        
    
    def forward(self, q, k, v, adaptation_bias, mask=None):
        B, T, _ = q.shape
        Q = q
        K = k
        V = v
        Q_sig = torch.sigmoid(Q)

        adapt_bias = torch.softmax(adaptation_bias, dim=-1)
        K = torch.softmax(K, dim=1)
        temp = torch.exp(adapt_bias) @ torch.mul(torch.exp(K), V)
        weighted = temp / (torch.exp(adapt_bias) @ torch.exp(K))

        Yt = torch.mul(Q_sig, weighted)
        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)

        # Compute logits
        logits = torch.matmul(Yt, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)

        return logits

