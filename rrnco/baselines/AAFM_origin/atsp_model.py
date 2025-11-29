import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rl4co.models.common.constructive.autoregressive.policy import AutoregressivePolicy
from rl4co.models.common.constructive.autoregressive.decoder import AutoregressiveDecoder
from rl4co.models.common.constructive.autoregressive.encoder import AutoregressiveEncoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class AAFMATSPPolicy(AutoregressivePolicy):
    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        env_name: str = "atsp",
        **kwargs,
    ):
        encoder = AAFMATSPEncoder(
            embedding_dim=embed_dim,
            encoder_layer_num=num_encoder_layers,
            **kwargs
        )
        decoder = AAFMATSPDecoder(
            embedding_dim=embed_dim,
            **kwargs
        )
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            **kwargs
        )

    def forward(
        self,
        td: TensorDict,
        env = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = True,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        **decoding_kwargs,
    ) -> dict:
        from rl4co.utils.decoding import get_log_likelihood, get_decoding_strategy
        from rl4co.utils.ops import calculate_entropy
        from rl4co.envs import get_env

        # Encoder: get encoder output and initial embeddings from initial state
        hidden = self.encoder(td)

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        decode_strategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            logits, mask = self.decoder(td, hidden, num_starts)
            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) duing decoding"
                )
                break
        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        # Output dictionary construction
        if calc_reward:
            if env.normalize:
                real_distances, normalized_distances = env.get_reward(td, actions)
                td.set("reward", real_distances)
            else:
                td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
        }
        if calc_reward and env.normalize:
            outdict["normalized_reward"] = normalized_distances
        if return_actions:
            outdict["actions"] = actions
        if return_entropy:
            outdict["entropy"] = calculate_entropy(logprobs)
        if return_hidden:
            outdict["hidden"] = hidden

        return outdict

class AAFMATSPEncoder(AutoregressiveEncoder):
    def __init__(self, embedding_dim=128, neighbors=20, encoder_layer_num=3, **kwargs):
        super().__init__()
        self.model_params = {
            'embedding_dim': embedding_dim,
            'neighbors': neighbors,
            'encoder_layer_num': encoder_layer_num,
            'ff_hidden_dim': kwargs.get('feedforward_hidden', 512),
            'normalization': "instance"
        }
        self.k = neighbors
        self.embedding_row = nn.Linear(self.k, embedding_dim)
        self.embedding_col = nn.Linear(self.k, embedding_dim)
        self.encoder = ATSP_Encoder(**self.model_params)

    def forward(self, td):
        problems = td["distance_matrix"]
        if torch.isnan(problems).any():
            raise ValueError("NaN in distance_matrix input to encoder")

        # problems.shape: (batch, node, node)
        
        # Assuming log_scale is 1.0 if not present, or we can learn it/pass it.
        # The original code had log_scale in reset_state.
        # We will assume it's 1.0 for now or extract from td if available.
        log_scale = td.get("log_scale", torch.tensor(1.0, device=problems.device))

        batch_size, num_loc, _ = problems.shape
        
        # Adjust k if num_loc is smaller than k
        k = min(self.k, num_loc)

        problems_k_row = torch.topk(problems, k=k, dim=2, largest=False, sorted=True).values
        problems_k_col = torch.topk(problems.transpose(1, 2), k=k, dim=2, largest=False, sorted=True).values
        # shape: (batch, node, k)
        
        # If k < self.k, we might need to pad or adjust linear layer. 
        # But for now assuming num_loc >= self.k usually.
        # If num_loc < self.k, the Linear layer will fail because input dim is k.
        # We should probably pad if needed, or assume fixed k.
        # For safety, let's pad if needed.
        if k < self.k:
            problems_k_row = F.pad(problems_k_row, (0, self.k - k))
            problems_k_col = F.pad(problems_k_col, (0, self.k - k))

        row_emb = self.embedding_row(problems_k_row)
        col_emb = self.embedding_col(problems_k_col)
        # shape: (batch, node, embedding)

        encoded_row, encoded_col = self.encoder(row_emb, col_emb, problems, log_scale)
        # encoded_nodes.shape: (batch, node, embedding)

        # Return tuple as hidden state
        return (encoded_row, encoded_col, log_scale), None

class AAFMATSPDecoder(AutoregressiveDecoder):
    def __init__(self, embedding_dim=128, **kwargs):
        super().__init__()
        self.model_params = {
            'embedding_dim': embedding_dim,
            'sqrt_embedding_dim': embedding_dim ** 0.5,
            'logit_clipping': kwargs.get('tanh_clipping', 10.0)
        }
        
        self.Wq_0 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.alpha1 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

    def forward(self, td, hidden, num_starts):
        # print("Decoder TD batch size:", td.batch_size)
        
        # Handle hidden state unpacking
        # hidden might be ((row, col, log), None) from encoder
        if isinstance(hidden, tuple) and len(hidden) == 2 and hidden[1] is None:
            hidden = hidden[0]
            
        encoded_row, encoded_col, log_scale = hidden
        
        # Check for batch size mismatch (e.g. multi-start)
        # current_node is retrieved later, but we need to check batch size now.
        # We can peek at current_node from td
        current_node_temp = td["current_node"]
        if current_node_temp.size(0) > encoded_row.size(0):
            num_repeats = current_node_temp.size(0) // encoded_row.size(0)
            encoded_row = encoded_row.repeat_interleave(num_repeats, dim=0)
            encoded_col = encoded_col.repeat_interleave(num_repeats, dim=0)
            if log_scale.dim() > 0:
                log_scale = log_scale.repeat_interleave(num_repeats, dim=0)

        
        # Prepare keys and values (cached per instance)
        # In rl4co, forward is called at each step.
        # We can recompute K, V every time or cache it.
        # Since encoded_col is static, K and V are static.
        # But AutoregressiveDecoder doesn't easily support stateful caching across steps without modifying td.
        # We will recompute for simplicity, or we can check if we can optimize.
        # Given the original code had set_kv, let's just compute it.
        
        k = self.Wk(encoded_col)
        v = self.Wv(encoded_col)
        single_head_key = encoded_col.transpose(1, 2)
        
        current_node = td["current_node"]
        if current_node.dim() == 1:
            current_node = current_node.unsqueeze(1)
        # current_node shape: (batch, pomo) or (batch, 1)
        
        # Get q1 (from first node)
        # If current_node is the first node (step 0), we need to determine what the "first node" is.
        # In rl4co, td["first_node"] is available.
        
        # We need to gather the encoding of the first node.
        first_node = td["first_node"] # (batch, 1) usually, or (batch, pomo)
        
        # If num_starts > 1, first_node might be different per pomo path.
        # encoded_row: (batch, num_loc, embed)
        
        encoded_first_row = _get_encoding(encoded_row, first_node)
        q1 = self.Wq_1(encoded_first_row)
        
        # Get q0 (from current node)
        encoded_current_row = _get_encoding(encoded_row, current_node)
        q0 = self.Wq_0(encoded_current_row)
        
        q = q1 + q0
        
        # Calculate current distance (cost matrix row for current node)
        # td["distance_matrix"]: (batch, num_loc, num_loc)
        # We need distance from current_node to all other nodes.
        # cur_dist: (batch, pomo, num_loc)
        
        batch_size = encoded_row.size(0)
        num_loc = encoded_row.size(1)
        pomo_size = current_node.size(1)
        
        # Gather distance matrix rows corresponding to current_node
        # distance_matrix: (batch, num_loc, num_loc)
        # current_node: (batch, pomo)
        # We want (batch, pomo, num_loc)
        
        # Expand distance matrix to (batch, pomo, num_loc, num_loc) ? No too big.
        # We can gather.
        
        # distance_matrix: (batch, num_loc, num_loc)
        # index: (batch, pomo) -> expand to (batch, pomo, num_loc)
        
        d_mat = td["distance_matrix"] # (batch, N, N)
        # We want d_mat[b, current_node[b, p], :]
        
        # (batch, 1, N, N)
        d_mat_expanded = d_mat.unsqueeze(1) 
        # (batch, pomo, 1, 1)
        idx_expanded = current_node.unsqueeze(-1).unsqueeze(-1).expand(batch_size, pomo_size, 1, num_loc)
        
        # This gather is tricky because we want to gather rows.
        # Let's use batched index select logic.
        
        # Reshape to (batch * pomo, ...) if needed, or use gather.
        # d_mat: (batch, N, N)
        # current_node: (batch, pomo)
        
        # We can use gather on dim 1 of d_mat.
        # d_mat: (batch, N, N)
        # index: (batch, pomo, N)
        gather_idx = current_node.unsqueeze(-1).expand(batch_size, pomo_size, num_loc)
        
        # But d_mat is (batch, N, N). We need to gather along dim 1.
        # We need to broadcast d_mat to (batch, pomo, N, N) first? No.
        # We can just use gather if we treat batch dim carefully.
        
        # Actually, simpler:
        # d_mat is (batch, N, N).
        # We want to select rows.
        # output: (batch, pomo, N)
        
        # d_mat.gather(1, index)
        # index must be (batch, pomo, N)
        
        cur_cost_mat = d_mat.gather(1, gather_idx)
        
        # ninf_mask
        # td["action_mask"] is 1 for available, 0 for unavailable.
        # We want 0 for available, -inf for unavailable.
        mask = td["action_mask"] # (batch, pomo, N) or (batch, N)
        
        # If mask is (batch, N), expand to (batch, pomo, N)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).expand(batch_size, pomo_size, num_loc)
            
        ninf_mask = torch.zeros_like(mask, dtype=torch.float32)
        ninf_mask[~mask] = float('-inf')
        
        # AAFM Logic
        alpha_relation_bias = -1 * self.alpha1 * log_scale * cur_cost_mat
        
        # Expand k, v, q to match pomo size if needed?
        # k, v: (batch, N, embed) -> (batch, 1, N, embed) -> (batch, pomo, N, embed) ?
        # Wait, AAFM uses matmul.
        # q: (batch, pomo, embed)
        # k: (batch, N, embed)
        # v: (batch, N, embed)
        # adaptation_bias: (batch, pomo, N)
        
        # We need to handle the dimensions in adaptation_attention_free_module
        # It expects q: (batch, n, embed), k: (batch, m, embed).
        # Here n=pomo, m=num_loc.
        # But k, v are (batch, N, embed). We need to broadcast over pomo?
        # Or does the function handle it?
        # The function does: bias = exp(bias) @ (exp(k) * v)
        # bias is (batch, pomo, N).
        # exp(k) * v is (batch, N, embed).
        # (batch, pomo, N) @ (batch, N, embed) -> (batch, pomo, embed).
        # This works!
        
        AAFM_OUT = adaptation_attention_free_module(q, k, v,
                                                    adaptation_bias=alpha_relation_bias,
                                                    ninf_mask=ninf_mask)
        
        # Single-Head Attention
        # AAFM_OUT: (batch, pomo, embed)
        # single_head_key: (batch, embed, N)
        
        score = torch.matmul(AAFM_OUT, single_head_key)
        # shape: (batch, pomo, N)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        score_scaled = score_scaled - self.alpha2 * log_scale * cur_cost_mat
        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask
        
        if score_masked.dim() == 3 and score_masked.size(1) == 1:
            score_masked = score_masked.squeeze(1)
            mask = mask.squeeze(1)
        
        # Return logits and mask
        return score_masked, mask


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    if node_index_to_pick.dim() == 1:
        node_index_to_pick = node_index_to_pick.unsqueeze(1)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER (Copied and adapted)
########################################
class ATSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        encoder_layer_num = model_params['encoder_layer_num']
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, row_emb, col_emb, cost_mat, log_scale):
        # col_emb.shape: (batch, col_cnt, embedding)
        # row_emb.shape: (batch, row_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        cost_mat = -1 * log_scale * cost_mat

        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, cost_mat)

        return row_emb, col_emb


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.row_encoding_block = EncodingBlock(**model_params)
        self.col_encoding_block = EncodingBlock(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        row_emb_out = self.row_encoding_block(row_emb, col_emb, cost_mat)
        col_emb_out = self.col_encoding_block(col_emb, row_emb, cost_mat.transpose(1, 2))

        return row_emb_out, col_emb_out


class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        normalization = self.model_params['normalization']

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

        self.alpha = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

    def forward(self, row_emb, col_emb, cost_mat):
        q = self.Wq(row_emb)
        k = self.Wk(col_emb)
        v = self.Wv(col_emb)

        alpha_relation_bias = self.alpha * cost_mat
        out_aft = adaptation_attention_free_module(q, k, v, adaptation_bias=alpha_relation_bias)

        out1 = self.add_n_normalization_1(row_emb, out_aft)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3


def adaptation_attention_free_module(q, k, v, adaptation_bias, ninf_mask=None):
    if torch.isnan(q).any(): raise ValueError("NaN in q")
    if torch.isnan(k).any(): raise ValueError("NaN in k")
    if torch.isnan(v).any(): raise ValueError("NaN in v")
    if torch.isnan(adaptation_bias).any(): raise ValueError("NaN in adaptation_bias")

    sigmoid_q = torch.sigmoid(q)

    if ninf_mask is not None:
        adaptation_bias = adaptation_bias + ninf_mask

    # Refactored to match AFTFull in attn_freenet.py
    adaptation_bias = torch.softmax(adaptation_bias, dim=-1)
    k = torch.softmax(k, dim=1)

    # Handle broadcasting for batch matrix multiplication
    # q: (batch, n, d)
    # k: (batch, m, d)
    # v: (batch, m, d)
    # adaptation_bias: (batch, n, m)
    
    # bias = exp(adaptation_bias) @ (exp(k) * v)
    # (batch, n, m) @ (batch, m, d) -> (batch, n, d)
    
    bias = torch.exp(adaptation_bias) @ torch.mul(torch.exp(k), v)
    a_k = torch.exp(adaptation_bias) @ torch.exp(k)

    weighted = bias / a_k
    if torch.isinf(bias).any() or torch.isinf(a_k).any():
        weighted = torch.nan_to_num_(bias) / torch.nan_to_num_(a_k)
    if torch.isnan(weighted).any():
        torch.nan_to_num_(weighted)

    out = torch.mul(sigmoid_q, weighted)
    return out


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        back_trans = normalized.transpose(1, 2)
        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))
