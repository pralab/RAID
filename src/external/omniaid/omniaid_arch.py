"""OmniAID (Cheng et al., ICML 2026) arch — vendored from yunncheng/OmniAID
(reward/omniaid.py, the self-contained inference variant; MIT, see LICENSE).

CLIP ViT-L/14@336 with an SVD mixture-of-experts on every self_attn
projection: frozen weight_main + per-expert rank-r residuals, routed
per image by a gating MLP on the frozen CLIP pooled feature (hybrid mode:
top-k semantic experts + an always-on artifact expert). 2-class head.

Changes vs. upstream: the CLIP vision tower is built from its config instead
of downloading openai/clip-vit-large-patch14-336 (the checkpoint holds every
weight, incl. feature_extractor), so the SVD init is skipped and the
state dict is loaded strictly; forward returns the (B, 2) logits instead of
softmax[:, 1].
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from transformers import CLIPVisionConfig
from transformers.models.clip.modeling_clip import (
    CLIPVisionEmbeddings, CLIPMLP, CLIPVisionTransformer,
)

# openai/clip-vit-large-patch14-336 vision tower
CLIP_L14_336 = dict(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
                    num_attention_heads=16, image_size=336, patch_size=14,
                    hidden_act="quick_gelu", layer_norm_eps=1e-5)


class OmniAID(nn.Module):
    """Input: (B, 3, 336, 336) CLIP-normalized. Output: (B, 2) logits (fake = 1)."""

    def __init__(self, num_experts, rank_per_expert, moe_router_hidden_dim,
                 moe_top_k, is_hybrid=True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = moe_top_k
        self.is_hybrid = is_hybrid
        if is_hybrid:
            self.artifact_expert_idx = num_experts - 1
            gating_num_experts = num_experts - 1
        else:
            self.artifact_expert_idx = -1
            gating_num_experts = num_experts

        vision_config = CLIPVisionConfig(**CLIP_L14_336)
        self.feature_extractor = CLIPVisionTransformer(vision_config)
        self.hidden_size = vision_config.hidden_size
        r_main = vision_config.hidden_size - rank_per_expert

        self.embeddings = CLIPVisionEmbeddings(vision_config)
        self.ln_pre = nn.LayerNorm(vision_config.hidden_size)
        self.encoder_layers = nn.ModuleList([
            ViTMoELayer(vision_config, num_experts, r_main, rank_per_expert,
                        self.artifact_expert_idx)
            for _ in range(vision_config.num_hidden_layers)
        ])
        self.ln_post = nn.LayerNorm(self.hidden_size, eps=vision_config.layer_norm_eps)
        self.gating_network = GatingNetwork(
            input_dim=self.hidden_size, num_experts=gating_num_experts,
            hidden_dim=moe_router_hidden_dim, top_k=moe_top_k,
        )
        self.head = nn.Linear(self.hidden_size, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.size(0)

        with torch.no_grad():
            routing_features = self.feature_extractor(images).pooler_output

        hidden_states = self.embeddings(images)
        hidden_states = self.ln_pre(hidden_states)

        if self.is_hybrid:
            artifact_expert_indices = torch.full(
                (batch_size, 1), self.artifact_expert_idx,
                dtype=torch.long, device=hidden_states.device,
            )
            artifact_expert_gates = torch.ones(
                (batch_size, 1), dtype=hidden_states.dtype, device=hidden_states.device,
            )
            semantic_gating_outputs = self.gating_network(routing_features)
            gating_outputs = {
                'top_k_indices': torch.cat(
                    [semantic_gating_outputs['top_k_indices'], artifact_expert_indices], dim=1
                ),
                'top_k_gates': torch.cat(
                    [semantic_gating_outputs['top_k_gates'], artifact_expert_gates], dim=1
                ),
            }
        else:
            gating_outputs = self.gating_network(routing_features)

        for layer_module in self.encoder_layers:
            hidden_states = layer_module(hidden_states, gating_outputs=gating_outputs)[0]

        pooled_output = self.ln_post(hidden_states[:, 0, :])
        return self.head(pooled_output)


class ViTMoEAttention(nn.Module):
    def __init__(self, config: CLIPVisionConfig, num_experts: int, r_main: int, rank_per_expert: int, artifact_expert_idx: int):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.q_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx)
        self.k_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx)
        self.v_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx)
        self.out_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, gating_outputs, attention_mask=None):
        bsz, tgt_len, embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states, gating_outputs) * self.scale
        key_states = self.k_proj(hidden_states, gating_outputs)
        value_states = self.v_proj(hidden_states, gating_outputs)

        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = query_states.reshape(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.bmm(attn_weights, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output, gating_outputs)
        return attn_output, None


class ViTMoELayer(nn.Module):
    def __init__(self, config: CLIPVisionConfig, num_experts: int, r_main: int, rank_per_expert: int, artifact_expert_idx: int):
        super().__init__()
        self.self_attn = ViTMoEAttention(config, num_experts, r_main, rank_per_expert, artifact_expert_idx)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, gating_outputs, attention_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            gating_outputs=gating_outputs,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,)


class SVDMoeLinear(nn.Module):
    def __init__(self, in_features, out_features, r_main, num_experts, rank_per_expert, artifact_expert_idx, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r_main = r_main
        self.num_experts = num_experts
        self.rank_per_expert = rank_per_expert
        self.artifact_expert_idx = artifact_expert_idx

        self.register_buffer('weight_main', torch.zeros(out_features, in_features))
        self.register_buffer('U_r', torch.zeros(out_features, r_main))
        self.register_buffer('V_r', torch.zeros(r_main, in_features))

        self.U_experts = nn.ParameterList([nn.Parameter(torch.zeros(out_features, rank_per_expert)) for _ in range(num_experts)])
        self.S_experts = nn.ParameterList([nn.Parameter(torch.zeros(rank_per_expert)) for _ in range(num_experts)])
        self.V_experts = nn.ParameterList([nn.Parameter(torch.zeros(rank_per_expert, in_features)) for _ in range(num_experts)])

        self.register_buffer('weight_original_fnorm', torch.tensor(0.0))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor, gating_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        output_main = F.linear(x, self.weight_main, None)

        top_k_indices = gating_outputs['top_k_indices']
        top_k_gates = gating_outputs['top_k_gates']
        k = top_k_indices.size(1)

        expert_output = torch.zeros_like(output_main)
        U_all = torch.stack([p for p in self.U_experts])
        S_all = torch.stack([p for p in self.S_experts])
        V_all = torch.stack([p for p in self.V_experts])

        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(1)
            expert_output = expert_output.unsqueeze(1)

        for i in range(k):
            chosen_expert_indices = top_k_indices[:, i]
            gate_values = top_k_gates[:, i].unsqueeze(-1)
            U_batch = U_all[chosen_expert_indices]
            S_batch = S_all[chosen_expert_indices]
            V_batch = V_all[chosen_expert_indices]

            x_v = torch.bmm(x, V_batch.transpose(1, 2))
            x_v_s = x_v * S_batch.unsqueeze(1)
            current_expert_output = torch.bmm(x_v_s, U_batch.transpose(1, 2))
            expert_output += current_expert_output * gate_values.unsqueeze(-1)

        if original_dim == 2:
            expert_output = expert_output.squeeze(1)

        final_output = output_main + expert_output
        if self.bias is not None:
            final_output = final_output + self.bias
        return final_output


class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 256, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.network(x)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        return {'top_k_indices': top_k_indices, 'top_k_gates': top_k_gates}


def build_omniaid(checkpoint_path: str, config_path: str, device: str = "cpu") -> OmniAID:
    with open(config_path) as f:
        cfg = json.load(f)
    model = OmniAID(num_experts=cfg["num_experts"], rank_per_expert=cfg["rank_per_expert"],
                    moe_router_hidden_dim=cfg["moe_router_hidden_dim"],
                    moe_top_k=cfg["moe_top_k"], is_hybrid=cfg.get("is_hybrid", True))
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                  for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # position_ids is a deterministic (non-learned) buffer; nothing else may differ.
    bad_missing = [k for k in missing if not k.endswith("position_ids")]
    unexpected = [k for k in unexpected if not k.endswith("position_ids")]
    if bad_missing or unexpected:
        raise RuntimeError(f"omniaid ckpt mismatch: missing={bad_missing[:5]} unexpected={unexpected[:5]}")
    return model.to(device).eval()
