"""Effort (Yan et al., ICML 2025 Oral) arch — vendored from
YZY-stack/Effort-AIGI-Detection (DeepfakeBench/training/detectors/effort_detector.py).

CLIP ViT-L/14 vision backbone with orthogonal-subspace (SVD) decomposition
applied to every self_attn Linear layer: each weight is split into a frozen
``weight_main`` (top r=1023 singular components) plus a trainable low-rank
residual (``U_residual, S_residual, V_residual``, rank 1) — the "Effort"
mechanism. A 2-class linear head sits on the CLIP pooled output.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import CLIPModel

R = 1024 - 1  # rank of the frozen main component (paper: ViT-L/14 1024-1)


def _clip_l14_path() -> str:
    # The shared HF_HOME cache on this cluster has metadata incompatible with
    # transformers==4.17.0's resolver (malformed-URL crash); snapshot_download
    # into a private cache sidesteps it cleanly.
    return snapshot_download("openai/clip-vit-large-patch14",
                              cache_dir=str(Path.home() / "hf_cache_local"))


class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r

        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.weight_original_fnorm = None
        min_dim = min(in_features, out_features)
        residual_rank = max(min_dim - r, 0)
        self.U_residual = nn.Parameter(torch.zeros(out_features, residual_rank))
        self.S_residual = nn.Parameter(torch.zeros(residual_rank))
        self.V_residual = nn.Parameter(torch.zeros(residual_rank, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.S_residual is not None and self.S_residual.numel() > 0:
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight = self.weight_main + residual_weight
        else:
            weight = self.weight_main
        return F.linear(x, weight, self.bias)


def _replace_with_svd_residual(module: nn.Linear, r: int) -> SVDResidualLinear:
    new_module = SVDResidualLinear(
        module.in_features, module.out_features, r,
        bias=module.bias is not None, init_weight=module.weight.data.clone(),
    )
    if module.bias is not None:
        new_module.bias.data.copy_(module.bias.data)
    new_module.weight_original_fnorm = torch.norm(module.weight.data, p="fro")
    return new_module


def _apply_svd_residual_to_self_attn(model: nn.Module, r: int) -> None:
    for name, child in model.named_children():
        if "self_attn" in name:
            for sub_name, sub_module in child.named_modules():
                if isinstance(sub_module, nn.Linear):
                    parent = child
                    parts = sub_name.split(".")
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], _replace_with_svd_residual(sub_module, r))
        else:
            _apply_svd_residual_to_self_attn(child, r)


class EffortModel(nn.Module):
    """CLIP ViT-L/14 + SVD-residual self_attn + 2-class linear head.
    Input: (B, 3, 224, 224) CLIP-normalized. Output: (B, 2) logits."""

    def __init__(self):
        super().__init__()
        clip_model = CLIPModel.from_pretrained(_clip_l14_path())
        vision_model = clip_model.vision_model
        _apply_svd_residual_to_self_attn(vision_model, r=R)
        self.backbone = vision_model
        self.head = nn.Linear(1024, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)["pooler_output"]
        return self.head(feat)


def build_effort(checkpoint_path: str, device: str = "cpu") -> EffortModel:
    model = EffortModel()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                  for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # position_ids is a deterministic (non-learned) buffer, fine to leave at its
    # freshly-constructed value; nothing else should be missing/unexpected.
    bad_missing = [k for k in missing if not k.endswith("position_ids")]
    if bad_missing or unexpected:
        raise RuntimeError(f"effort ckpt mismatch: missing={bad_missing[:5]} unexpected={list(unexpected)[:5]}")
    return model.to(device).eval()
