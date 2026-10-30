"""DDA (Chen et al., NeurIPS 2025) arch — vendored from roy-ch/Dual-Data-Alignment
(Inference/models/dinov2_models.py + dinov2_models_lora.py; Apache-2.0, see
LICENSE).

DINOv2 ViT-L/14 (torch.hub facebookresearch/dinov2) with LoRA (rank 8,
alpha 1) on every attn.qkv / attn.proj / mlp.fc1 / mlp.fc2 Linear, and a
single-logit linear head on the normalized CLS token.

Changes vs. upstream: the hub backbone is built with pretrained=False (the
checkpoint holds every weight, incl. the frozen DINOv2 ones) and the two
wrapper classes are merged; state-dict keys are unchanged (``base_model.*``).
"""
import torch
import torch.nn as nn

from .lora import apply_lora_to_linear_layers

CHANNELS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}
LORA_TARGETS = ['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2']


class DINOv2Model(nn.Module):
    def __init__(self, name, num_classes=1):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', name, pretrained=False)
        self.fc = nn.Linear(CHANNELS[name], num_classes)

    def forward(self, x):
        features = self.model.forward_features(x)['x_norm_clstoken']
        return self.fc(features)


class DINOv2ModelWithLoRA(nn.Module):
    """Input: (B, 3, 336, 336) CLIP-normalized. Output: (B, 1) logit (fake > 0)."""

    def __init__(self, name="dinov2_vitl14", num_classes=1, lora_rank=8, lora_alpha=1.0):
        super().__init__()
        self.base_model = DINOv2Model(name=name, num_classes=num_classes)
        self.base_model.model = apply_lora_to_linear_layers(
            self.base_model.model, rank=lora_rank, alpha=lora_alpha,
            target_modules=LORA_TARGETS, trainable_orig=False,
        )

    def forward(self, x):
        return self.base_model(x)


def build_dda(checkpoint_path: str, device: str = "cpu") -> DINOv2ModelWithLoRA:
    model = DINOv2ModelWithLoRA()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
