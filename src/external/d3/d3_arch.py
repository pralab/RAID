"""D3 (Yang et al., CVPR 2025) arch builder — frozen CLIP ViT-L/14 penultimate
features over (shuffled + original) views, TransformerAttention head -> 1 logit.
Vendored from BigAandSmallq/D3 (models/clip_models.py, paper defaults:
shuffle_times=1, original_times=1, patch_size=[14])."""
import torch

from .clip_models import CLIPModelShuffleAttentionPenultimateLayer


def build_d3_train(device="cpu"):
    """Fresh D3 for training: frozen CLIP backbone, trainable attention head."""
    model = CLIPModelShuffleAttentionPenultimateLayer(
        "ViT-L/14", num_classes=1, shuffle_times=1, patch_size=[14], original_times=1,
    )
    return model.to(device)


def build_d3(checkpoint_path, device="cpu"):
    """Eval loader: build + load released or GUARD-trained checkpoint."""
    model = build_d3_train(device="cpu")
    dat = torch.load(checkpoint_path, map_location="cpu")
    sd = dat.get("model", dat.get("state_dict", dat))
    sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("model.")]  # frozen CLIP may be absent
    if bad or unexpected:
        raise RuntimeError(f"d3 ckpt mismatch: missing={bad[:5]} unexpected={list(unexpected)[:5]}")
    return model.to(device).eval()
