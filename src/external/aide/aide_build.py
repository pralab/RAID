"""AIDE build helpers: the batched 5-view DCT preprocessing module and the
checkpoint loader. Vendored/adapted from shilinyan99/AIDE (data/datasets.py's
TestDataset.__getitem__ eval pipeline): per image, ToTensor -> DCT_base_Rec_Module
(4 frequency-graded 32x32 patches) -> stack with the full image as a 5th view
-> Resize(256,256) + ImageNet-normalize, applied identically to all 5 views.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import transforms

from .aide_dct import DCT_base_Rec_Module
from .aide_model import AIDE_Model

_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class AIDEFiveView(nn.Module):
    """(B, 3, H, W) float RGB in [0,1] -> (B, 5, 3, 256, 256) ImageNet-normalized.

    The DCT patch-selection module (data/dct.py) is inherently per-image
    (unbatched argsort over unfolded patches), so this loops over the batch.
    The discrete patch *choice* is non-differentiable, but the selected patch
    pixel values still carry gradient (index_select/fold are differentiable).
    """

    def __init__(self):
        super().__init__()
        self.dct = DCT_base_Rec_Module()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        views = []
        for i in range(x.shape[0]):
            img = x[i]  # (3, H, W)
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(img)
            x_0 = img
            five = torch.stack([
                _resize_norm(x_minmin), _resize_norm(x_maxmax),
                _resize_norm(x_minmin1), _resize_norm(x_maxmax1),
                _resize_norm(x_0),
            ], dim=0)  # (5, 3, 256, 256)
            views.append(five)
        return torch.stack(views, dim=0)  # (B, 5, 3, 256, 256)


def _resize_norm(x: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False).squeeze(0)
    return _NORM(x)


def build_aide(checkpoint_path: str, device: str = "cpu") -> AIDE_Model:
    model = AIDE_Model(resnet_path=None, convnext_path=None)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"aide ckpt mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.to(device).eval()
