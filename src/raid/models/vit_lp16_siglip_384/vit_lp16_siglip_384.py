import re
from typing import Union

import torch
import torch.nn as nn
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier

from timm.models import create_model

from .vit_lp16_siglip_384_postprocess import Vit_lp16_siglip_384PostProcess
from .vit_lp16_siglip_384_preprocess import Vit_lp16_siglip_384PreProcess


def vit_lp16_siglip_384(checkpoint_path: str, device: Union[str, torch.device] = "cpu"):
    model = create_model(
        "vit_large_patch16_siglip_384", 
        pretrained=False
    )
    if isinstance(model.head, nn.Identity):
        model_head_in_features = model.norm.weight.shape[0]
    else:
        model_head_in_features = model.head.in_features
    model.head = nn.Linear(in_features=model_head_in_features, out_features=1, bias=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['state_dict'])
    model = model.to(device)
    model.eval()
    preprocess = Vit_lp16_siglip_384PreProcess()
    postprocess = Vit_lp16_siglip_384PostProcess()
    wrapped_model = BasePytorchClassifier(
        model, preprocessing=preprocess, postprocessing=postprocess
    )
    return wrapped_model
