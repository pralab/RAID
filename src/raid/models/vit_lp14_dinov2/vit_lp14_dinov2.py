import re
from typing import Union

import torch
import torch.nn as nn
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier

from timm.models import create_model

from .vit_lp14_dinov2_postprocess import Vit_lp14_dinov2PostProcess
from .vit_lp14_dinov2_preprocess import Vit_lp14_dinov2PreProcess


def vit_lp14_dinov2(checkpoint_path: str, device: Union[str, torch.device] = "cpu"):
    model = create_model(
        "vit_large_patch14_dinov2.lvd142m", 
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
    preprocess = Vit_lp14_dinov2PreProcess()
    postprocess = Vit_lp14_dinov2PostProcess()
    wrapped_model = BasePytorchClassifier(
        model, preprocessing=preprocess, postprocessing=postprocess
    )
    return wrapped_model
