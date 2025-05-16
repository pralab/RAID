import re
from typing import Union

import torch
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier

from external.koutlis2024.models import Model

from .koutlis2024_postprocess import Koutlis2024PostProcess
from .koutlis2024_preprocess import Koutlis2024PreProcess


def get_our_trained_model_mod(checkpoint_path, device):
    ncls = (
        re.search(r"model_(.*)_trainable.pth", checkpoint_path)
        .group(1)
        .removesuffix("class")
    )
    if ncls == "1":
        nproj = 4
        proj_dim = 1024
    elif ncls == "2":
        nproj = 4
        proj_dim = 128
    elif ncls == "4":
        nproj = 2
        proj_dim = 1024
    elif ncls == "ldm":
        nproj = 4
        proj_dim = 1024

    model = Model(
        backbone=("ViT-L/14", 1024),
        nproj=nproj,
        proj_dim=proj_dim,
        device=device,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    for name in state_dict:
        exec(
            f'model.{name.replace(".", "[", 1).replace(".", "].", 1)} = torch.nn.Parameter(state_dict["{name}"])'
        )
    return model


def koutlis2024(checkpoint_path: str, device: Union[str, torch.device] = "cpu"):
    model = get_our_trained_model_mod(checkpoint_path=checkpoint_path, device=device)
    model = model.to(device)
    model.eval()
    preprocess = Koutlis2024PreProcess()
    postprocess = Koutlis2024PostProcess()
    wrapped_model = BasePytorchClassifier(
        model, preprocessing=preprocess, postprocessing=postprocess
    )
    return wrapped_model
