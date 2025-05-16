from typing import Union

import torch
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier

from external.cavia2024.networks.LaDeDa import LaDeDa9
from external.cavia2024.networks.Tiny_LaDeDa import tiny_ladeda

from .cavia2024_postprocess import Cavia2024PostProcess
from .cavia2024_preprocess import Cavia2024PreProcess

from collections import OrderedDict
from copy import deepcopy


def cavia2024(checkpoint_path: str, device: Union[str, torch.device] = "cpu"):
    if "Tiny" in checkpoint_path:
        features_dim = 8
        model = tiny_ladeda(num_classes=1)
    else:
        features_dim = 2048
        model = LaDeDa9(num_classes=1)

    model.fc = torch.nn.Linear(features_dim, 1)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=True)
    else:
        pretrained_dict = OrderedDict()
        for ki in state_dict.keys():
            pretrained_dict[ki] = deepcopy(state_dict[ki])
        model.load_state_dict(pretrained_dict, strict=True)
        
    model.eval()
    model.to(device)
    preprocess = Cavia2024PreProcess()
    postprocess = Cavia2024PostProcess()
    wrapped_model = BasePytorchClassifier(
        model, preprocessing=preprocess, postprocessing=postprocess
    )
    return wrapped_model
