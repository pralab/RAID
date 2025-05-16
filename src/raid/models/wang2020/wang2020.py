from typing import Union

import torch
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier

from external.wang2020.resnet import resnet50

from .wang2020_postprocess import Wang2020PostProcess
from .wang2020_preprocess import Wang2020PreProcess


def wang2020(checkpoint_path: str, device: Union[str, torch.device] = "cpu"):
    model = resnet50(num_classes=1)
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu"
    )
    if "model" in state_dict:
        model.load_state_dict(state_dict["model"])
    elif "state_dict" in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        raise ValueError("Dictionary with unexpected key")
    
    model = model.to(device)
    model.eval()
    preprocess = Wang2020PreProcess()
    postprocess = Wang2020PostProcess()
    wrapped_model = BasePytorchClassifier(
        model, preprocessing=preprocess, postprocessing=postprocess
    )
    return wrapped_model
