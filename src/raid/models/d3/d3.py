import torch
from typing import Union
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from external.d3.d3_arch import build_d3

from .d3_preprocess import D3PreProcess
from .d3_postprocess import D3PostProcess


def d3(checkpoint_path: str,
       device: Union[str, torch.device] = "cpu"):
    model = build_d3(checkpoint_path, device)
    return BasePytorchClassifier(
        model,
        preprocessing=D3PreProcess(),
        postprocessing=D3PostProcess(),
    )
