import torch
from typing import Union
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from external.dda.dda_arch import build_dda

from .dda_preprocess import DDAPreProcess
from .dda_postprocess import DDAPostProcess


def dda(checkpoint_path: str,
        device: Union[str, torch.device] = "cpu"):
    model = build_dda(checkpoint_path, device)
    return BasePytorchClassifier(
        model,
        preprocessing=DDAPreProcess(),
        postprocessing=DDAPostProcess(),
    )
