import torch
from typing import Union
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from external.aide.aide_build import build_aide

from .aide_preprocess import AIDEPreProcess
from .aide_postprocess import AIDEPostProcess


def aide(checkpoint_path: str,
         device: Union[str, torch.device] = "cpu"):
    model = build_aide(checkpoint_path, device)
    return BasePytorchClassifier(
        model,
        preprocessing=AIDEPreProcess(),
        postprocessing=AIDEPostProcess(),
    )
