import torch
from typing import Union
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from external.effort.effort_arch import build_effort

from .effort_preprocess import EffortPreProcess
from .effort_postprocess import EffortPostProcess


def effort(checkpoint_path: str,
           device: Union[str, torch.device] = "cpu"):
    model = build_effort(checkpoint_path, device)
    return BasePytorchClassifier(
        model,
        preprocessing=EffortPreProcess(),
        postprocessing=EffortPostProcess(),
    )
