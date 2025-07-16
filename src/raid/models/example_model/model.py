from typing import Union

import torch
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier

from external.model import external_model_code

from .model_postprocess import ModelPostProcess
from .model_preprocess import ModelPreProcess


def model(checkpoint_path: str, device: Union[str, torch.device] = "cpu"):
    """
        Main model wrapper template for testing the RAID dataset 
        on an external model.
        
        In the case where external code is required to load the model,
        it should be placed in external.model, with the appropriate name
    """

    model = ...
        
    model.eval()
    model.to(device)
    preprocess = ModelPreProcess()
    postprocess = ModelPostProcess()
    wrapped_model = BasePytorchClassifier(
        model, preprocessing=preprocess, postprocessing=postprocess
    )
    return wrapped_model
