import torch
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from typing import Union
from external.corvi2023.get_method_here import get_method_here, def_model

from .corvi2023_preprocess import Corvi2023PreProcess
from .corvi2023_postprocess import Corvi2023PostProcess


def corvi2023(checkpoint_path: str, 
                device: Union[str, torch.device] = "cpu"):
    
    _, model_path, arch, norm_type, patch_size = get_method_here('Grag2021_progan', weights_path=checkpoint_path)
    model = def_model(arch, model_path, localize=False)
    model = model.to(device)
    model.eval()
    preprocess = Corvi2023PreProcess()
    postprocess = Corvi2023PostProcess()
    model = BasePytorchClassifier(
        model, preprocessing=preprocess, postprocessing=postprocess)
    return model
