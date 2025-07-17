from typing import Union

import torch
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier

from external.chen2024.models import get_models

from .chen2024_postprocess import Chen2024PostProcess
from .chen2024_preprocess import Chen2024PreProcess


def chen2024(checkpoint_path: str, device: Union[str, torch.device] = "cpu"):
    if "convnext" in checkpoint_path:
        model_name = "convnext_base_in22k"
    else:
        model_name = "clip-ViT-L-14"
    model = get_models(model_name=model_name, embedding_size=1024)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)
    model.eval()
    preprocess = Chen2024PreProcess()
    postprocess = Chen2024PostProcess()
    wrapped_model = BasePytorchClassifier(
        model, preprocessing=preprocess, postprocessing=postprocess
    )
    return wrapped_model
