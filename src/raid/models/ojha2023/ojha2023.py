import torch
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from typing import Union
from timm.models import create_model
from external.ojha2023.models import get_model
from .ojha2023_preprocess import Ojha2023PreProcess
from .ohja2023_postprocess import Ojha2023PostProcess


def ojha2023(checkpoint_path: str, device: Union[str, torch.device] = "cpu"):
    # here manage the load of the entire model
    if "model_best.pth.tar" in checkpoint_path:
        model = create_model("vit_large_patch14_clip_224.openai", pretrained=False)
        model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=1, bias=True)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict['state_dict'])
    else:
        model = get_model('CLIP:ViT-L/14')
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.fc.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    preprocess = Ojha2023PreProcess()
    postprocess = Ojha2023PostProcess()
    model = BasePytorchClassifier(
        model, preprocessing=preprocess, postprocessing=postprocess
    )
    return model
