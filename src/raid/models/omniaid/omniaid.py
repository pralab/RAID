import torch
from huggingface_hub import hf_hub_download
from typing import Union
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from external.omniaid.omniaid_arch import build_omniaid

from .omniaid_preprocess import OmniAIDPreProcess
from .omniaid_postprocess import OmniAIDPostProcess


def omniaid(checkpoint_path: str,
            device: Union[str, torch.device] = "cpu"):
    config_path = hf_hub_download(repo_id="aimagelab/RAID_ckpt",
                                  filename="omniaid/config_omniaid_genimage_paper.json")
    model = build_omniaid(checkpoint_path, config_path, device)
    return BasePytorchClassifier(
        model,
        preprocessing=OmniAIDPreProcess(),
        postprocessing=OmniAIDPostProcess(),
    )
