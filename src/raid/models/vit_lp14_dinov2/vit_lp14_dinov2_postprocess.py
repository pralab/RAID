import torch
from secmlt.models.data_processing.data_processing import DataProcessing


class Vit_lp14_dinov2PostProcess(DataProcessing):
    def _process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([-x,x], dim=-1)

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented
