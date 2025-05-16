import torch
from secmlt.models.data_processing.data_processing import DataProcessing


class Wang2020PostProcess(DataProcessing):
    def _process(self, x: torch.Tensor) -> torch.Tensor:
        #out = x.sigmoid() - 0.5
        #return torch.cat([-out, out], dim=-1)
        return torch.cat([-x,x], dim=-1)

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented
